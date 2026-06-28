from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import io
from datetime import datetime

app = FastAPI(title="Zomato Portfolio API")

# Allow your React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load data once at startup ──────────────────────────────────────────────────
orders      = pd.read_csv("data/orders.csv")
users       = pd.read_csv("data/users.csv")
restaurants = pd.read_csv("data/restaurants.csv")
menu        = pd.read_csv("data/menu.csv")
food        = pd.read_csv("data/food.csv")
model       = joblib.load("models/churn_model.pkl")

# Clean orders: parse date, convert sales_amount to numeric
orders["order_date"]   = pd.to_datetime(orders["order_date"], errors="coerce")
orders["sales_amount"] = pd.to_numeric(orders["sales_amount"], errors="coerce")
SNAPSHOT_DATE = orders["order_date"].max()


# ── /insights ─────────────────────────────────────────────────────────────────
@app.get("/insights")
def get_insights():
    """City spend, cuisine breakdown, busiest days, veg vs non-veg ratings."""

    # 1. Average spend per city
    city_spend = (
        orders.merge(restaurants[["id", "city"]], left_on="r_id", right_on="id")
              .groupby("city")["sales_amount"]
              .mean()
              .round(2)
              .sort_values(ascending=False)
              .head(10)
              .reset_index()
              .rename(columns={"sales_amount": "avg_spend"})
              .to_dict(orient="records")
    )

    # 2. Busiest day of week
    orders_copy = orders.copy()
    orders_copy["day"] = orders_copy["order_date"].dt.day_name()
    busy_days = (
        orders_copy.groupby("day").size()
                   .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
                   .reset_index()
                   .rename(columns={0: "order_count"})
                   .to_dict(orient="records")
    )

    # 3. Veg vs non-veg restaurant ratings
    rest_food = (
        menu.merge(food[["f_id","veg_or_non_veg"]], on="f_id")
            .merge(restaurants[["id","rating"]], left_on="r_id", right_on="id")
    )
    rest_food["rating"] = pd.to_numeric(rest_food["rating"], errors="coerce")
    veg_ratings = (
        rest_food.groupby("veg_or_non_veg")["rating"]
                 .mean()
                 .round(2)
                 .reset_index()
                 .rename(columns={"rating": "avg_rating"})
                 .to_dict(orient="records")
    )

    # 4. Top cuisines by order count
    top_cuisines = (
        orders.merge(restaurants[["id","cuisine"]], left_on="r_id", right_on="id")
              .assign(cuisine=lambda df: df["cuisine"].str.split(","))
              .explode("cuisine")
              .assign(cuisine=lambda df: df["cuisine"].str.strip())
              .groupby("cuisine").size()
              .sort_values(ascending=False)
              .head(10)
              .reset_index()
              .rename(columns={0: "order_count"})
              .to_dict(orient="records")
    )

    return {
        "city_spend": city_spend,
        "busy_days": busy_days,
        "veg_ratings": veg_ratings,
        "top_cuisines": top_cuisines,
    }


# ── /rfm-segments ─────────────────────────────────────────────────────────────
@app.get("/rfm-segments")
def get_rfm(segment: str = Query(default=None)):
    """Return RFM-scored customers. Optionally filter by segment label."""

    rfm = (
        orders.groupby("user_id")
              .agg(
                  recency  = ("order_date",   lambda x: (SNAPSHOT_DATE - x.max()).days),
                  frequency= ("user_id",      "count"),
                  monetary = ("sales_amount", "sum"),
              )
              .reset_index()
    )

    # Score 1-5 (5 = best)
    rfm["r_score"] = pd.qcut(rfm["recency"],   5, labels=[5,4,3,2,1], duplicates="drop").astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"], 5, labels=[1,2,3,4,5], duplicates="drop").astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"],  5, labels=[1,2,3,4,5], duplicates="drop").astype(int)
    rfm["rfm_score"] = rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)

    def label(row):
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        if r >= 4 and f >= 4 and m >= 4: return "Champion"
        if r >= 3 and f >= 3:            return "Loyal"
        if r <= 2 and f >= 3:            return "At Risk"
        if r == 1 and f == 1:            return "Lost"
        if r >= 4 and f <= 2:            return "New Customer"
        return "Potential Loyalist"

    rfm["segment"] = rfm.apply(label, axis=1)

    # Summary stats per segment
    summary = (
        rfm.groupby("segment")
           .agg(
               customer_count = ("user_id",   "count"),
               total_revenue  = ("monetary",  "sum"),
               avg_recency    = ("recency",   "mean"),
               avg_frequency  = ("frequency", "mean"),
           )
           .round(1)
           .reset_index()
           .to_dict(orient="records")
    )

    # Optionally return individual customers for a segment
    customers = []
    if segment:
        seg_df = rfm[rfm["segment"] == segment].merge(
            users[["user_id","name","occupation","monthly_income"]], on="user_id", how="left"
        )
        customers = seg_df.head(200).to_dict(orient="records")

    return {"summary": summary, "customers": customers}


# ── /predict-churn ─────────────────────────────────────────────────────────────
@app.post("/predict-churn")
async def predict_churn(file: UploadFile = File(...)):
    """
    Upload a CSV of user_ids. Returns churn probability + revenue at risk
    for each user, sorted by revenue impact.

    Expected CSV columns: user_id  (we join the rest from our data)
    """
    contents = await file.read()
    upload_df = pd.read_csv(io.BytesIO(contents))

    # Build RFM features for uploaded users
    rfm = (
        orders[orders["user_id"].isin(upload_df["user_id"])]
              .groupby("user_id")
              .agg(
                  recency  = ("order_date",   lambda x: (SNAPSHOT_DATE - x.max()).days),
                  frequency= ("user_id",      "count"),
                  monetary = ("sales_amount", "sum"),
              )
              .reset_index()
    )

    if rfm.empty:
        return {"error": "No matching user_ids found in orders data."}

    features = rfm[["recency", "frequency", "monetary"]]
    rfm["churn_probability"] = model.predict_proba(features)[:, 1].round(3)
    rfm["revenue_at_risk"]   = (rfm["churn_probability"] * rfm["monetary"]).round(2)
    rfm["churn_predicted"]   = (rfm["churn_probability"] >= 0.5).astype(int)

    result = (
        rfm.merge(users[["user_id","name","occupation","monthly_income"]], on="user_id", how="left")
           .sort_values("revenue_at_risk", ascending=False)
           .head(500)
           .to_dict(orient="records")
    )

    summary = {
        "total_users_scored": len(rfm),
        "predicted_churners": int(rfm["churn_predicted"].sum()),
        "total_revenue_at_risk": round(float(rfm["revenue_at_risk"].sum()), 2),
    }

    return {"summary": summary, "at_risk_customers": result}


# ── /restaurants ──────────────────────────────────────────────────────────────
@app.get("/restaurants")
def get_restaurants(
    city:    str = Query(default=None),
    cuisine: str = Query(default=None),
    veg_only: bool = Query(default=False),
):
    """Filter restaurants. Used by the map and SQL insights page."""

    df = restaurants.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["cost"]   = pd.to_numeric(df["cost"],   errors="coerce")

    if city:
        df = df[df["city"].str.lower() == city.lower()]
    if cuisine:
        df = df[df["cuisine"].str.contains(cuisine, case=False, na=False)]
    if veg_only:
        veg_restaurant_ids = (
            menu.merge(food[food["veg_or_non_veg"] == "Veg"]["f_id"], on="f_id")["r_id"].unique()
        )
        df = df[df["id"].isin(veg_restaurant_ids)]

    # Top cities list (for dropdown)
    cities = sorted(restaurants["city"].dropna().unique().tolist())

    return {
        "total": len(df),
        "cities": cities,
        "restaurants": df[["id","name","city","rating","cost","cuisine","address"]]
                        .dropna(subset=["rating"])
                        .sort_values("rating", ascending=False)
                        .head(200)
                        .to_dict(orient="records"),
    }


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Zomato Portfolio API is running"}
