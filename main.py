from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import io
import os
import urllib.request

app = FastAPI(title="Zomato Portfolio API v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Download files from Hugging Face if not present ───────────────────────────
HF_BASE = "https://huggingface.co/datasets/keertana-sri/zomato-data/resolve/main"

FILES = {
    "data/orders.csv":       f"{HF_BASE}/orders.csv",
    "data/users.csv":        f"{HF_BASE}/users.csv",
    "data/restaurants.csv":  f"{HF_BASE}/restaurant.csv",
    "data/menu.csv":         f"{HF_BASE}/menu.csv",
    "data/food.csv":         f"{HF_BASE}/food.csv",
    "models/churn_model.pkl": f"{HF_BASE}/churn_model%20(1).pkl",
}

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

for local_path, url in FILES.items():
    if not os.path.exists(local_path):
        print(f"Downloading {local_path}...")
        urllib.request.urlretrieve(url, local_path)
        print(f"Done: {local_path}")

# ── Load data ─────────────────────────────────────────────────────────────────
orders      = pd.read_csv("data/orders.csv")
users       = pd.read_csv("data/users.csv")
restaurants = pd.read_csv("data/restaurants.csv")
menu        = pd.read_csv("data/menu.csv", low_memory=False)
food        = pd.read_csv("data/food.csv")
model       = joblib.load("models/churn_model.pkl")

orders["order_date"]   = pd.to_datetime(orders["order_date"], errors="coerce")
orders["sales_amount"] = pd.to_numeric(orders["sales_amount"], errors="coerce")
SNAPSHOT_DATE = orders["order_date"].max()


# ── /insights ─────────────────────────────────────────────────────────────────
@app.get("/insights")
def get_insights():
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

    orders_copy = orders.copy()
    orders_copy["day"] = orders_copy["order_date"].dt.day_name()
    busy_days = (
        orders_copy.groupby("day").size()
                   .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
                   .reset_index()
                   .rename(columns={0: "order_count"})
                   .to_dict(orient="records")
    )

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
    rfm = (
        orders.groupby("user_id")
              .agg(
                  recency  = ("order_date",   lambda x: (SNAPSHOT_DATE - x.max()).days),
                  frequency= ("user_id",      "count"),
                  monetary = ("sales_amount", "sum"),
              )
              .reset_index()
    )

    def rank_score(series, ascending=True):
        ranked = series.rank(method="first", ascending=ascending)
        scaled = ((ranked - 1) / (len(ranked) - 1) * 4 + 1).round().astype(int)
        return scaled.clip(1, 5)

    rfm["r_score"] = rank_score(rfm["recency"],   ascending=False)
    rfm["f_score"] = rank_score(rfm["frequency"], ascending=True)
    rfm["m_score"] = rank_score(rfm["monetary"],  ascending=True)

    def label(row):
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        if r >= 4 and f >= 4 and m >= 4: return "Champion"
        if r >= 3 and f >= 3:            return "Loyal"
        if r <= 2 and f >= 3:            return "At Risk"
        if r == 1 and f == 1:            return "Lost"
        if r >= 4 and f <= 2:            return "New Customer"
        return "Potential Loyalist"

    rfm["segment"] = rfm.apply(label, axis=1)

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
           .fillna(0)
           .to_dict(orient="records")
    )

    customers = []
    if segment:
        seg_df = rfm[rfm["segment"] == segment].merge(
            users[["user_id","name","occupation","monthly_income"]], on="user_id", how="left"
        )
        customers = seg_df.head(200).fillna("").to_dict(orient="records")

    return {"summary": summary, "customers": customers}


# ── /predict-churn ─────────────────────────────────────────────────────────────
@app.post("/predict-churn")
async def predict_churn(file: UploadFile = File(...)):
    try:
        contents = await file.read()
    upload_df = pd.read_csv(io.BytesIO(contents))

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

    # Rule-based churn scoring (high recency + low frequency = churn risk)
    rfm["recency_score"]  = (rfm["recency"]   / rfm["recency"].max())
    rfm["freq_score"]     = 1 - (rfm["frequency"] / rfm["frequency"].max())
    rfm["churn_probability"] = ((rfm["recency_score"] * 0.6) + (rfm["freq_score"] * 0.4)).round(3)
    rfm["revenue_at_risk"]   = (rfm["churn_probability"] * rfm["monetary"]).round(2)
    rfm["churn_predicted"]   = (rfm["churn_probability"] >= 0.5).astype(int)

    result = (
        rfm.merge(users[["user_id","name","occupation","monthly_income"]], on="user_id", how="left")
           .sort_values("revenue_at_risk", ascending=False)
           .head(500)
           .fillna("")
           .to_dict(orient="records")
    )

    summary = {
        "total_users_scored": len(rfm),
        "predicted_churners": int(rfm["churn_predicted"].sum()),
        "total_revenue_at_risk": round(float(rfm["revenue_at_risk"].sum()), 2),
    }

        return {"summary": summary, "at_risk_customers": result}
    except Exception as e:
        return {"error": str(e)}


# ── /restaurants ──────────────────────────────────────────────────────────────
@app.get("/restaurants")
def get_restaurants(
    city:     str  = Query(default=None),
    cuisine:  str  = Query(default=None),
    veg_only: bool = Query(default=False),
):
    df = restaurants.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["cost"]   = pd.to_numeric(df["cost"],   errors="coerce")

    if city:
        df = df[df["city"].str.lower() == city.lower()]
    if cuisine:
        df = df[df["cuisine"].str.contains(cuisine, case=False, na=False)]
    if veg_only:
        veg_fids = food[food["veg_or_non_veg"] == "Veg"][["f_id"]]
        veg_ids = menu.merge(veg_fids, on="f_id")["r_id"].unique()
        df = df[df["id"].isin(veg_ids)]

    cities = sorted(restaurants["city"].dropna().unique().tolist())

    df = df[pd.to_numeric(df["rating"], errors="coerce").notna()].copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["cost"]   = pd.to_numeric(df["cost"],   errors="coerce")

    result = (
        df[["id","name","city","rating","cost","cuisine","address"]]
          .sort_values("rating", ascending=False)
          .head(200)
          .fillna("")
          .to_dict(orient="records")
    )

    return {
        "total": len(df),
        "cities": cities,
        "restaurants": result,
    }


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Zomato Portfolio API is running"}
