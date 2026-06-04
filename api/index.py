"""Vercel serverless entry point — reads CSVs committed to the repo."""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

ROOT = Path(__file__).parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR_OVERRIDE", str(ROOT / "data")))
STAMP_FILE = DATA_DIR / ".last_refresh"

app = FastAPI(title="Faireez Demand Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── In-memory DataFrame cache ──────────────────────────────────────────────────
_frames: dict = {}


def _load(name: str) -> pd.DataFrame:
    if name in _frames:
        return _frames[name]
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV {name}.csv missing")
    df = pd.read_csv(path, low_memory=False)
    _frames[name] = df
    return df


def _last_refresh() -> str:
    try:
        return STAMP_FILE.read_text().strip() if STAMP_FILE.exists() else "never"
    except Exception:
        return "unknown"


def _is_stale() -> bool:
    if not STAMP_FILE.exists():
        return True
    try:
        last = datetime.fromisoformat(STAMP_FILE.read_text().strip())
        from datetime import timedelta
        return datetime.utcnow() - last > timedelta(hours=24)
    except Exception:
        return True


# ── Currency conversion ────────────────────────────────────────────────────────
ILS_TO_USD = 0.27  # fixed approximate rate: 1 ILS → 0.27 USD

def _usd_multipliers(apt_ids: set) -> pd.Series:
    """Return a Series indexed by ApartmentId with a USD conversion multiplier.
    Apartments in IL / Israel use ILS; everything else is treated as USD."""
    apts = _load("apartments")[["Id", "LocationId"]].copy()
    apts = apts[apts["Id"].isin(apt_ids)]
    loc  = _load("locations")[["Id", "Country"]].copy()
    merged = apts.merge(loc, left_on="LocationId", right_on="Id", suffixes=("_apt", "_loc"))
    merged["multiplier"] = np.where(merged["Country"].isin(["IL", "Israel"]), ILS_TO_USD, 1.0)
    return merged.set_index("Id_apt")["multiplier"]


# ── Filter helpers ─────────────────────────────────────────────────────────────
def _loc_ids(country=None, city=None, project=None, neighbourhood=None) -> set:
    loc = _load("locations")
    if country and country != "all":
        loc = loc[loc["Country"] == country]
    if city and city != "all":
        loc = loc[loc["City"] == city]
    if project and project != "all":
        loc = loc[loc["Project"] == project]
    if neighbourhood and neighbourhood != "all":
        loc = loc[loc["neighbourhood"] == neighbourhood]
    return set(loc["Id"].tolist())


def _apts_df(loc_ids: set, date_from=None, date_to=None) -> pd.DataFrame:
    apts = _load("apartments")
    apts = apts[apts["LocationId"].isin(loc_ids)].copy()
    if date_from or date_to:
        sd = pd.to_datetime(apts["ServiceStartDate"], errors="coerce")
        if date_from:
            apts = apts[sd >= pd.Timestamp(date_from)]
        if date_to:
            apts = apts[sd <= pd.Timestamp(date_to)]
    return apts


def _visits_df(apt_ids: set, date_from=None, date_to=None) -> pd.DataFrame:
    v = _load("visits")
    v = v[v["ApartmentId"].isin(apt_ids)].copy()
    vd = pd.to_datetime(v["Date"], errors="coerce")
    if date_from or date_to:
        if date_from:
            v = v[vd >= pd.Timestamp(date_from)]
        if date_to:
            v = v[vd <= pd.Timestamp(date_to)]
    else:
        now = pd.Timestamp.now()
        v = v[(vd.dt.year == now.year) & (vd.dt.month == now.month)]
    return v


# ── API: status ────────────────────────────────────────────────────────────────
@app.get("/api/refresh")
def trigger_refresh():
    return {"status": "not_available", "message": "Data is refreshed automatically via GitHub Actions every 12h. Trigger the workflow manually from GitHub if needed.", "last_refresh": _last_refresh()}


@app.get("/api/data-status")
def data_status():
    return {"last_refresh": _last_refresh(), "stale": _is_stale()}


# ── API: filters ───────────────────────────────────────────────────────────────
@app.get("/api/filters")
def get_filters():
    loc = _load("locations")
    return {
        "countries":      sorted(loc["Country"].dropna().unique().tolist()),
        "cities":         sorted(loc["City"].dropna().unique().tolist()),
        "projects":       sorted(loc["Project"].dropna().unique().tolist()),
        "neighbourhoods": sorted(loc["neighbourhood"][loc["neighbourhood"] != ""].dropna().unique().tolist()),
    }


# ── API: portfolio KPIs ────────────────────────────────────────────────────────
@app.get("/api/demand/portfolio")
def demand_portfolio(
    date_from: Optional[str] = None, date_to: Optional[str] = None,
    country: Optional[str] = None, city: Optional[str] = None,
    project: Optional[str] = None, neighbourhood: Optional[str] = None,
):
    lids = _loc_ids(country, city, project, neighbourhood)
    loc_f = _load("locations")
    loc_f = loc_f[loc_f["Id"].isin(lids)]
    apts = _apts_df(lids, date_from, date_to)
    apt_ids = set(apts["Id"].tolist())
    rr = _load("registration_requests")
    reg_ids = set(rr["ApartmentId"].tolist())

    return {
        "apts_active_buildings": int(loc_f["ApproximateNumberOfApartments"].sum()),
        "location_count":        int(len(loc_f)),
        "total_leads":           int(len(apts)),
        "registered_leads":      int(len(apt_ids & reg_ids)),
        "subscribers":           int(apts["Status"].isin(["RECURRING_SUBSCRIPTION","ON_DEMAND_SUBSCRIPTION"]).sum()),
        "recurring_subs":        int((apts["Status"] == "RECURRING_SUBSCRIPTION").sum()),
        "ondemand_subs":         int((apts["Status"] == "ON_DEMAND_SUBSCRIPTION").sum()),
        "trial":                 int((apts["Status"] == "TRIAL").sum()),
        "frozen":                int((apts["Status"] == "FROZEN").sum()),
        "churn":                 int((apts["Status"] == "CHURN").sum()),
    }


# ── API: financials ────────────────────────────────────────────────────────────
@app.get("/api/demand/financials")
def demand_financials(
    date_from: Optional[str] = None, date_to: Optional[str] = None,
    country: Optional[str] = None, city: Optional[str] = None,
    project: Optional[str] = None, neighbourhood: Optional[str] = None,
):
    lids = _loc_ids(country, city, project, neighbourhood)
    apts = _apts_df(lids)
    apt_ids = set(apts["Id"].tolist())
    sub_ids = set(apts[apts["Status"].isin(["RECURRING_SUBSCRIPTION","ON_DEMAND_SUBSCRIPTION"])]["Id"].tolist())

    mul = _usd_multipliers(apt_ids)

    v = _visits_df(apt_ids, date_from, date_to)
    v = v.join(mul.rename("_mul"), on="ApartmentId")
    v["_mul"] = v["_mul"].fillna(1.0)
    v["_price_usd"] = v["FinalPrice"] * v["_mul"]
    monthly_revenue = round(float(v["_price_usd"].sum()), 2)
    adhoc_revenue   = round(float(v.loc[v["EntityType"] == "ad-hoc", "_price_usd"].sum()), 2)

    rb = _load("recurring_bookings")
    rb = rb[rb["ApartmentId"].isin(apt_ids)]
    rb = rb.join(mul.rename("_mul"), on="ApartmentId")
    rb["_mul"] = rb["_mul"].fillna(1.0)
    mrr = round(float((rb["Price"] * rb["_mul"]).sum()), 2)
    mrr_apt_count = int(rb["ApartmentId"].nunique())
    avg_hours_per_apt = rb.groupby("ApartmentId")["weekly_hours"].sum()
    avg_rec_hours = round(float(avg_hours_per_apt.mean()) if len(avg_hours_per_apt) else 0, 1)

    promo = _load("promotions")
    promo = promo[promo["ApartmentId"].isin(sub_ids)].copy()
    vis_all = _load("visits")
    vis_all = vis_all[vis_all["ApartmentId"].isin(sub_ids)].copy()
    vis_all["Date"] = pd.to_datetime(vis_all["Date"], errors="coerce")
    promo["StartDate"] = pd.to_datetime(promo["StartDate"], errors="coerce")
    promo["EndDate"]   = pd.to_datetime(promo["EndDate"],   errors="coerce")
    if date_from or date_to:
        if date_from:
            vis_all = vis_all[vis_all["Date"] >= pd.Timestamp(date_from)]
        if date_to:
            vis_all = vis_all[vis_all["Date"] <= pd.Timestamp(date_to)]
    else:
        now = pd.Timestamp.now()
        vis_all = vis_all[(vis_all["Date"].dt.year == now.year) & (vis_all["Date"].dt.month == now.month)]
    maintenance_cost = 0.0
    for _, p in promo.iterrows():
        apt_vis = vis_all[
            (vis_all["ApartmentId"] == p["ApartmentId"]) &
            (vis_all["Date"] >= p["StartDate"]) &
            (vis_all["Date"] <= p["EndDate"])
        ]
        if p["PromotionType"] == "percentage":
            maintenance_cost += float((apt_vis["FinalPrice"] * p["PromotionValue"] / 100.0).sum())
        elif p["PromotionType"] == "fixed":
            maintenance_cost += float(p["PromotionValue"]) * len(apt_vis)
    maintenance_cost = round(maintenance_cost, 2)

    cr = _load("coupon_redeems")
    cr = cr[cr["ApartmentId"].isin(apt_ids)].copy()
    cr["Date"] = pd.to_datetime(cr["Date"], errors="coerce")
    if date_from or date_to:
        if date_from:
            cr = cr[cr["Date"] >= pd.Timestamp(date_from)]
        if date_to:
            cr = cr[cr["Date"] <= pd.Timestamp(date_to)]
    else:
        now = pd.Timestamp.now()
        cr = cr[(cr["Date"].dt.year == now.year) & (cr["Date"].dt.month == now.month)]
    cac_cost = round(float(cr["Value"].sum()), 2)

    cac_pct       = round(cac_cost * 12 / mrr * 100, 2) if mrr > 0 else 0
    retention_pct = round(maintenance_cost * 12 / mrr * 100, 2) if mrr > 0 else 0

    return {
        "monthly_revenue": monthly_revenue,
        "adhoc_revenue":   adhoc_revenue,
        "mrr":             mrr,
        "mrr_apt_count":   mrr_apt_count,
        "avg_rec_hours":   avg_rec_hours,
        "maintenance_cost": maintenance_cost,
        "cac_cost":        cac_cost,
        "cac_pct_annualised": cac_pct,
        "retention_cost_pct_annualised": retention_pct,
    }


# ── API: histograms ────────────────────────────────────────────────────────────
@app.get("/api/demand/histograms")
def demand_histograms(
    date_from: Optional[str] = None, date_to: Optional[str] = None,
    country: Optional[str] = None, city: Optional[str] = None,
    project: Optional[str] = None, neighbourhood: Optional[str] = None,
):
    lids = _loc_ids(country, city, project, neighbourhood)
    apts = _apts_df(lids)
    apt_ids = set(apts["Id"].tolist())

    rb = _load("recurring_bookings")
    rb = rb[rb["ApartmentId"].isin(apt_ids)]
    mul = _usd_multipliers(apt_ids)
    rb = rb.join(mul.rename("_mul"), on="ApartmentId")
    rb["_mul"] = rb["_mul"].fillna(1.0)
    rb["_price_usd"] = rb["Price"] * rb["_mul"]
    mrr_per_apt = rb.groupby("ApartmentId")["_price_usd"].sum()
    mrr_per_apt = mrr_per_apt[mrr_per_apt > 0]
    buckets = (np.floor(mrr_per_apt / 50) * 50).astype(int)
    mrr_hist = [{"bucket_start": int(k), "count": int(c)} for k, c in
                sorted(buckets.value_counts().items())][:30]

    v = _visits_df(apt_ids, date_from, date_to)
    adhoc = v[v["EntityType"] == "ad-hoc"].groupby("ApartmentId")["FinalPrice"].sum()
    adhoc = adhoc[adhoc > 0]
    buckets2 = (np.floor(adhoc / 10) * 10).astype(int)
    adhoc_hist = [{"bucket_start": int(k), "count": int(c)} for k, c in
                  sorted(buckets2.value_counts().items())][:30]

    return {"mrr": mrr_hist, "adhoc": adhoc_hist}


# ── API: cumulative subscribers ────────────────────────────────────────────────
@app.get("/api/demand/cumulative-subscribers")
def cumulative_subscribers(
    country: Optional[str] = None, city: Optional[str] = None,
    project: Optional[str] = None, neighbourhood: Optional[str] = None,
):
    lids = _loc_ids(country, city, project, neighbourhood)
    apts = _apts_df(lids)
    apt_ids = set(apts["Id"].tolist())

    ash = _load("status_history").copy()
    ash = ash[ash["ApartmentId"].isin(apt_ids)]
    ash["CreatedAt"] = pd.to_datetime(ash["CreatedAt"], format="mixed", utc=True).dt.tz_localize(None)

    SUB = {"RECURRING_SUBSCRIPTION", "ON_DEMAND_SUBSCRIPTION"}
    months = pd.date_range("2024-06-01", pd.Timestamp.now(), freq="MS")
    rows, prev_live = [], None

    for month_start in months:
        month_end = month_start + pd.offsets.MonthEnd(0)
        snap = ash[ash["CreatedAt"] <= month_end]
        if snap.empty:
            rows.append({"month": month_start.strftime("%Y-%m"), "live_subs": 0,
                         "recurring": 0, "on_demand": 0, "mom_change": None})
            prev_live = 0
            continue
        latest = snap.sort_values("CreatedAt").groupby("ApartmentId").last()
        live_subs = int(latest["NewStatus"].isin(SUB).sum())
        recurring  = int((latest["NewStatus"] == "RECURRING_SUBSCRIPTION").sum())
        on_demand  = int((latest["NewStatus"] == "ON_DEMAND_SUBSCRIPTION").sum())
        mom = (live_subs - prev_live) if prev_live is not None else None
        rows.append({"month": month_start.strftime("%Y-%m"), "live_subs": live_subs,
                     "recurring": recurring, "on_demand": on_demand, "mom_change": mom})
        prev_live = live_subs

    return rows


# ── API: churn monthly ─────────────────────────────────────────────────────────
@app.get("/api/demand/churn-monthly")
def churn_monthly(
    country: Optional[str] = None, city: Optional[str] = None,
    project: Optional[str] = None, neighbourhood: Optional[str] = None,
):
    lids = _loc_ids(country, city, project, neighbourhood)
    apts = _apts_df(lids)
    apt_ids = set(apts["Id"].tolist())

    ash = _load("status_history").copy()
    ash = ash[ash["ApartmentId"].isin(apt_ids)]
    ash["CreatedAt"] = pd.to_datetime(ash["CreatedAt"], format="mixed", utc=True).dt.tz_localize(None)

    SUB = {"RECURRING_SUBSCRIPTION", "ON_DEMAND_SUBSCRIPTION"}
    months = pd.date_range("2024-06-01", pd.Timestamp.now(), freq="MS")
    rows, prev_live = [], None

    for month_start in months:
        month_end = month_start + pd.offsets.MonthEnd(0)
        snap = ash[ash["CreatedAt"] <= month_end]
        live_subs = 0
        if not snap.empty:
            latest = snap.sort_values("CreatedAt").groupby("ApartmentId").last()
            live_subs = int(latest["NewStatus"].isin(SUB).sum())
        churned = int(len(ash[
            (ash["CreatedAt"] >= month_start) &
            (ash["CreatedAt"] <= month_end) &
            (ash["OldStatus"].isin(SUB)) &
            (~ash["NewStatus"].isin(SUB))
        ]))
        churn_pct = round(churned / prev_live * 100, 1) if prev_live else None
        rows.append({"month": month_start.strftime("%Y-%m"), "live_subs": live_subs,
                     "churned": churned, "churn_pct": churn_pct})
        prev_live = live_subs

    return rows


# ── API: monthly active users ──────────────────────────────────────────────────
@app.get("/api/demand/active-users-monthly")
def active_users_monthly(
    country: Optional[str] = None, city: Optional[str] = None,
    project: Optional[str] = None, neighbourhood: Optional[str] = None,
):
    lids = _loc_ids(country, city, project, neighbourhood)
    apts = _apts_df(lids)
    apt_ids = set(apts["Id"].tolist())

    v = _load("visits").copy()
    v = v[v["ApartmentId"].isin(apt_ids)]
    v["CreatedAt"] = pd.to_datetime(v["CreatedAt"], format="mixed", utc=True).dt.tz_localize(None)
    v["month"] = v["CreatedAt"].dt.to_period("M").astype(str)

    monthly = v.groupby("month")["ApartmentId"].nunique().reset_index()
    monthly.columns = ["month", "active_users"]
    monthly = monthly.sort_values("month").reset_index(drop=True)

    rows = monthly.to_dict("records")
    for i, row in enumerate(rows):
        prev = rows[i - 1]["active_users"] if i > 0 else None
        row["active_users"] = int(row["active_users"])
        row["mom_change"]   = int(row["active_users"] - prev) if prev is not None else None
        row["mom_pct"]      = round((row["active_users"] - prev) / prev * 100, 1) if prev else None
    return rows


# ── API: cumulative MRR ────────────────────────────────────────────────────────
@app.get("/api/demand/cumulative-mrr")
def cumulative_mrr(
    country: Optional[str] = None, city: Optional[str] = None,
    project: Optional[str] = None, neighbourhood: Optional[str] = None,
):
    lids = _loc_ids(country, city, project, neighbourhood)
    apts = _apts_df(lids)
    apt_ids = set(apts["Id"].tolist())

    rb = _load("recurring_bookings").copy()
    rb = rb[rb["ApartmentId"].isin(apt_ids)]
    mul = _usd_multipliers(apt_ids)
    rb = rb.join(mul.rename("_mul"), on="ApartmentId")
    rb["_mul"] = rb["_mul"].fillna(1.0)
    rb["_price_usd"] = rb["Price"] * rb["_mul"]
    rb["CreatedAt"] = pd.to_datetime(rb["CreatedAt"], format="mixed", utc=True).dt.tz_localize(None)
    rb["month"] = rb["CreatedAt"].dt.to_period("M").astype(str)

    monthly = rb.groupby("month")["_price_usd"].sum().reset_index()
    monthly.columns = ["month", "new_mrr"]
    monthly = monthly.sort_values("month").reset_index(drop=True)
    monthly["cumulative_mrr"] = monthly["new_mrr"].cumsum()
    monthly["new_mrr"]        = monthly["new_mrr"].round(2)
    monthly["cumulative_mrr"] = monthly["cumulative_mrr"].round(2)
    return monthly.to_dict("records")


# ── API: cumulative visits ─────────────────────────────────────────────────────
@app.get("/api/demand/cumulative-visits")
def cumulative_visits(
    country: Optional[str] = None, city: Optional[str] = None,
    project: Optional[str] = None, neighbourhood: Optional[str] = None,
):
    lids = _loc_ids(country, city, project, neighbourhood)
    apts = _apts_df(lids)
    apt_ids = set(apts["Id"].tolist())

    v = _load("visits").copy()
    v = v[v["ApartmentId"].isin(apt_ids)]
    v["CreatedAt"] = pd.to_datetime(v["CreatedAt"], format="mixed", utc=True).dt.tz_localize(None)
    v["month"] = v["CreatedAt"].dt.to_period("M").astype(str)

    monthly = v.groupby("month").size().reset_index(name="visits")
    monthly = monthly.sort_values("month").reset_index(drop=True)
    monthly["cumulative_visits"] = monthly["visits"].cumsum()

    rows = monthly.to_dict("records")
    for i, row in enumerate(rows):
        prev = rows[i - 1]["visits"] if i > 0 else None
        row["visits"]            = int(row["visits"])
        row["cumulative_visits"] = int(row["cumulative_visits"])
        row["mom_change"]        = int(row["visits"] - prev) if prev is not None else None
        row["mom_pct"]           = round((row["visits"] - prev) / prev * 100, 1) if prev else None
    return rows




# ── Serve frontend ─────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(str(Path(__file__).parent / "index.html"))
