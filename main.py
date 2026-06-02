#!/usr/bin/env python3
"""Faireez Demand Dashboard — FastAPI backend with daily CSV cache."""

import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)
STAMP_FILE = DATA_DIR / ".last_refresh"

app = FastAPI(title="Faireez Demand Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_DB = dict(
    host=os.environ.get("DB_HOST", "faireez-db.ceaaeaabvoqy.us-east-1.rds.amazonaws.com"),
    port=int(os.environ.get("DB_PORT", "5432")),
    dbname=os.environ.get("DB_NAME", "faireez"),
    user=os.environ.get("DB_USER", "postgres"),
    password=os.environ["DB_PASSWORD"],  # required — no default
)

# ── In-memory DataFrame cache ──────────────────────────────────────────────────
_frames: dict = {}
_frames_lock = threading.Lock()


def _load(name: str) -> pd.DataFrame:
    with _frames_lock:
        if name in _frames:
            return _frames[name]
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV {name}.csv missing — call /api/refresh")
    df = pd.read_csv(path, low_memory=False)
    with _frames_lock:
        _frames[name] = df
    return df


def _clear():
    with _frames_lock:
        _frames.clear()


# ── DB connection ──────────────────────────────────────────────────────────────
@contextmanager
def db():
    conn = psycopg2.connect(**_DB)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Data download ──────────────────────────────────────────────────────────────
def _download_all():
    print(f"[refresh] Starting at {datetime.utcnow().isoformat()}")
    with db() as cur:

        cur.execute("""
            SELECT l."Id", l."City", l."Project", l."Country",
                   l."ApproximateNumberOfApartments", l."NeighborhoodId",
                   COALESCE(n."Project", '') AS neighbourhood
            FROM "Locations" l
            LEFT JOIN "Neighborhoods" n ON n."Id" = l."NeighborhoodId"
            WHERE l."Status" = 'ACTIVE' AND (l."IsTest" IS NULL OR l."IsTest" = false)
        """)
        pd.DataFrame(cur.fetchall()).to_csv(DATA_DIR / "locations.csv", index=False)
        print("[refresh] locations done")

        cur.execute("""
            SELECT "Id", "LocationId", "Status", "ServiceStartDate"
            FROM "Apartments"
        """)
        pd.DataFrame(cur.fetchall()).to_csv(DATA_DIR / "apartments.csv", index=False)
        print("[refresh] apartments done")

        cur.execute("""
            SELECT "ApartmentId", "OldStatus", "NewStatus",
                   "CreatedAt"::text AS "CreatedAt"
            FROM "ApartmentStatusHistory"
            WHERE "CreatedAt" >= '2024-01-01'
        """)
        pd.DataFrame(cur.fetchall()).to_csv(DATA_DIR / "status_history.csv", index=False)
        print("[refresh] status_history done")

        cur.execute("""
            SELECT "ApartmentId", "EntityType",
                   "FinalPrice", "Status",
                   "Date"::text AS "Date",
                   "CreatedAt"::text AS "CreatedAt"
            FROM "VisitsNew"
            WHERE "Status" IN ('FINISHED','COMPLETED')
              AND "CreatedAt" >= '2023-01-01'
        """)
        pd.DataFrame(cur.fetchall()).to_csv(DATA_DIR / "visits.csv", index=False)
        print("[refresh] visits done")

        cur.execute("""
            SELECT "ApartmentId", "Price",
                   "CreatedAt"::text AS "CreatedAt",
                   ("EndHour" - "StartHour") AS weekly_hours
            FROM "RecurringBookings"
            WHERE "Active" = true
        """)
        pd.DataFrame(cur.fetchall()).to_csv(DATA_DIR / "recurring_bookings.csv", index=False)
        print("[refresh] recurring_bookings done")

        cur.execute("""
            SELECT cr."ApartmentId", cr."Date"::text AS "Date", c."Value"
            FROM "CouponRedeems" cr
            JOIN "Coupons" c ON c."Id" = cr."CouponId"
        """)
        pd.DataFrame(cur.fetchall()).to_csv(DATA_DIR / "coupon_redeems.csv", index=False)
        print("[refresh] coupon_redeems done")

        cur.execute("""
            SELECT ap."ApartmentId",
                   ap."StartDate"::text AS "StartDate",
                   COALESCE(ap."EndDate"::text, CURRENT_DATE::text) AS "EndDate",
                   p."PromotionType", p."PromotionValue"
            FROM "ApartmentsPromotions" ap
            JOIN "Promotions" p ON ap."PromotionId" = p."Id"
            WHERE ap."Active" = true AND p."Type" NOT IN ('free-trial')
        """)
        pd.DataFrame(cur.fetchall()).to_csv(DATA_DIR / "promotions.csv", index=False)
        print("[refresh] promotions done")

        cur.execute('SELECT DISTINCT "ApartmentId" FROM "RegistrationRequests"')
        pd.DataFrame(cur.fetchall()).to_csv(DATA_DIR / "registration_requests.csv", index=False)
        print("[refresh] registration_requests done")

    STAMP_FILE.write_text(datetime.utcnow().isoformat())
    _clear()
    print(f"[refresh] Done at {datetime.utcnow().isoformat()}")


def _is_stale() -> bool:
    if not STAMP_FILE.exists():
        return True
    try:
        last = datetime.fromisoformat(STAMP_FILE.read_text().strip())
        return datetime.utcnow() - last > timedelta(hours=24)
    except Exception:
        return True


def _last_refresh() -> str:
    try:
        return STAMP_FILE.read_text().strip() if STAMP_FILE.exists() else "never"
    except Exception:
        return "unknown"


def _bg_loop():
    while True:
        if _is_stale():
            try:
                _download_all()
            except Exception as e:
                print(f"[refresh] ERROR: {e}")
        time.sleep(3600)


threading.Thread(target=_bg_loop, daemon=True).start()


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


# ── API: refresh & status ──────────────────────────────────────────────────────
@app.get("/api/refresh")
def trigger_refresh():
    try:
        _download_all()
        return {"status": "ok", "refreshed_at": _last_refresh()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


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

    # Revenue from visits (date-filtered)
    v = _visits_df(apt_ids, date_from, date_to)
    monthly_revenue = round(float(v["FinalPrice"].sum()), 2)
    adhoc_revenue   = round(float(v[v["EntityType"] == "ad-hoc"]["FinalPrice"].sum()), 2)

    # MRR from active recurring bookings
    rb = _load("recurring_bookings")
    rb = rb[rb["ApartmentId"].isin(apt_ids)]
    mrr = round(float(rb["Price"].sum()) * 4.33, 2)
    mrr_apt_count = int(rb["ApartmentId"].nunique())
    avg_hours_per_apt = rb.groupby("ApartmentId")["weekly_hours"].sum()
    avg_rec_hours = round(float(avg_hours_per_apt.mean()) if len(avg_hours_per_apt) else 0, 1)

    # Maintenance cost via promotions × visits
    promo = _load("promotions")
    promo = promo[promo["ApartmentId"].isin(sub_ids)].copy()
    vis_all = _load("visits")
    vis_all = vis_all[vis_all["ApartmentId"].isin(sub_ids)].copy()
    vis_all["Date"] = pd.to_datetime(vis_all["Date"], errors="coerce")
    promo["StartDate"] = pd.to_datetime(promo["StartDate"], errors="coerce")
    promo["EndDate"]   = pd.to_datetime(promo["EndDate"],   errors="coerce")
    # Apply visit date filter
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

    # CAC from coupon redeems (current month default)
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
    mrr_per_apt = rb.groupby("ApartmentId")["Price"].sum() * 4.33
    mrr_per_apt = mrr_per_apt[mrr_per_apt > 0]
    buckets = (np.floor(mrr_per_apt / 50) * 50).astype(int)
    mrr_hist = (buckets.value_counts()
                .reset_index()
                .rename(columns={"index": "bucket_start", "count": "count"})
                .sort_values("bucket_start")
                .head(30)
                .to_dict("records"))
    # pandas value_counts column naming varies by version
    if mrr_hist and "bucket_start" not in mrr_hist[0]:
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
    rb["CreatedAt"] = pd.to_datetime(rb["CreatedAt"], format="mixed", utc=True).dt.tz_localize(None)
    rb["month"] = rb["CreatedAt"].dt.to_period("M").astype(str)

    monthly = rb.groupby("month")["Price"].sum().reset_index()
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
    return FileResponse(str(HERE / "index.html"))
