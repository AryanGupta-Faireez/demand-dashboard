#!/usr/bin/env python3
"""Standalone data refresh script — downloads all CSVs from the database.

Run directly:  python refresh_data.py
Used by:       GitHub Actions workflow (refresh-data.yml)
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras
from contextlib import contextmanager

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

_DB = dict(
    host=os.environ.get("DB_HOST", "faireez-db.ceaaeaabvoqy.us-east-1.rds.amazonaws.com"),
    port=int(os.environ.get("DB_PORT", "5432")),
    dbname=os.environ.get("DB_NAME", "faireez"),
    user=os.environ.get("DB_USER", "postgres"),
    password=os.environ["DB_PASSWORD"],
)


@contextmanager
def db():
    conn = psycopg2.connect(**_DB)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
        conn.commit()
    finally:
        conn.close()


def main():
    print(f"[refresh] Starting at {datetime.utcnow().isoformat()}")
    with db() as cur:

        cur.execute("""
            SELECT l."Id", l."City", l."Project", l."Country",
                   l."ApproximateNumberOfApartments", l."NeighborhoodId",
                   COALESCE(n."Project", '') AS neighbourhood,
                   l."Status" AS building_status
            FROM "Locations" l
            LEFT JOIN "Neighborhoods" n ON n."Id" = l."NeighborhoodId"
            WHERE (l."IsTest" IS NULL OR l."IsTest" = false)
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

    (DATA_DIR / ".last_refresh").write_text(datetime.utcnow().isoformat())
    print(f"[refresh] Done at {datetime.utcnow().isoformat()}")


if __name__ == "__main__":
    main()
