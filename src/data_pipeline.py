"""
Data extraction, cleaning, aggregation, and synthetic data generation.
"""
import sys
from pathlib import Path

# Ensure project root is on path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.database import get_engine, get_session, Transaction
from src.elasticity_model import ElasticityModel
import config


class DataPipeline:
    def __init__(self, db_path=None):
        db_path = db_path or config.DB_PATH
        self.engine = get_engine(db_path)
        self.session = get_session(self.engine)

    # ------------------------------------------------------------------ #
    #  CATEGORY MAPPING                                                    #
    # ------------------------------------------------------------------ #

    _CATEGORY_RULES = [
        (["wall art", "framed", "poster", "canvas"], "Wall Art"),
        (["gift", "card"], "Giftware & Cards"),
        (["heart", "rose", "valentine"], "Romantic Gifts"),
        (["star wars", "doctor who", "disney", "ninja turtle", "muppet"], "Character Gifts"),
        (["bird", "animal", "butterfly", "bee"], "Animal-themed"),
        (["party", "candle", "cake", "balloon"], "Party Supplies"),
        (["retro", "vintage"], "Vintage Collection"),
        (["music", "instrument", "guitar", "piano"], "Musical"),
        (["baby", "child", "kid", "children"], "Baby & Kids"),
        (["plant", "flower", "garden", "pot"], "Garden & Plants"),
        (["jewellery", "jewelry", "necklace", "bracelet", "earring", "ring"], "Jewellery"),
        (["home", "kitchen", "table", "cup", "mug", "plate", "bowl", "utensil"], "Home & Kitchen"),
        (["bag", "wallet", "handbag", "purse", "tote"], "Bags & Accessories"),
    ]

    def _assign_category(self, description):
        if pd.isna(description):
            return "Other"
        desc = str(description).lower()
        for keywords, cat in self._CATEGORY_RULES:
            if any(kw in desc for kw in keywords):
                return cat
        return "Other"

    # ------------------------------------------------------------------ #
    #  PROMOTION / SEASONALITY CONTROLS                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def add_controls(df):
        """Add promotion_flag, is_winter, is_holiday_season to a DataFrame."""
        df = df.sort_values("invoice_date").copy()
        df["prev_price"] = df.groupby("product_id")["unit_price"].shift(1)
        df["promotion_flag"] = (df["unit_price"] < df["prev_price"] * 0.80).astype(int)
        df = df.dropna(subset=["prev_price"])

        df["month"] = pd.to_datetime(df["invoice_date"]).dt.month
        df["is_winter"] = df["month"].isin([11, 12, 1, 2]).astype(int)
        df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)
        df.drop(columns=["month", "prev_price"], inplace=True, errors="ignore")
        return df

    # ------------------------------------------------------------------ #
    #  LOADING & CLEANING                                                  #
    # ------------------------------------------------------------------ #

    def load_csv_to_db(self, csv_path):
        """Load raw CSV into the SQLite database (idempotent — truncates first)."""
        print(f"Loading data from {csv_path} ...")
        df = pd.read_csv(csv_path)

        df.columns = df.columns.str.lower().str.replace(" ", "_")
        df = df.rename(columns={
            "invoice_no": "invoice_no",
            "invoice_date": "invoice_date",
            "quantity": "quantity",
            "unitprice": "unit_price",
            "stockcode": "product_id",
            "description": "product_name",
            "customerid": "customer_id",
        })

        df = self._clean_data(df)
        df["revenue"] = df["quantity"] * df["unit_price"]
        df["category"] = df["product_name"].apply(self._assign_category)
        df["invoice_date"] = pd.to_datetime(df["invoice_date"])

        # Compute derived control variables BEFORE inserting into the DB
        df = DataPipeline.add_controls(df)

        # Truncate before insert so re-runs don't duplicate
        self.session.execute(Transaction.__table__.delete())
        self.session.commit()

        # Fast bulk insert via pandas to_sql (10-100x faster than row-by-row ORM)
        insert_cols = [
            "invoice_no", "invoice_date", "quantity", "unit_price",
            "product_id", "product_name", "category", "customer_id",
            "country", "revenue", "promotion_flag", "is_winter", "is_holiday_season",
        ]
        df_insert = df[insert_cols].copy()

        df_insert.to_sql(
            "transactions",
            con=self.engine,
            if_exists="append",
            index=False,
            chunksize=5000,
        )
        print(f"Loaded {len(df_insert)} records into database")
        return df

    def _clean_data(self, df):
        print(f"Initial records: {len(df)}")

        # Drop cancellation invoices (start with 'C')
        df = df[~df["invoice_no"].astype(str).str.startswith("C", na=False)]

        df = df.dropna(subset=["quantity", "unit_price", "customer_id"])
        df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]

        # Hierarchical price-outlier filter: per-product if enough data, else category-level
        q = config.PRICE_OUTLIER_QUANTILE
        min_for_product = config.MIN_PER_PRODUCT_FOR_OUTLIER

        # Per-product count
        prod_counts = df.groupby("product_id").size()
        # Per-product quantile bounds (only where enough observations)
        eligible_pids = prod_counts[prod_counts >= min_for_product].index
        p_q = df[df["product_id"].isin(eligible_pids)].groupby("product_id")["unit_price"].quantile(q)
        # Per-category quantile bounds (fallback)
        c_q = df.groupby("category")["unit_price"].quantile(q)

        # Build bound as a lookup: prefer product, fall back to category
        bound_s = pd.Series(np.inf, index=df.index)
        # Set product bounds where available
        pid_to_bound = dict(zip(p_q.index, p_q.values))
        cat_to_bound = dict(zip(c_q.index, c_q.values))
        bound_s = df["product_id"].map(pid_to_bound).fillna(
            df["category"].map(cat_to_bound)
        ).fillna(np.inf)

        df = df[df["unit_price"] < bound_s]

        print(f"After cleaning: {len(df)}")
        return df

    # ------------------------------------------------------------------ #
    #  AGGREGATION & ELASTICITY INPUTS                                     #
    # ------------------------------------------------------------------ #

    def aggregate_by_product_period(self, period_days=7):
        """Aggregate transactions by product and time bucket for elasticity input."""
        # Use STRFTIME to properly parse date strings with time components
        query = """
        SELECT
            product_id,
            category,
            STRFTIME('%Y-%m-%d', invoice_date) AS period_date,
            AVG(unit_price) AS avg_price,
            SUM(quantity) AS total_quantity,
            SUM(revenue) AS total_revenue,
            COUNT(*) AS transaction_count,
            AVG(promotion_flag) AS avg_promotion,
            AVG(is_winter) AS avg_winter,
            AVG(is_holiday_season) AS avg_holiday
        FROM transactions
        GROUP BY product_id, STRFTIME('%Y-%m-%d', invoice_date)
        ORDER BY product_id, period_date
        """
        df = pd.read_sql(query, self.engine)
        df["period_date"] = pd.to_datetime(df["period_date"])

        df = df.sort_values(["product_id", "period_date"])
        df["price_lag"] = df.groupby("product_id")["avg_price"].shift(1)
        df["quantity_lag"] = df.groupby("product_id")["total_quantity"].shift(1)

        df["price_change_pct"] = (
            ((df["avg_price"] - df["price_lag"]) / df["price_lag"] * 100).round(2)
        )
        df["quantity_change_pct"] = (
            ((df["total_quantity"] - df["quantity_lag"]) / df["quantity_lag"] * 100).round(2)
        )

        df = df.dropna(subset=["price_change_pct", "quantity_change_pct"])
        df = df[df["quantity_lag"] > 0]  # drop periods with zero baseline quantity

        # Price-change filter (kept as-is since price jumps are more informative)
        df = df[df["price_change_pct"].abs() < config.PRICE_CHANGE_THRESHOLD]

        # Volume-aware quantity filter: only cap extreme changes for high-volume periods
        high_vol = df["quantity_lag"] >= config.HIGH_VOLUME_THRESHOLD
        extreme_change = df["quantity_change_pct"].abs() > config.MAX_HIGH_VOLUME_CHANGE_PCT
        df = df[~(high_vol & extreme_change)]

        return df

    def get_products_for_elasticity(self, min_transactions=None, min_price_variance=None):
        min_transactions = min_transactions or config.MIN_TRANSACTIONS_PER_PRODUCT
        min_price_variance = min_price_variance or config.MIN_PRICE_VARIANCE

        query = """
        SELECT
            product_id,
            category,
            COUNT(*) AS transaction_count,
            MIN(unit_price) AS min_price,
            MAX(unit_price) AS max_price,
            AVG(unit_price) AS avg_price,
            (MAX(unit_price) - MIN(unit_price)) / NULLIF(AVG(unit_price), 0) AS price_variance,
            SUM(quantity) AS total_units_sold,
            SUM(revenue) AS total_revenue
        FROM transactions
        GROUP BY product_id
        HAVING COUNT(*) >= ?
        ORDER BY total_revenue DESC
        """
        df = pd.read_sql(query, self.engine, params=(min_transactions,))
        df = df[df["price_variance"] >= min_price_variance]
        return df

    def get_segment_elasticities(self):
        """Compute mean-level elasticity per (category, country).

        Estimates elasticity for each product independently, then reports a
        sales-weighted average within the segment.  This avoids the aggregation
        bias that arises from pooling heterogeneous product-level data into a
        single regression.
        """
        segment_results = []
        group_cols = ["category", "country"]

        for col in group_cols:
            groups = self.session.query(Transaction.__table__.c[col]).distinct().all()
            for (group_value,) in groups:
                # Gather transactions for this segment
                df = self.session.query(
                    Transaction.unit_price, Transaction.quantity,
                    Transaction.invoice_date, Transaction.product_id,
                ).filter(Transaction.__table__.c[col] == group_value).all()

                if len(df) < config.MIN_TRANSACTIONS_FOR_SEGMENT:
                    continue

                record_df = pd.DataFrame(
                    [{"unit_price": r.unit_price, "quantity": r.quantity,
                      "invoice_date": r.invoice_date, "product_id": r.product_id,
                      col: group_value} for r in df]
                )
                record_df = DataPipeline.add_controls(record_df)

                # Estimate elasticity per product within the segment
                product_results = []
                for pid, gdf in record_df.groupby("product_id"):
                    if len(gdf) < config.MIN_TRANSACTIONS_PER_PRODUCT:
                        continue
                    model = ElasticityModel()
                    controls = config.CONTROL_VARIABLES
                    elastic = model.calculate_elasticity(gdf, control_variables=controls)
                    if elastic is not None:
                        total_units = gdf["quantity"].sum()
                        product_results.append({
                            "product_id": pid,
                            "elasticity": elastic,
                            "n_transactions": len(gdf),
                            "total_units": total_units,
                            col: group_value,
                        })

                if not product_results:
                    continue

                # Sales-weighted average elasticity across products in the segment
                prod_df = pd.DataFrame(product_results)
                total_seg_units = prod_df["total_units"].sum()
                weighted_elasticity = (prod_df["elasticity"] * prod_df["total_units"]).sum() / total_seg_units

                segment_results.append({
                    col: group_value,
                    "elasticity": round(weighted_elasticity, 4),
                    "n_products": len(product_results),
                    "n_transactions": int(prod_df["n_transactions"].sum()),
                    "avg_price": float(record_df["unit_price"].mean()),
                    "total_units": int(total_seg_units),
                })

        return pd.DataFrame(segment_results)

    # ------------------------------------------------------------------ #
    #  SYNTHETIC DATA GENERATOR                                            #
    # ------------------------------------------------------------------ #

    def generate_synthetic_data(self, n_products=30, n_days=365, seed=42):
        """Generate synthetic transaction data with known elasticities per product."""
        np.random.seed(seed)
        categories = [
            "Wall Art", "Giftware & Cards", "Character Gifts", "Home & Kitchen",
            "Jewellery", "Party Supplies", "Bags & Accessories", "Other",
        ]

        # Assign each product an elasticity and base price
        products = []
        for i in range(n_products):
            elasticities = [-2.0, -1.5, -1.0, -0.7, -0.4, -0.2]
            elasticity = np.random.choice(elasticities)
            base_price = round(np.random.uniform(3, 30), 2)
            base_qty = int(np.random.uniform(5, 80))
            category = np.random.choice(categories)
            country = np.random.choice(["UK", "Germany", "France", "Netherlands", "Spain"])
            products.append({
                "product_id": f"P{i+1:03d}",
                "category": category,
                "country": country,
                "base_price": base_price,
                "base_quantity": base_qty,
                "elasticity": elasticity,
            })

        records = []
        start_date = datetime(2023, 1, 1)

        for p in products:
            for day in range(n_days):
                date = start_date + timedelta(days=day)
                # Simulate 1-5 transactions per product per day
                n_txns = np.random.poisson(2) + 1
                for _ in range(n_txns):
                    # Vary price slightly around base
                    price_shock = np.random.normal(0, p["base_price"] * 0.05)
                    price = max(p["base_price"] + price_shock, 0.5)

                    # Quantity depends on price via elasticity + noise
                    log_q = (
                        np.log(max(p["base_quantity"], 1))
                        + p["elasticity"] * np.log(price / p["base_price"])
                        + np.random.normal(0, 0.3)
                    )
                    qty = max(int(np.exp(log_q)), 1)

                    revenue = round(qty * price, 2)
                    cust_id = f"C{np.random.randint(1, 500):04d}"

                    records.append({
                        "invoice_no": f"INV-{len(records)+1:06d}",
                        "invoice_date": date + timedelta(hours=np.random.randint(8, 21)),
                        "quantity": qty,
                        "unit_price": round(price, 2),
                        "product_id": p["product_id"],
                        "product_name": f"{p['category']} Item {p['product_id']}",
                        "category": p["category"],
                        "customer_id": cust_id,
                        "country": p["country"],
                        "revenue": revenue,
                        "promotion_flag": False,
                        "is_winter": int(date.month in [11, 12, 1, 2]),
                        "is_holiday_season": int(date.month in [11, 12]),
                    })

        df = pd.DataFrame(records)
        csv_path = config.RAW_DATA_DIR / "synthetic_retail.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Synthetic dataset: {len(df)} records written to {csv_path}")
        return df

    # ------------------------------------------------------------------ #
    #  UTILITIES                                                           #
    # ------------------------------------------------------------------ #

    def close(self):
        self.session.close()


# ---------------------------------------------------------------------- #
#  RUNNER (python src/data_pipeline.py)                                   #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    pipeline = DataPipeline()
    synthetic_csv = config.RAW_DATA_DIR / "synthetic_retail.csv"

    if not synthetic_csv.exists():
        pipeline.generate_synthetic_data()

    pipeline.load_csv_to_db(str(synthetic_csv))

    products = pipeline.get_products_for_elasticity()
    print(f"\nFound {len(products)} products suitable for elasticity analysis\n")
    print(products.head(10).to_string(index=False))

    seg = pipeline.get_segment_elasticities()
    if not seg.empty:
        print(f"\nSegment-level elasticity:\n{seg.head(10).to_string(index=False)}")

    pipeline.close()
