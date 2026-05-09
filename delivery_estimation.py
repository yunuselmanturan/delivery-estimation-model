"""
==========================================================================
 Logistics Delivery Time Estimation using a Feed-Forward Neural Network
 Dataset : Brazilian E-Commerce Public Dataset by Olist (Kaggle)
 Author  : <Yunus TURAN - Paloma Ruiz MOLINA>
 Course  : Neural Network - Mendel University in Brno
==========================================================================

Goal
----
Predict the actual delivery time (in days) between the purchase moment
and the moment the order is physically delivered to the customer.

Modeling strategy
-----------------
The network is a regression MLP whose output is the raw delivery time
in days. The model is trained ONLY on physical / temporal features
(distance, weight, volume, calendar, seller throughput, region, etc.).

For context, after evaluation the script also reports how the carrier's
own estimate would score on the same test set as a simple baseline.
That comparison is informational — the model is trained completely
independently of the carrier's prediction.
"""

# --------------------------------------------------------------------------
# 1. Imports
# --------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# --------------------------------------------------------------------------
# 2. Data loading
# --------------------------------------------------------------------------
DATA_DIR = "data"

def load_data(data_dir=DATA_DIR):
    orders      = pd.read_csv(os.path.join(data_dir, "olist_orders_dataset.csv"))
    order_items = pd.read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv"))
    customers   = pd.read_csv(os.path.join(data_dir, "olist_customers_dataset.csv"))
    sellers     = pd.read_csv(os.path.join(data_dir, "olist_sellers_dataset.csv"))
    products    = pd.read_csv(os.path.join(data_dir, "olist_products_dataset.csv"))
    geolocation = pd.read_csv(os.path.join(data_dir, "olist_geolocation_dataset.csv"))
    return orders, order_items, customers, sellers, products, geolocation


# --------------------------------------------------------------------------
# 3. Feature engineering
# --------------------------------------------------------------------------
def build_dataset(orders, order_items, customers, sellers, products, geolocation):
    """Merge tables, engineer features, return one big dataframe."""

    # -- Keep only delivered orders ----------------------------------------
    orders = orders[orders["order_status"] == "delivered"].copy()

    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for c in date_cols:
        orders[c] = pd.to_datetime(orders[c], errors="coerce")

    orders = orders.dropna(subset=["order_purchase_timestamp",
                                   "order_delivered_customer_date",
                                   "order_estimated_delivery_date"])

    # -- Time-based variables ----------------------------------------------
    orders["delivery_days"] = (
        orders["order_delivered_customer_date"]
        - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / (3600 * 24)

    orders["estimated_days"] = (
        orders["order_estimated_delivery_date"]
        - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / (3600 * 24)

    orders["approval_to_carrier_days"] = (
        orders["order_delivered_carrier_date"]
        - orders["order_approved_at"]
    ).dt.total_seconds() / (3600 * 24)

    # -- Outlier filter (cap at 99th percentile) ---------------------------
    orders = orders[orders["delivery_days"] > 0]
    upper_cap = orders["delivery_days"].quantile(0.99)
    orders = orders[orders["delivery_days"] <= upper_cap]
    print(f"  Outlier cap (99th pct): {upper_cap:.1f} days")

    # -- Order-level aggregates from items ---------------------------------
    items_agg = order_items.groupby("order_id").agg(
        n_items       = ("order_item_id", "count"),
        total_price   = ("price",         "sum"),
        total_freight = ("freight_value", "sum"),
        n_sellers     = ("seller_id",     "nunique"),
        seller_id     = ("seller_id",     "first"),
        product_id    = ("product_id",    "first"),
    ).reset_index()

    # Seller throughput: how many orders each seller fulfilled in total.
    # Busy sellers usually take longer to ship.
    seller_volume = order_items.groupby("seller_id").size().rename("seller_total_orders")
    items_agg = items_agg.merge(seller_volume, on="seller_id", how="left")

    # -- Geolocation: average lat/lng per zip prefix -----------------------
    geo = geolocation.groupby("geolocation_zip_code_prefix").agg(
        lat=("geolocation_lat", "mean"),
        lng=("geolocation_lng", "mean"),
    ).reset_index()
    geo.columns = ["zip_prefix", "lat", "lng"]

    # -- Merge everything --------------------------------------------------
    df = orders.merge(items_agg,  on="order_id",   how="inner")
    df = df.merge(customers,      on="customer_id", how="left")
    df = df.merge(sellers,        on="seller_id",   how="left")
    df = df.merge(products[["product_id", "product_weight_g",
                            "product_length_cm", "product_height_cm",
                            "product_width_cm"]],
                  on="product_id", how="left")

    df = df.merge(geo.rename(columns={"zip_prefix": "customer_zip_code_prefix",
                                      "lat": "cust_lat", "lng": "cust_lng"}),
                  on="customer_zip_code_prefix", how="left")
    df = df.merge(geo.rename(columns={"zip_prefix": "seller_zip_code_prefix",
                                      "lat": "sell_lat", "lng": "sell_lng"}),
                  on="seller_zip_code_prefix", how="left")

    # -- Geographic distance (haversine, km) -------------------------------
    def haversine(lat1, lng1, lat2, lng2):
        R = 6371.0
        lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlng/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    df["distance_km"] = haversine(df["cust_lat"], df["cust_lng"],
                                  df["sell_lat"], df["sell_lng"])

    # -- Engineered features -----------------------------------------------

    # Same state?
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)

    # Same region? (rough Brazilian regions)
    region_map = {
        "SP": "SE", "RJ": "SE", "MG": "SE", "ES": "SE",
        "RS": "S",  "PR": "S",  "SC": "S",
        "BA": "NE", "PE": "NE", "CE": "NE", "MA": "NE", "PB": "NE",
        "RN": "NE", "AL": "NE", "SE": "NE", "PI": "NE",
        "GO": "CW", "MT": "CW", "MS": "CW", "DF": "CW",
        "AM": "N",  "PA": "N",  "RO": "N",  "AC": "N",
        "AP": "N",  "RR": "N",  "TO": "N",
    }
    df["cust_region"] = df["customer_state"].map(region_map)
    df["sell_region"] = df["seller_state"].map(region_map)
    df["same_region"] = (df["cust_region"] == df["sell_region"]).astype(int)

    # Distance brackets (network learns non-linearity through these)
    df["dist_bracket_local"]    = (df["distance_km"] <  50).astype(int)
    df["dist_bracket_regional"] = ((df["distance_km"] >= 50)  & (df["distance_km"] < 500)).astype(int)
    df["dist_bracket_long"]     = ((df["distance_km"] >= 500) & (df["distance_km"] < 1500)).astype(int)
    df["dist_bracket_xlong"]    = (df["distance_km"] >= 1500).astype(int)

    # Calendar
    pt = df["order_purchase_timestamp"]
    df["purchase_month"]   = pt.dt.month
    df["purchase_weekday"] = pt.dt.weekday
    df["purchase_hour"]    = pt.dt.hour
    df["is_weekend"]       = (pt.dt.weekday >= 5).astype(int)

    # Holiday/busy-period flags (Brazilian context)
    df["is_black_friday"] = ((pt.dt.month == 11) & (pt.dt.day >= 20)
                             & (pt.dt.day <= 30)).astype(int)
    df["is_christmas"]    = ((pt.dt.month == 12) & (pt.dt.day >= 15)).astype(int)
    df["is_carnival"]     = ((pt.dt.month == 2)  & (pt.dt.day <= 20)).astype(int)

    # Product physical features
    df["product_volume_cm3"] = (df["product_length_cm"]
                                * df["product_height_cm"]
                                * df["product_width_cm"])
    df["density_g_per_cm3"]  = df["product_weight_g"] / (df["product_volume_cm3"] + 1)

    # Interaction features (heavy + far is much slower than either alone)
    df["dist_x_weight"] = df["distance_km"] * np.log1p(df["product_weight_g"])
    df["dist_x_items"]  = df["distance_km"] * df["n_items"]

    # Log transforms for skewed numeric inputs
    df["log_distance_km"]      = np.log1p(df["distance_km"])
    df["log_total_price"]      = np.log1p(df["total_price"])
    df["log_total_freight"]    = np.log1p(df["total_freight"])
    df["log_product_weight_g"] = np.log1p(df["product_weight_g"])
    df["log_product_volume"]   = np.log1p(df["product_volume_cm3"])
    df["log_seller_orders"]    = np.log1p(df["seller_total_orders"])

    # -- Final feature selection -------------------------------------------
    # NOTE: estimated_days (the carrier's own estimate) is deliberately
    # NOT included as a feature.  We want the model to learn delivery
    # time from physical / temporal evidence, not by piggybacking on
    # another model's prediction.  We do keep it in the dataframe so the
    # evaluation step can report the carrier baseline for comparison.
    feature_cols = [
        "approval_to_carrier_days",

        # geography
        "log_distance_km", "distance_km",
        "same_state", "same_region",
        "dist_bracket_local", "dist_bracket_regional",
        "dist_bracket_long", "dist_bracket_xlong",

        # order
        "n_items", "n_sellers",
        "log_total_price", "log_total_freight",

        # product
        "log_product_weight_g", "log_product_volume",
        "density_g_per_cm3",

        # interactions
        "dist_x_weight", "dist_x_items",

        # seller load
        "log_seller_orders",

        # calendar
        "purchase_month", "purchase_weekday", "purchase_hour",
        "is_weekend", "is_black_friday", "is_christmas", "is_carnival",
    ]
    target_col = "delivery_days"   # network predicts raw delivery time

    # We keep the raw categorical key columns in the dataframe so target
    # encoding can be done AFTER the train/test split (using training
    # rows only, to avoid leakage). They are NOT in feature_cols above
    # and will be replaced with their encoded versions in split_data.
    key_cols = ["customer_state", "seller_state", "seller_id"]
    keep_cols = feature_cols + key_cols + [target_col, "estimated_days"]
    df = df[keep_cols].dropna()

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype("int8")

    return df, target_col


# --------------------------------------------------------------------------
# 4. Train / validation / test split
# --------------------------------------------------------------------------
def split_data(df, target_col):
    """
    70 / 15 / 15 split, with target encoding for high-cardinality
    categoricals computed ONLY on training rows (no leakage).

    Target-encoded features added:
      - lane_mean_delivery       : mean delivery_days per
                                   (seller_state -> customer_state)
      - seller_mean_delivery     : mean delivery_days per seller_id
                                   (with smoothing for rare sellers)
      - cust_state_mean_delivery : mean delivery_days per customer_state
      - sell_state_mean_delivery : mean delivery_days per seller_state
    """
    # Step 1: Split the raw dataframe (still has the categorical keys)
    df_tr, df_tmp = train_test_split(df, test_size=0.30, random_state=SEED)
    df_val, df_te = train_test_split(df_tmp, test_size=0.50, random_state=SEED)

    # Step 2: Compute target encodings using TRAINING rows only ----------
    global_mean = df_tr[target_col].mean()

    # Lane (seller_state -> customer_state)
    lane_means = df_tr.groupby(["seller_state", "customer_state"])[target_col].mean()

    # Seller-level mean with smoothing.
    # smoothed = (n*group_mean + k*global_mean) / (n + k)
    # Sellers with fewer orders get pulled toward the global mean.
    k = 30
    seller_stats = df_tr.groupby("seller_id")[target_col].agg(["mean", "count"])
    seller_means = (seller_stats["count"] * seller_stats["mean"]
                    + k * global_mean) / (seller_stats["count"] + k)

    # State-level means
    cust_state_means = df_tr.groupby("customer_state")[target_col].mean()
    sell_state_means = df_tr.groupby("seller_state")[target_col].mean()

    def add_encodings(d):
        d = d.copy()
        # Lane mean: lookup; fall back to global mean for unseen pairs
        idx = list(zip(d["seller_state"], d["customer_state"]))
        d["lane_mean_delivery"] = [lane_means.get(p, global_mean) for p in idx]

        d["seller_mean_delivery"]     = d["seller_id"].map(seller_means).fillna(global_mean)
        d["cust_state_mean_delivery"] = d["customer_state"].map(cust_state_means).fillna(global_mean)
        d["sell_state_mean_delivery"] = d["seller_state"].map(sell_state_means).fillna(global_mean)
        return d

    df_tr  = add_encodings(df_tr)
    df_val = add_encodings(df_val)
    df_te  = add_encodings(df_te)

    # Step 3: One-hot encode the state columns (low cardinality, ~27 each)
    # Use the same set of dummy columns across all three splits.
    state_cols = ["customer_state", "seller_state"]
    df_tr  = pd.get_dummies(df_tr,  columns=state_cols, drop_first=True)
    df_val = pd.get_dummies(df_val, columns=state_cols, drop_first=True)
    df_te  = pd.get_dummies(df_te,  columns=state_cols, drop_first=True)

    # Align columns across splits (some dummies may be missing in val/te)
    df_val = df_val.reindex(columns=df_tr.columns, fill_value=0)
    df_te  = df_te.reindex(columns=df_tr.columns,  fill_value=0)

    # Step 4: Drop columns we shouldn't feed to the network ----------------
    drop_from_features = [target_col, "estimated_days", "seller_id"]
    feature_cols = [c for c in df_tr.columns if c not in drop_from_features]

    def to_array(d, cols):
        a = d[cols].copy()
        bool_cols = a.select_dtypes(include="bool").columns
        a[bool_cols] = a[bool_cols].astype("int8")
        return a.values.astype("float32")

    X_tr  = to_array(df_tr,  feature_cols)
    X_val = to_array(df_val, feature_cols)
    X_te  = to_array(df_te,  feature_cols)

    y_tr  = df_tr[target_col].values.astype("float32")
    y_val = df_val[target_col].values.astype("float32")
    y_te  = df_te[target_col].values.astype("float32")

    e_tr  = df_tr["estimated_days"].values.astype("float32")
    e_val = df_val["estimated_days"].values.astype("float32")
    e_te  = df_te["estimated_days"].values.astype("float32")

    # Step 5: Standardize the inputs (fit on train only)
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    print(f"Training set   : {X_tr.shape}")
    print(f"Validation set : {X_val.shape}")
    print(f"Test set       : {X_te.shape}")

    return (X_tr, X_val, X_te,
            y_tr, y_val, y_te,
            e_tr, e_val, e_te,
            scaler, feature_cols)


# --------------------------------------------------------------------------
# 5. Model
# --------------------------------------------------------------------------
def build_model(input_dim):
    """
    Wider feed-forward MLP with Huber loss (robust to outliers).
    256 -> 128 -> 64 -> 32 -> 1
    """
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.25),

        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.15),

        Dense(32, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=Huber(delta=2.0),    # less sensitive to outlier residuals
        metrics=["mae"],
    )
    return model


# --------------------------------------------------------------------------
# 6. Training and evaluation
# --------------------------------------------------------------------------
def train_model(model, X_train, y_train, X_val, y_val,
                epochs=120, batch_size=512):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-5, verbose=1),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def accuracy_within_tolerance(y_true, y_pred, tolerances_days=(1, 2, 3, 5, 7),
                              label="Model"):
    abs_err = np.abs(y_true - y_pred)
    print(f"\n=== Accuracy within tolerance ({label}) ===")
    results = {}
    for t in tolerances_days:
        pct = (abs_err <= t).mean() * 100
        results[f"+/- {t} day(s)"] = pct
        print(f"  Within +/- {t} day(s): {pct:6.2f} %  "
              f"({(abs_err <= t).sum()} / {len(y_true)})")
    return results


def evaluate_model(model, X_test, y_test, est_test):
    """
    y_test    = actual delivery_days
    est_test  = carrier's estimated_days (1-D array, used only for baseline)
    """
    actual_days = y_test
    estimated_days = est_test

    # Network outputs delivery_days directly
    pred_days = model.predict(X_test, verbose=0).flatten()

    # ---- Model metrics ---------------------------------------------------
    mae  = mean_absolute_error(actual_days, pred_days)
    rmse = np.sqrt(mean_squared_error(actual_days, pred_days))
    r2   = r2_score(actual_days, pred_days)

    print("\n=========== Test-set performance ===========")
    print(f"MAE  : {mae:.3f} days")
    print(f"RMSE : {rmse:.3f} days")
    print(f"R^2  : {r2:.3f}")
    print("============================================")

    # ---- Tolerance accuracy (model) --------------------------------------
    model_acc = accuracy_within_tolerance(actual_days, pred_days,
                                          label="Neural Network")

    # ---- Baseline = carrier's own estimate -------------------------------
    base_mae  = mean_absolute_error(actual_days, estimated_days)
    base_rmse = np.sqrt(mean_squared_error(actual_days, estimated_days))
    base_r2   = r2_score(actual_days, estimated_days)
    print("\n--- Baseline (carrier's estimated delivery date) ---")
    print(f"  MAE  : {base_mae:.3f} days")
    print(f"  RMSE : {base_rmse:.3f} days")
    print(f"  R^2  : {base_r2:.3f}")
    base_acc = accuracy_within_tolerance(actual_days, estimated_days,
                                         label="Carrier baseline")

    # ---- Improvement summary --------------------------------------------
    print("\n=== Improvement vs. carrier baseline ===")
    print(f"  MAE reduction   : {base_mae - mae:+.3f} days")
    print(f"  +/- 2 day acc.  : {model_acc['+/- 2 day(s)'] - base_acc['+/- 2 day(s)']:+.2f} pp")
    print(f"  +/- 1 day acc.  : {model_acc['+/- 1 day(s)'] - base_acc['+/- 1 day(s)']:+.2f} pp")

    return pred_days, {
        "mae": mae, "rmse": rmse, "r2": r2,
        "baseline_mae": base_mae, "baseline_rmse": base_rmse, "baseline_r2": base_r2,
        "model_accuracy": model_acc,
        "baseline_accuracy": base_acc,
    }


# --------------------------------------------------------------------------
# 7. Plotting
# --------------------------------------------------------------------------
os.makedirs("results", exist_ok=True)

def plot_history(history):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"],     label="train loss",      color="red")
    plt.plot(history.history["val_loss"], label="validation loss", color="blue")
    plt.title("Loss (Huber)")
    plt.xlabel("Epoch"); plt.ylabel("Huber loss"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"],     label="train MAE",      color="teal")
    plt.plot(history.history["val_mae"], label="validation MAE", color="orange")
    plt.title("Mean Absolute Error (residual)")
    plt.xlabel("Epoch"); plt.ylabel("MAE (days)"); plt.legend()

    plt.tight_layout()
    plt.savefig("results/training_history.png", dpi=120)
    plt.show()


def plot_predictions(actual_days, pred_days):
    plt.figure(figsize=(6, 6))
    plt.scatter(actual_days, pred_days, alpha=0.2, s=8)
    lim = max(actual_days.max(), pred_days.max())
    plt.plot([0, lim], [0, lim], color="red", linestyle="--",
             label="perfect prediction")
    plt.xlabel("Actual delivery time (days)")
    plt.ylabel("Predicted delivery time (days)")
    plt.title("Predicted vs. Actual delivery time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/pred_vs_actual.png", dpi=120)
    plt.show()


def plot_error_distribution(actual_days, pred_days, est_days):
    """Side-by-side error distribution: model vs carrier baseline."""
    err_model = pred_days - actual_days
    err_base  = est_days  - actual_days

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(err_model, bins=60, color="steelblue",
             edgecolor="black", alpha=0.8)
    plt.axvline(0, color="red", linestyle="--")
    plt.axvline(-2, color="orange", linestyle=":")
    plt.axvline(+2, color="orange", linestyle=":")
    plt.xlim(-15, 15)
    plt.title("Neural network errors")
    plt.xlabel("predicted - actual (days)")
    plt.ylabel("orders")

    plt.subplot(1, 2, 2)
    plt.hist(err_base, bins=60, color="indianred",
             edgecolor="black", alpha=0.8)
    plt.axvline(0, color="red", linestyle="--")
    plt.axvline(-2, color="orange", linestyle=":")
    plt.axvline(+2, color="orange", linestyle=":")
    plt.xlim(-15, 15)
    plt.title("Carrier baseline errors")
    plt.xlabel("estimated - actual (days)")
    plt.ylabel("orders")

    plt.tight_layout()
    plt.savefig("results/error_distribution.png", dpi=120)
    plt.show()


def plot_model_vs_carrier(actual_days, pred_days, est_days):
    """
    Side-by-side scatter: model predictions vs carrier estimates,
    both plotted against the true delivery time.
    Lets the reader see at a glance which one tracks the diagonal better.
    """
    lim = max(actual_days.max(), pred_days.max(), est_days.max())

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(actual_days, pred_days, alpha=0.2, s=8, color="steelblue")
    plt.plot([0, lim], [0, lim], color="red", linestyle="--",
             label="perfect prediction")
    plt.xlabel("Actual delivery time (days)")
    plt.ylabel("Predicted delivery time (days)")
    plt.title("Neural network")
    plt.xlim(0, lim); plt.ylim(0, lim)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(actual_days, est_days, alpha=0.2, s=8, color="indianred")
    plt.plot([0, lim], [0, lim], color="red", linestyle="--",
             label="perfect prediction")
    plt.xlabel("Actual delivery time (days)")
    plt.ylabel("Carrier estimated time (days)")
    plt.title("Carrier baseline")
    plt.xlim(0, lim); plt.ylim(0, lim)
    plt.legend()

    plt.suptitle("Model vs. carrier: predicted vs. actual delivery time",
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig("results/model_vs_carrier_scatter.png", dpi=120, bbox_inches="tight")
    plt.show()


def plot_accuracy_comparison(model_acc, base_acc):
    """
    Grouped bar chart: % of orders predicted within +/- N days,
    neural network vs carrier baseline. This is the most readable
    summary of "did we do better?".
    """
    tolerances = list(model_acc.keys())
    model_vals = [model_acc[t] for t in tolerances]
    base_vals  = [base_acc[t]  for t in tolerances]

    x = np.arange(len(tolerances))
    width = 0.38

    plt.figure(figsize=(8, 5))
    bars1 = plt.bar(x - width/2, model_vals, width,
                    label="Neural network", color="steelblue",
                    edgecolor="black")
    bars2 = plt.bar(x + width/2, base_vals,  width,
                    label="Carrier baseline", color="indianred",
                    edgecolor="black")

    # Number labels on top of each bar
    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            plt.text(b.get_x() + b.get_width()/2, h + 0.8,
                     f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, tolerances)
    plt.ylabel("Percentage of orders within tolerance")
    plt.title("Accuracy within tolerance: neural network vs. carrier")
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("results/accuracy_comparison.png", dpi=120)
    plt.show()


# --------------------------------------------------------------------------
# 8. Main
# --------------------------------------------------------------------------
def main():
    print(">>> Loading data ...")
    orders, order_items, customers, sellers, products, geolocation = load_data()

    print(">>> Building feature matrix ...")
    df, target_col = build_dataset(orders, order_items, customers,
                                   sellers, products, geolocation)
    print(f"  Final dataset shape: {df.shape}")
    print(f"  Target column     : {target_col}")

    print(">>> Splitting data ...")
    (X_tr, X_val, X_te,
     y_tr, y_val, y_te,
     e_tr, e_val, e_te,
     scaler, feat_cols) = split_data(df, target_col)
    print(f"  Number of features: {len(feat_cols)}")

    print(">>> Building model ...")
    model = build_model(input_dim=X_tr.shape[1])
    model.summary()

    print(">>> Training ...")
    history = train_model(model, X_tr, y_tr, X_val, y_val,
                          epochs=120, batch_size=512)

    print(">>> Evaluating ...")
    pred_days, metrics = evaluate_model(model, X_te, y_te, e_te)

    print(">>> Plotting ...")
    actual_days = y_te
    estim_days  = e_te
    plot_history(history)
    plot_predictions(actual_days, pred_days)
    plot_error_distribution(actual_days, pred_days, estim_days)
    plot_model_vs_carrier(actual_days, pred_days, estim_days)
    plot_accuracy_comparison(metrics["model_accuracy"],
                             metrics["baseline_accuracy"])

    model.save("delivery_time_model.keras")
    print(">>> Model saved to delivery_time_model.keras")


if __name__ == "__main__":
    main()