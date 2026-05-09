# Logistics Delivery Time Estimation - Brazilian E-Commerce (Olist)

A feed-forward neural network (Multi-Layer Perceptron) that predicts how
many days an order will take to be delivered to the customer, trained on
the public Olist dataset from Kaggle.

---

## 1. Folder layout

```
delivery_estimation/
│
├── delivery_estimation.py                  # main script (everything is here)
├── results                                 # folder where the performance result of the model will go
├──delivery_estimation_model_report.docx    # report for the project
├── README.md                               # this file
└── data/                                   # put the 9 csv files from Kaggle here
    ├── olist_orders_dataset.csv
    ├── olist_order_items_dataset.csv
    ├── olist_customers_dataset.csv
    ├── olist_sellers_dataset.csv
    ├── olist_products_dataset.csv
    ├── olist_geolocation_dataset.csv
    ├── olist_order_payments_dataset.csv          (not used, optional)
    ├── olist_order_reviews_dataset.csv           (not used, optional)
    └── product_category_name_translation.csv    (not used, optional)
```

The download link is:
<https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>

---

## 2. Required Python libraries

```
pip install numpy pandas matplotlib scikit-learn tensorflow
```

Tested with Python 3.10 / 3.11 and TensorFlow 2.15+.

---

## 3. How to run

From inside the project folder:

```
python delivery_estimation.py
```

The script will:

1. Load all csv files from `./data/`.
2. Clean them and build one big table (one row per order).
3. Engineer features (distance, distance brackets, holiday flags,
   interactions, log-transformed numerics, seller throughput, calendar
   features, region encoding, …).
4. Split the data **70 % / 15 % / 15 %** (train / validation / test).
5. Compute **target-encoded features** (mean delivery time per
   seller→customer lane, per seller, per state) using **training rows
   only**, to give the network high-signal categorical features without
   leakage.
6. Build and train an MLP (256 → 128 → 64 → 32 → 1) for up to 120 epochs
   with early-stopping and learning-rate decay.
7. Report MAE, RMSE, R² and "accuracy within ±N days" for both the
   neural network **and the carrier's own estimate as a baseline**.
8. Save five figures (`training_history.png`, `pred_vs_actual.png`,
   `error_distribution.png`, `model_vs_carrier_scatter.png`,
   `accuracy_comparison.png`) and the model itself
   (`delivery_time_model.keras`).

Expected runtime: 5 – 15 minutes on a normal CPU, less on GPU.

---

## 4. Modeling strategy: honest, end-to-end prediction

The network predicts the raw delivery time (in days) directly from
physical and temporal evidence:

```
network input  : distance, weight, volume, calendar features,
                 seller throughput, region, holiday flags, ...
network output : predicted delivery time in days
```

The carrier's own estimate (`estimated_days`) is **deliberately NOT
fed to the network as a feature**. Doing so would let the model
"cheat" by simply copying the carrier instead of learning the
underlying logistics from scratch. Keeping the carrier's estimate out
of the inputs makes the network genuinely learn the relationships
between distance, package size, season, etc. and delivery time, which
is the real point of the project.

The carrier's estimate is, however, used as an external **baseline at
evaluation time** so the report can compare the network's accuracy
against the existing logistics system. The script reports both side by
side and produces a comparison plot
(`accuracy_comparison.png`).

---

## 5. How the data is split

| Subset         | Proportion | Purpose                                        |
| -------------- | ---------- | ---------------------------------------------- |
| Training       |  70 %      | Updating the network weights                   |
| Validation     |  15 %      | Monitoring overfitting + early-stopping        |
| Test           |  15 %      | Final, untouched evaluation of the model       |

The split uses `train_test_split` from scikit-learn with a fixed random
seed (`SEED = 42`), so the results are reproducible.

---

## 6. How to "teach" / re-train the model

Re-training is just calling the script again. To do hyperparameter
experiments (the kind the example project documents in its report),
edit these in `delivery_estimation.py`:

| Hyper-parameter        | Where to change                                    |
| ---------------------- | -------------------------------------------------- |
| Number of layers       | `build_model()` — add / remove `Dense(...)` lines  |
| Number of neurons      | `Dense(256, ...)`, `Dense(128, ...)` etc.          |
| Activation function    | `activation="relu"` → `"tanh"`, `"selu"`, …        |
| Dropout rate           | `Dropout(0.3)` → `0.5`, `0.1`, …                   |
| Learning rate          | `Adam(learning_rate=1e-3)`                         |
| Optimizer              | `Adam(...)` → `SGD()`, `RMSprop()` …               |
| Loss function          | `Huber(delta=2.0)` → `"mse"`, `"mae"`              |
| Batch size             | `train_model(..., batch_size=512)`                 |
| Epochs                 | `train_model(..., epochs=120)`                     |
| Outlier cap            | `quantile(0.99)` in `build_dataset()`              |

After training, the saved model can be reloaded:

```python
import tensorflow as tf
model = tf.keras.models.load_model("delivery_time_model.keras")
prediction_days = model.predict(X_new).flatten()  # raw delivery time
```

---

## 7. Expected outputs

A successful run produces numbers like this (small variation between
runs is normal because dropout and weight initialization use random
draws):

```
Training set   : (66 504, 77)
Validation set : (14 251, 77)
Test set       : (14 251, 77)

=========== Test-set performance ===========
MAE  : 3.57 days
RMSE : 5.55 days
R^2  : 0.49
============================================

=== Accuracy within tolerance (Neural Network) ===
  Within +/- 1 day(s):  25.7 %
  Within +/- 2 day(s):  47.1 %    <-- headline number
  Within +/- 3 day(s):  61.5 %
  Within +/- 5 day(s):  78.5 %
  Within +/- 7 day(s):  86.7 %

--- Baseline (carrier's estimated delivery date) ---
  MAE  : 12.5 days

=== Accuracy within tolerance (Carrier baseline) ===
  Within +/- 1 day(s):   2.8 %
  Within +/- 2 day(s):   5.3 %
  Within +/- 3 day(s):   7.6 %
  Within +/- 5 day(s):  12.8 %
  Within +/- 7 day(s):  22.5 %
```

So the headline finding for the report:

> The neural network predicts delivery time with **MAE = 3.57 days
> and R² = 0.49**, and gets within ±2 days of the actual delivery date
> for **47 %** of orders — versus only **5 %** for the platform's
> contractual estimate, an improvement of more than **40 percentage
> points**. The model achieves this *without ever seeing* the carrier's
> prediction during training.

Note about the carrier baseline: Olist's `order_estimated_delivery_date`
is set very generously (mean ≈ 24 days from purchase even when actual
delivery is around 12 days), so the carrier's estimate is essentially
a contractual upper bound rather than a real-time prediction. This is
why its MAE is so large — it is not a "competing model" so much as a
buffered service-level guarantee.

The script also produces five figures, ready to drop into the report:

| File                          | What it shows                                  |
| ----------------------------- | ---------------------------------------------- |
| `training_history.png`        | Loss & MAE per epoch (train + validation)      |
| `pred_vs_actual.png`          | Scatter of model predictions vs ground truth   |
| `error_distribution.png`      | Side-by-side error histograms (model & carrier)|
| `model_vs_carrier_scatter.png`| Side-by-side scatter (model & carrier vs truth)|
| `accuracy_comparison.png`     | Bar chart: % within ±N days, model vs carrier  |

---