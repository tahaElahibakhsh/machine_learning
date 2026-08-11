# ML Data-Cleaning Practice Dataset

A deliberately messy, synthetic dataset packed with classic ML / data-cleaning traps.
Below you'll find (1) a hands-on practice roadmap (no code provided — write it yourself), and (2) a full list of the challenges intentionally planted in the data.

---

## Part 1 — Practice Roadmap (no code, do it yourself)

### Phase 1 — Know Your Data (EDA)
- Run `info()`, compute the null percentage and unique count per column. Rule of thumb: any column whose `nunique` is roughly equal to the number of rows is probably an ID and useless.
- Plot histograms and boxplots of numeric columns → look for skew (`income` is log-normal). Also compute skewness numerically.
- Correlation matrix → it should expose `income_usd`, `monthly_spend`, and `credit_score_bucket_num` as redundant.
- **Leakage detection:** for each feature, compute a single-feature AUC (or mutual information with the target). Anything with AUC close to 1 is suspicious → `account_frozen` and `risk_review_score` should blow up here. This is a habit worth building.

### Phase 2 — Cleaning
- Find and drop duplicate rows — **before** the split, otherwise the same row lands in both train and test (which is itself a form of leakage).
- Convert impossible values (`age = -5`, `income = -1000`) to `NaN`.
- Convert empty strings `''` to `NaN` (pandas does **not** treat them as missing!).
- Normalize city variants (`tehran`, `TEHRAN`, `Esfahan`, …).
- Cast `branch_code` and `error_code` to categorical even though they're numeric — dtype alone doesn't tell you what a column means.

### Phase 3 — Splitting Properly
- Use a **stratified** split because of the class imbalance. Also try once without stratification and see how badly the positive-class ratio drifts.
- Golden rule: every `fit` (imputer, scaler, encoder) happens **only on train**.

### Phase 4 — Imputers (the fun sklearn part)
- `SimpleImputer` with `add_indicator=True` — missingness itself can carry signal (that 140-row `balance` block).
- Try `KNNImputer` and `IterativeImputer` (MICE) on the same columns and compare with CV to see which helps the model more.
- There's also a standalone `MissingIndicator`.

### Phase 5 — Less Common Scalers / Transformers
- `RobustScaler` (because of outliers), `PowerTransformer(method='yeo-johnson')`, and `QuantileTransformer` (for the heavily skewed `income`). Plot before/after.
- See the difference between `Normalizer` (row-wise, each row's norm = 1) and `StandardScaler` (column-wise) in practice — people confuse these all the time.
- Winsorize / clip for outliers, and `IsolationForest` or `LocalOutlierFactor` to detect them.

### Phase 6 — Encoding
- `OneHotEncoder` with `min_frequency` and `max_categories` (to bucket rare categories) and `handle_unknown='infrequent_if_exist'`.
- sklearn's own `TargetEncoder` (v1.3+) for high-cardinality columns like `merchant_id` — and understand why it **must** live inside a Pipeline/CV, otherwise it leaks.
- Engineer features from `signup_date`: account age, month, day of week.

### Phase 7 — Pipeline & ColumnTransformer
- Build a `ColumnTransformer` with three branches: numeric (imputer → scaler), categorical (imputer → encoder), and passthrough/drop.
- Write your own custom transformer (a class with `fit`/`transform`, or `FunctionTransformer`) — e.g. for city-name normalization or clipping — so the entire cleaning step lives inside the pipeline.

### Phase 8 — Imbalance & Label Noise
- Baseline with `class_weight='balanced'`, then threshold tuning (forget accuracy entirely; look at PR-AUC, F1, recall).
- `SMOTE` from imblearn — make sure it's applied **only on train** (use imblearn's own Pipeline).
- For flipped labels: the *confident learning* idea — list samples the model confidently predicts against their label; you'll likely find that corrupted ~4%.

### Phase 9 — Dropping the Useless Stuff
- `VarianceThreshold`, `mutual_info_classif`, and `permutation_importance` → `noise_*` and `row_hash` should fall out, and the leaky features should expose themselves again.

---

## Part 2 — Challenges Planted in the Dataset

This dataset intentionally embeds several classic ML / data-cleaning challenges:

### 1. Class Imbalance + Label Noise
- The positive class is only ~5%, and on top of that 4% of labels were flipped (the `noisy` block). Accuracy is meaningless — go for PR-AUC / F1 / recall, and know there's a performance ceiling caused by label noise.

### 2. Data Leakage — The Most Important Trap
- `risk_review_score` is built directly from `fraud_flag` (mean 0.8 for positives, 0.2 for negatives).
- `account_frozen` **is** `fraud_flag` with only 2% flips.
- If you don't drop these two, the model looks great but is effectively cheating.

### 3. Duplicate Rows + Split Leakage Risk
- 70 rows were duplicated verbatim and the whole dataframe was shuffled; if you split train/test before deduplicating, the same record ends up on both sides.

### 4. Missing Values in Multiple Flavors
- Random missingness: 4–11% in numeric columns, 2–8% in categoricals.
- Block (non-random) missingness: `balance` in 140 consecutive rows, `merchant_category` in 80 rows.
- Empty strings `''` in object columns, which pandas does **not** count as NaN — hidden missing values.

### 5. Dirty, Inconsistent Categoricals
- City variants: `tehran`, `TEHRAN`, `Teh ran`, `Esfahan`, … which must be normalized, otherwise one city becomes several categories.

### 6. High-Cardinality / ID-like Columns
- `customer_id`, `merchant_id`, `product_sku`, `postal_code`, `device_model` — one-hot explodes; either drop them or use target/frequency encoding carefully (overfit/leakage risk).

### 7. Outliers & Impossible Values
- Injected outliers in `income`, `balance`, `avg_order_value`, `transactions`, `credit_score`.
- Nonsensical values: ages of -120, -5, and 0; negative income — need validation plus clip/removal.

### 8. Redundant Features & Multicollinearity
- `income_usd` is a noisy copy of `income`; `monthly_spend` is the product of two existing features; `credit_score_bucket_num` is a bucketed version of `credit_score`.
- `noise_0..4`, `row_hash`, `branch_code`, `error_code` are pure random noise and should be removed during feature selection.

### 9. Features That Need Engineering
- `signup_date` (extract year / month / account age), and `notes` — text with a weak but real signal (`chargeback`, `suspicious pattern` correlate with the positive class).

---

## TL;DR
The real signal lives in `income`, `balance`, `transactions`, `days_since_last_login`, `credit_score`, and `notes`.
The main traps are leakage (`risk_review_score`, `account_frozen`), duplicates before the split, and `''` as hidden missing values.
`
