import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

n = 10000

# Target: imbalanced ~5% positive class
fraud_flag = rng.choice([0, 1], size=n, p=[0.95, 0.05])

# Base numeric features
age = np.clip(rng.normal(38, 12, n).round(), 18, 80).astype(float)
income = np.clip(rng.lognormal(mean=10.3, sigma=0.55, size=n), 15000, 350000)
balance = np.clip(rng.normal(12000, 18000, n), -5000, 250000)
tenure_months = np.clip(rng.poisson(28, n), 0, 180).astype(float)
transactions_30d = np.clip(rng.poisson(18, n), 0, 220).astype(float)
avg_order_value = np.clip(rng.gamma(shape=2.0, scale=45.0, size=n), 3, 1500)
days_since_last_login = np.clip(rng.exponential(scale=14, size=n), 0, 240)
credit_score = np.clip(rng.normal(670, 70, n), 300, 850)

# Make target-related signal to keep the task learnable
income = income * (1 + fraud_flag * rng.normal(0.12, 0.08, n))
balance = balance + fraud_flag * rng.normal(3500, 7000, n)
transactions_30d = np.clip(transactions_30d + fraud_flag * rng.normal(9, 5, n), 0, 250)
days_since_last_login = np.clip(days_since_last_login + fraud_flag * rng.normal(8, 6, n), 0, 240)
credit_score = np.clip(credit_score - fraud_flag * rng.normal(55, 25, n), 300, 850)

# Categorical features with some messy variants
cities = np.array([
    'Tehran', 'Karaj', 'Mashhad', 'Shiraz', 'Isfahan', 'Tabriz', 'Qom', 'Ahvaz',
    'Rasht', 'Kermanshah', 'Yazd', 'Sari', 'Kerman', 'Qazvin', 'Arak'
], dtype=object)
city = rng.choice(cities, size=n, replace=True)

# Inject inconsistent variants / dirty labels
variant_map = {
    'Tehran': ['tehran', 'TEHRAN', 'Teh ran'],
    'Mashhad': ['mashhad', 'MASHHAD'],
    'Isfahan': ['isfahan', 'Esfahan'],
    'Shiraz': ['shiraz', 'SHIRAZ'],
    'Tabriz': ['tabriz', 'TABRIZ'],
}
for base, variants in variant_map.items():
    idx = np.where(city == base)[0]
    if len(idx):
        pick = rng.choice(idx, size=max(1, len(idx)//8), replace=False)
        city[pick] = rng.choice(variants, size=len(pick), replace=True)

channel = rng.choice(['web', 'mobile', 'branch', 'call_center', 'partner'], size=n, p=[0.42, 0.34, 0.14, 0.06, 0.04])
device = rng.choice(['android', 'ios', 'windows', 'macos', 'linux', 'other'], size=n, p=[0.44, 0.18, 0.18, 0.12, 0.05, 0.03])
plan_type = rng.choice(['basic', 'silver', 'gold', 'platinum'], size=n, p=[0.50, 0.27, 0.17, 0.06])
merchant_category = rng.choice(
    [
        'grocery', 'electronics', 'fashion', 'travel',
        'education', 'gaming', 'health', 'home',
        'food', 'automotive', 'luxury', 'utilities'
    ],
    size=n,
    p=[0.16, 0.12, 0.11, 0.08, 0.09, 0.06, 0.11, 0.08, 0.11, 0.04, 0.03, 0.01]
)

region = rng.choice(['north', 'south', 'east', 'west', 'central'], size=n, p=[0.20,0.18,0.21,0.16,0.25])
signup_source = rng.choice(['organic', 'ad', 'referral', 'affiliate', 'event', 'unknown'], size=n, p=[0.34,0.28,0.16,0.08,0.06,0.08])

# High-cardinality identifiers / categories
customer_id = [f"CUST-{i:06d}" for i in range(1, n+1)]
postal_code = np.array([f"{rng.integers(10000, 99999)}" for _ in range(n)], dtype=object)
device_model = np.array([f"M{rng.integers(100,999)}-{rng.choice(['A','B','C','X','Z'])}{rng.integers(10,99)}" for _ in range(n)], dtype=object)
merchant_id = np.array([f"MCH-{rng.integers(100000,999999)}" for _ in range(n)], dtype=object)
product_sku = np.array([f"SKU-{rng.integers(100000,999999)}" for _ in range(n)], dtype=object)

# Date feature
start_date = datetime(2021, 1, 1)
signup_date = np.array([start_date + timedelta(days=int(x)) for x in rng.integers(0, 1800, size=n)], dtype='datetime64[ns]')

# Add some text column with weak signal
notes_pool = np.array([
    'ok', 'late payment', 'verified', 'manual review', 'coupon applied',
    'chargeback', 'vip', 'new device', 'high value', 'suspicious pattern'
], dtype=object)
notes = rng.choice(notes_pool, size=n, p=[0.18,0.08,0.22,0.10,0.08,0.05,0.08,0.10,0.06,0.05])
# Increase suspicious notes for positive class
pos_idx = np.where(fraud_flag == 1)[0]
boost = rng.choice(pos_idx, size=max(1, len(pos_idx)//2), replace=False)
notes[boost] = rng.choice(['chargeback', 'manual review', 'suspicious pattern', 'new device'], size=len(boost), replace=True)

# Create noisy / dirty numeric anomalies
outlier_idx = rng.choice(np.arange(n), size=90, replace=False)
income[outlier_idx[:20]] *= rng.uniform(3, 8, size=20)
balance[outlier_idx[20:40]] *= rng.uniform(3, 10, size=20)
avg_order_value[outlier_idx[40:55]] *= rng.uniform(6, 20, size=15)
transactions_30d[outlier_idx[55:70]] += rng.integers(80, 200, size=15)
credit_score[outlier_idx[70:90]] -= rng.uniform(120, 260, size=20)
credit_score = np.clip(credit_score, 300, 850)

# Dirty impossible values / typos in a small fraction
age[rng.choice(n, size=15, replace=False)] = rng.choice([-5, 0, 120], size=15, replace=True)
income[rng.choice(n, size=10, replace=False)] = rng.choice([-1000, 0], size=10, replace=True)

# Missing values: random + blockwise
cols_numeric = ['age','income','balance','tenure_months','transactions_30d','avg_order_value','days_since_last_login','credit_score']
cols_categorical = ['city','channel','device','plan_type','merchant_category','region','signup_source','device_model','postal_code']

# Random missingness
for arr_name, arr in [('age', age), ('income', income), ('balance', balance), ('tenure_months', tenure_months),
                      ('transactions_30d', transactions_30d), ('avg_order_value', avg_order_value),
                      ('days_since_last_login', days_since_last_login), ('credit_score', credit_score)]:
    miss = rng.choice(n, size=int(n * rng.uniform(0.04, 0.11)), replace=False)
    arr[miss] = np.nan

for arr in [city, channel, device, plan_type, merchant_category, region, signup_source, device_model, postal_code, notes]:
    miss = rng.choice(n, size=int(n * rng.uniform(0.02, 0.08)), replace=False)
    arr[miss] = None

# Block missingness in one numeric and one categorical feature
block_start = 1200
balance[block_start:block_start+140] = np.nan
merchant_category[2400:2480] = None

# Duplicate some rows intentionally
base = pd.DataFrame({
    'customer_id': customer_id,
    'signup_date': signup_date,
    'age': age,
    'income': income,
    'balance': balance,
    'tenure_months': tenure_months,
    'transactions_30d': transactions_30d,
    'avg_order_value': avg_order_value,
    'days_since_last_login': days_since_last_login,
    'credit_score': credit_score,
    'city': city,
    'region': region,
    'channel': channel,
    'device': device,
    'device_model': device_model,
    'plan_type': plan_type,
    'merchant_category': merchant_category,
    'signup_source': signup_source,
    'postal_code': postal_code,
    'merchant_id': merchant_id,
    'product_sku': product_sku,
    'notes': notes,
    'fraud_flag': fraud_flag,
})

dup_rows = base.sample(70, random_state=42)
df = pd.concat([base, dup_rows], ignore_index=True)

df['risk_review_score'] = (df['fraud_flag'] * rng.normal(0.8, 0.1, len(df))
                           + (1 - df['fraud_flag']) * rng.normal(0.2, 0.1, len(df))).clip(0, 1)

flip = rng.random(len(df)) < 0.02
df['account_frozen'] = np.where(flip, 1 - df['fraud_flag'], df['fraud_flag'])

noisy = rng.choice(len(df), size=int(0.04 * len(df)), replace=False)
df.loc[noisy, 'fraud_flag'] = 1 - df.loc[noisy, 'fraud_flag']

df['income_usd'] = df['income'] / 42000 + rng.normal(0, 0.5, len(df))
df['monthly_spend'] = df['transactions_30d'] * df['avg_order_value'] * rng.uniform(0.9, 1.1, len(df))
df['credit_score_bucket_num'] = (df['credit_score'] // 50) * 50

df['branch_code'] = rng.integers(100, 300, len(df))
df['error_code'] = rng.choice([0, 101, 205, 404, 500], size=len(df), p=[0.7, 0.1, 0.08, 0.07, 0.05])

for i in range(5):
    df[f'noise_{i}'] = rng.normal(0, 1, len(df))
df['row_hash'] = rng.integers(1e9, 9e9, len(df))

# Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Make some columns read as object/strings with mixed blanks
for col in ['city', 'channel', 'device', 'plan_type', 'merchant_category', 'region', 'signup_source', 'device_model', 'postal_code', 'notes']:
    df[col] = df[col].astype(object)
    # introduce a few empty strings
    idx = rng.choice(df.index, size=max(5, len(df)//200), replace=False)
    df.loc[idx, col] = ''

# Save files
csv_path = 'dirty_imbalanced_customer_risk.csv'
df.to_csv(csv_path, index=False)

# Quick profile report
profile = {
    'rows': int(len(df)),
    'cols': int(df.shape[1]),
    'target_positive_rate': float(df['fraud_flag'].mean()),
    'missing_values_total': int(df.isna().sum().sum()),
    'duplicate_rows_exact': int(df.duplicated().sum()),
    'numeric_columns': cols_numeric,
    'categorical_columns': cols_categorical,
}

print(profile)
print(csv_path)