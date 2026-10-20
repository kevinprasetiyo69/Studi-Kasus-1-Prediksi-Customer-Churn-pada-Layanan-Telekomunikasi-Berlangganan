import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. SIMULASI DATA (Realitas Industri)
np.random.seed(42)
n_users = 1200

tenure_months = np.random.randint(1, 72, size=n_users)
monthly_charges = np.random.uniform(20.0, 120.0, size=n_users)
contract_type = np.random.choice(
    ['Month-to-month', 'One year', 'Two year'],
    size=n_users,
    p=[0.55, 0.25, 0.20],
)
support_tickets = np.random.poisson(lam=1.5, size=n_users)

churn_score = (
    (contract_type == 'Month-to-month').astype(int) * 1.8
    + (support_tickets >= 3).astype(int) * 1.2
    - (tenure_months / 24)
    + np.random.normal(0, 0.8, size=n_users)
)
churn = (churn_score > 0.5).astype(int)

df = pd.DataFrame({
    'customer_id': [f'CUST-{i:05d}' for i in range(1, n_users + 1)],
    'tenure_months': tenure_months,
    'monthly_charges': monthly_charges,
    'contract_type': contract_type,
    'support_tickets': support_tickets,
    'cancellation_reason': np.where(churn == 1, 'Price/Competitor', 'Active'),
    'churn': churn,
})

# Missing value injection
df.loc[np.random.choice(n_users, 30), 'monthly_charges'] = np.nan

# 2. DATA PREPARATION (Mitigasi GIGO & Data Leakage)
df['monthly_charges'] = df['monthly_charges'].fillna(df['monthly_charges'].median())

# Hapus ID & variabel pasca-kejadian (Leakage)
X = df.drop(columns=['customer_id', 'cancellation_reason', 'churn'])
y = df['churn']
X = pd.get_dummies(X, columns=['contract_type'], drop_first=True)

# 3. PEMODELAN & EVALUASI (Mitigasi Imbalance Data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=100, max_depth=5, class_weight='balanced', random_state=42
)
model.fit(X_train_scaled, y_train)

y_probs = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_probs >= 0.5).astype(int)

print('=== EVALUASI PERFORMA MODEL ===')
print(classification_report(y_test, y_pred, target_names=['Retain', 'Churn']))
print(f'ROC-AUC Score: {roc_auc_score(y_test, y_probs):.4f}')

# 4. TINGKAT PRESKRIPTIF (Decision Engine)
test_results = X_test.copy()
test_results['churn_probability'] = y_probs

def assign_retention_action(prob):
  if prob >= 0.70:
    return 'Diskon Kontrak 1 Thn + Prioritas CS'
  elif prob >= 0.40:
    return 'Voucher Kuota Tambahan'
  return 'Layanan Standar'

test_results['action'] = test_results['churn_probability'].apply(assign_retention_action)
print('\n=== OUTPUT PRESKRIPTIF ===')
print(test_results[['tenure_months', 'churn_probability', 'action']].head(5))