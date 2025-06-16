import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.utils import resample
from scipy.stats import ttest_1samp
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# ✅ Path to extracted folder, not ZIP
data_path = r"C:\Users\USER\Documents\SECOM PRESENTATION DATASET"

# ✅ Using os.path.join for safety
secom_data = pd.read_csv(os.path.join(data_path, 'secom.data'), sep='\s+', header=None, encoding='latin1')

# ✅ Only loading the label column (2nd column), ignoring the datetime
secom_labels = pd.read_csv(os.path.join(data_path, 'secom_labels.data'), sep='\s+', header=None, encoding='latin1', usecols=[1])

# ✅ Drop any NaNs in labels (if they exist)
secom_labels.dropna(inplace=True)

# Handling missing values
imputer = SimpleImputer(strategy='mean')
secom_data_imputed = imputer.fit_transform(secom_data)

# Feature scaling
scaler = StandardScaler()
secom_data_scaled = scaler.fit_transform(secom_data_imputed)

# Feature selection with Lasso
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(secom_data_scaled, secom_labels.values.ravel())
coef = lasso.coef_

# Select features where coefficients are not zero
selected_features = np.where(coef != 0)[0]
X_selected = secom_data_scaled[:, selected_features]
print(f"Number of selected features: {len(selected_features)}")

# Bootstrapping for confidence intervals
n_iterations = 1000
n_size = len(X_selected)
bootstrap_means = []
for _ in range(n_iterations):
    sample = resample(X_selected[:, 0], n_samples=n_size)
    bootstrap_means.append(np.mean(sample))
bootstrap_means = np.array(bootstrap_means)

# Calculate 95% confidence interval
ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)
print(f"95% confidence interval for mean of first selected feature: ({ci_lower}, {ci_upper})")

# Hypothesis testing
t_stat, p_value = ttest_1samp(X_selected[:, 0], 0)
print(f"T-statistic: {t_stat}, P-value: {p_value}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, secom_labels.values.ravel(), test_size=0.3, random_state=42)

# Model training with XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature importance visualization
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[sorted_idx])
plt.xlabel("Feature Index (sorted by importance)")
plt.ylabel("Importance")
plt.title("Feature Importances (XGBoost)")
plt.show()
