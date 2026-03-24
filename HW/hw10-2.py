import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.ensemble import GradientBoostingClassifier


df = pd.read_csv("sales_data.csv")
df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])

# Sort by date (important for time-based logic)
df = df.sort_values('Sale_Date')


# 3. CREATE REALISTIC CUSTOMER IDs

np.random.seed(42)
df['Customer_ID'] = np.random.randint(0, 500, size=len(df))


# 4. SPLIT INTO PAST vs FUTURE

split_date = df['Sale_Date'].quantile(0.7)

past_df = df[df['Sale_Date'] <= split_date]
future_df = df[df['Sale_Date'] > split_date]


# 5. FEATURE ENGINEERING

past_df['Profit'] = (past_df['Unit_Price'] - past_df['Unit_Cost']) * past_df['Quantity_Sold']

customers = past_df.groupby('Customer_ID').agg({
    'Sales_Amount': ['sum', 'mean'],
    'Quantity_Sold': 'sum',
    'Discount': 'mean',
    'Profit': 'sum',
    'Sale_Date': ['min', 'max', 'count']
})

customers.columns = [
    'total_sales', 'avg_sales',
    'total_quantity',
    'avg_discount',
    'total_profit',
    'first_purchase', 'last_purchase', 'purchase_count'
]

# --- Recency (days since last purchase) ---
max_date = past_df['Sale_Date'].max()
customers['recency_days'] = (max_date - customers['last_purchase']).dt.days

# Drop raw dates
customers = customers.drop(['first_purchase', 'last_purchase'], axis=1)


# 6. TARGET = FUTURE RETURN

future_customers = future_df['Customer_ID'].unique()

customers['Target'] = customers.index.isin(future_customers).astype(int)


# 7. CUSTOMER DISTRIBUTION (EXACT VALUES)

counts = customers['Target'].value_counts().sort_index()
labels = ['Not Returning', 'Returning']

total = counts.sum()
percentages = (counts / total * 100).round(2)

print("\nCustomer Future Return Distribution:")
for i, label in enumerate(labels):
    print(f"{label}: {counts[i]} ({percentages[i]}%)")

plt.figure()
bars = plt.bar(labels, counts)

for i, bar in enumerate(bars):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"{counts[i]}\n({percentages[i]}%)",
        ha='center'
    )

plt.title("Future Customer Return Distribution")
plt.ylabel("Customers")
plt.show()


# 8. VISUAL INSIGHTS


# Recency vs Return
plt.figure()
plt.scatter(customers['recency_days'], customers['total_sales'], c=customers['Target'])
plt.xlabel("Recency (days since last purchase)")
plt.ylabel("Total Sales")
plt.title("Customer Behavior (Colored by Return)")
plt.show()


# 9. PREPARE DATA
X = customers.drop(['Target'], axis=1)
y = customers['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# 10. MODEL
model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)


# 11. EVALUATION

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# 12. CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.show()


# 13. ROC CURVE

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# 14. FEATURE IMPORTANCE

importances = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure()
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.title("Feature Importance")
plt.show()

print("\nTop Drivers of Customer Return:\n")
print(importance_df.head())