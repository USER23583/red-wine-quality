import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, classification_report
import joblib
import warnings
warnings.filterwarnings("ignore")

print("Loading Dataset...\n")


df = pd.read_csv("winequality-red.csv", header=0)

if df.shape[1] == 1:
    df = df.iloc[:, 0].str.split(",", expand=True)

df.columns = [
    "fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides",
    "free_sulfur_dioxide","total_sulfur_dioxide","density","pH","sulphates","alcohol","quality"
]

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("Dataset Loaded Successfully!")
print(f"Total Samples: {len(df)}\n")

print("\n--- SAMPLE DATA ---")
print(df.head(), "\n")


print("="*50)
print("EXPLORATORY DATA ANALYSIS: QUALITY")
print("="*50)
print(f"Quality Range: {df['quality'].min()} - {df['quality'].max()}")
print(f"Average Quality: {df['quality'].mean():.2f}")
print(f"Most Common Score: {df['quality'].mode()[0]}")

df['quality'].value_counts().sort_index().plot(kind="bar")
plt.title("Wine Quality Distribution")
plt.xlabel("Quality Score")
plt.ylabel("Count")
plt.show()


print("\nCorrelation matrix:")
print(df.corr()['quality'].sort_values(ascending=False), "\n")


df["alcohol_acidity_ratio"] = df["alcohol"] / (df["volatile_acidity"] + 1e-5)

features = ["alcohol","volatile_acidity","sulphates","citric_acid","alcohol_acidity_ratio"]
X = df[features]
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training: {len(X_train)} | Testing: {len(X_test)}\n")

print("Training Linear Regression Model...\n")
lr = LinearRegression()
lr.fit(X_train, y_train)

cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring="r2")
print(f"CV R2 Scores: {cv_scores}")
print(f"Mean R2: {cv_scores.mean():.3f}")

pred_lr = lr.predict(X_test)
print(f"Test R2: {r2_score(y_test, pred_lr):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred_lr)):.3f}\n")


print("Training Random Forest Regressor...\n")
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)
print(f"R2 Score: {r2_score(y_test, pred_rf):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred_rf)):.3f}\n")


df["quality_category"] = pd.cut(df["quality"],
                                bins=[2, 4.5, 6.5, 9],
                                labels=["Low", "Medium", "High"],
                                include_lowest=True)

y_cat = df["quality_category"]
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(Xc_train, yc_train)
pred_cat = clf.predict(Xc_test)

print("Classification Report:")
print(classification_report(yc_test, pred_cat), "\n")


joblib.dump(rf, "wine_model.pkl")
print("Model Saved Successfully: wine_model.pkl\n")

def predict(alcohol, volatile_acid, sulphates, citric_acid):
    ratio = alcohol / (volatile_acid + 1e-5)
    new = pd.DataFrame([[alcohol, volatile_acid, sulphates, citric_acid, ratio]], columns=features)
    result = rf.predict(new)[0]
    return result

print("Sample Predictions:")
print(f"Prediction Example 1: {predict(13.5, 0.3, 0.8, 0.4):.2f}/10")
print(f"Prediction Example 2: {predict(8.7, 1.1, 0.2, 0.1):.2f}/10")
print(f"Prediction Example 3: {predict(10.0, 0.5, 0.6, 0.3):.2f}/10\n")
print("All tasks completed successfully!")