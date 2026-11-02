import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,accuracy_score, precision_score, recall_score, f1_score,log_loss, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)


# 1. LOAD DATASET
phones = pd.read_csv(r"C:/Users/ASUS/Downloads/phone_data.csv", on_bad_lines="skip")
phones.columns = phones.columns.str.strip().str.lower()      # sab column names small letters me

print("First 5 rows:\n", phones.head(), "\n")
print("Shape (rows, columns):", phones.shape, "\n")
print("Dataset info:\n")
phones.info()
print("\nMissing values in each column:\n", phones.isna().sum(), "\n")

# 2. DATA CLEANING (text -> numbers)

clean = pd.DataFrame()
clean["brand"] = phones["brand"]

clean["year"] = phones["announced"].astype(str).str.extract(r"((?:19|20)\d{2})")[0].astype(float)


clean["weight_g"] = pd.to_numeric(phones["weight_g"], errors="coerce")

display_text = phones["display_size"].fillna("") + " | " + phones["display_resolution"].fillna("")
clean["display_inch"] = display_text.str.extract(r"(\d+\.?\d*)\s*inches")[0].astype(float)
pixels = display_text.str.extract(r"(\d{2,5})\s*x\s*(\d{2,5})\s*pixels")
clean["resolution_mp"] = (pixels[0].astype(float) * pixels[1].astype(float)) / 1e6

ram = phones["ram"].astype(str).str.extract(r"(\d+\.?\d*)\s*(GB|MB)", flags=re.I)
clean["ram_gb"] = np.where(ram[1].str.upper() == "GB", ram[0].astype(float), ram[0].astype(float) / 1024)

storage = phones["internal_memory"].astype(str).str.extract(r"(\d+\.?\d*)\s*(GB|MB)", flags=re.I)
clean["storage_gb"] = np.where(storage[1].str.upper() == "GB", storage[0].astype(float), storage[0].astype(float) / 1024)

clean["battery_mah"] = phones["battery"].astype(str).str.extract(r"(\d{3,5})\s*mAh")[0].astype(float)

clean["camera_mp"] = phones["primary_camera"].astype(str).str.extract(r"(\d+\.?\d*)\s*MP")[0].astype(float)

clean["has_nfc"] = phones["nfc"].astype(str).str.contains("Yes").astype(int)
clean["has_audio_jack"] = phones["audio_jack"].astype(str).str.contains("Yes").astype(int)
clean["has_4g"] = phones["network_technology"].astype(str).str.contains("LTE").astype(int)
clean["is_android"] = phones["os"].astype(str).str.contains("Android").astype(int)

clean["price"] = phones["approx_price_eur"]

clean.loc[clean["year"] > 2026, "year"] = np.nan
clean.loc[clean["display_inch"] > 20, "display_inch"] = np.nan
clean.loc[clean["weight_g"] > 2000, "weight_g"] = np.nan
clean.loc[clean["battery_mah"] > 30000, "battery_mah"] = np.nan

print("Missing values after cleaning:\n", clean.isna().sum(), "\n")

clean = clean.dropna(subset=["price"])
clean = clean[clean["price"] > 0]

num_cols = ["year", "weight_g", "display_inch", "resolution_mp", "ram_gb",
            "storage_gb", "battery_mah", "camera_mp"]
for c in num_cols:
    clean[c] = clean[c].fillna(clean[c].median())

print("Rows used for ML (price known):", len(clean), "\n")

# 3. EDA (Exploratory Data Analysis)

print("Summary of price:\n", clean["price"].describe(), "\n")

plt.figure(figsize=(8, 6))
plt.hist(clean["price"], bins=40, color="skyblue", edgecolor="black")
plt.title("Histogram of Phone Price (EUR)")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

print("Top 10 brands:\n", clean["brand"].value_counts().head(10), "\n")
plt.figure(figsize=(9, 5))
clean["brand"].value_counts().head(10).plot(kind="bar", color="orange")
plt.title("Top 10 Brands")
plt.ylabel("Number of phones")
plt.show()

yearly = clean.groupby("year").size()
plt.figure(figsize=(10, 5))
plt.bar(yearly.index, yearly.values, color="green")
plt.title("Phones per Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()

corr_cols = num_cols + ["has_nfc", "has_audio_jack", "has_4g", "is_android", "price"]
corr = clean[corr_cols].corr()
print("Correlation with price:\n", corr["price"].sort_values(ascending=False), "\n")

plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(clean["ram_gb"], clean["price"], alpha=0.4, color="green")
plt.title("RAM vs Price")
plt.xlabel("RAM (GB)")
plt.ylabel("Price (EUR)")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(clean["battery_mah"], clean["price"], alpha=0.4, color="purple")
plt.title("Battery vs Price")
plt.xlabel("Battery (mAh)")
plt.ylabel("Price (EUR)")
plt.show()

# 4. ENCODING + NEW FEATURES

top_brands = clean["brand"].value_counts().head(15).index
clean["brand"] = np.where(clean["brand"].isin(top_brands), clean["brand"], "Other")
le = LabelEncoder()
clean["brand"] = le.fit_transform(clean["brand"])

clean["price_cat"] = pd.qcut(clean["price"], 3, labels=[0, 1, 2]).astype(int)
class_names = ["Budget", "Mid-range", "Premium"]

features = num_cols + ["has_nfc", "has_audio_jack", "has_4g", "is_android", "brand"]
X = clean[features]

# 5. REGRESSION (price predict karna)
#    Model 1: Multiple Linear Regression
#    Model 2: Polynomial Regression

plt.figure(figsize=(8, 6))
plt.hist(np.log(clean["price"]), bins=40, color="orange", edgecolor="black")
plt.title("Histogram of log(Price)")
plt.xlabel("log(Price)")
plt.ylabel("Frequency")
plt.show()

y = np.log(clean["price"])

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=123)

scaler = StandardScaler()
Xtrainscaled = scaler.fit_transform(Xtrain)
Xtestscaled = scaler.transform(Xtest)

lr = LinearRegression()
lr.fit(Xtrainscaled, ytrain)
ypred = lr.predict(Xtestscaled)

mae = mean_absolute_error(ytest, ypred)
mse = mean_squared_error(ytest, ypred)
rmse = np.sqrt(mse)
r2 = r2_score(ytest, ypred)

print("Multiple Linear Regression - Model Evaluation Metrics (log price):")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R-squared (R²): {r2:.3f}\n")

coef = pd.Series(lr.coef_, index=features).sort_values()
print("Coefficients (bada value = price pe zyada asar):\n", coef, "\n")

plt.figure(figsize=(8, 6))
plt.scatter(ytest, ypred, color="blue", alpha=0.6, label="Predicted vs Actual")
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color="red", linewidth=2, label="Ideal Fit Line")
plt.title("Actual vs Predicted log(Price) - Linear Regression")
plt.xlabel("Actual log(Price)")
plt.ylabel("Predicted log(Price)")
plt.legend()
plt.grid(True)
plt.show()

poly = PolynomialFeatures(degree=2, include_bias=False)
Xtrainpoly = poly.fit_transform(Xtrainscaled)
Xtestpoly = poly.transform(Xtestscaled)

lr_poly = LinearRegression()
lr_poly.fit(Xtrainpoly, ytrain)
ypred_poly = lr_poly.predict(Xtestpoly)

mae_p = mean_absolute_error(ytest, ypred_poly)
mse_p = mean_squared_error(ytest, ypred_poly)
rmse_p = np.sqrt(mse_p)
r2_p = r2_score(ytest, ypred_poly)

print("Polynomial Regression (degree 2) - Model Evaluation Metrics (log price):")
print(f"Mean Absolute Error (MAE): {mae_p:.3f}")
print(f"Mean Squared Error (MSE): {mse_p:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse_p:.3f}")
print(f"R-squared (R²): {r2_p:.3f}\n")

reg_results = pd.DataFrame({
    "Model": ["Multiple Linear Regression", "Polynomial Regression (deg 2)"],
    "MAE": [mae, mae_p],
    "MSE": [mse, mse_p],
    "RMSE": [rmse, rmse_p],
    "R2": [r2, r2_p],
}).round(3)
print("Regression Comparison:\n", reg_results.to_string(index=False), "\n")

# 6. CLASSIFICATION (price category predict karna)
#    KNN, Naive Bayes, Decision Tree, SVM, Logistic Regression

yc = clean["price_cat"]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, yc, test_size=0.2, random_state=123, stratify=yc)

scaler = StandardScaler()
Xtrainscaled = scaler.fit_transform(Xtrain)
Xtestscaled = scaler.transform(Xtest)

models = {
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=123),
    "SVM": SVC(kernel="rbf", probability=True, random_state=123),
    "Logistic Regression": LogisticRegression(max_iter=1000),
}

results = []
for name, model in models.items():
    model.fit(Xtrainscaled, ytrain)
    pred = model.predict(Xtestscaled)
    proba = model.predict_proba(Xtestscaled)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(ytest, pred),
        "Precision": precision_score(ytest, pred, average="macro"),
        "Recall": recall_score(ytest, pred, average="macro"),
        "F1 Score": f1_score(ytest, pred, average="macro"),
        "Log Loss": log_loss(ytest, proba),
        "AUC": roc_auc_score(ytest, proba, multi_class="ovr"),
    })

    print(name, "- Confusion Matrix:\n", confusion_matrix(ytest, pred), "\n")
    ConfusionMatrixDisplay.from_predictions(ytest, pred, display_labels=class_names, cmap="Blues")
    plt.title("Confusion Matrix - " + name)
    plt.show()

results = pd.DataFrame(results).round(3)
print("Classification Results:\n", results.to_string(index=False), "\n")

results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(kind="bar", figsize=(10, 5), rot=0)
plt.title("Comparison of Classification Models")
plt.ylim(0, 1)
plt.grid(True)
plt.show()

plt.figure(figsize=(16, 7))
plot_tree(DecisionTreeClassifier(max_depth=3, random_state=123).fit(Xtrainscaled, ytrain),
          feature_names=features, class_names=class_names, filled=True, fontsize=8)
plt.title("Decision Tree (depth 3)")
plt.show()

best = results.sort_values("Accuracy", ascending=False).iloc[0]
print("Best classification model:", best["Model"], "with accuracy", best["Accuracy"])