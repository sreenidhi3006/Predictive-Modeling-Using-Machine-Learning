# ============================================
# Predictive Modeling Using Machine Learning
# Step 5: Load and Inspect the Dataset
# ============================================

# Import libraries
import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/customer_churn.csv")

# Display the first 5 rows
print("\nFirst 5 Rows of the Dataset:")
print(df.head())

# Display the dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns)

# Display data types
print("\nData Types:")
print(df.dtypes)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())
# Display unique values in each column
print("\nUnique Values in Each Column:\n")

for column in df.columns:
    print(f"{column}:")
    print(df[column].unique())
    print("-" * 50)
    # ============================================
# Step 7: Data Cleaning
# ============================================

# Remove customerID column
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove rows with missing values
df.dropna(inplace=True)

# Convert Churn to numeric
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Display cleaned dataset information
print("\n========== Cleaned Dataset ==========")
print(df.head())

print("\nDataset Shape After Cleaning:")
print(df.shape)

print("\nData Types After Cleaning:")
print(df.dtypes)
# ============================================
# Step 8: Encode Categorical Columns
# ============================================

from sklearn.preprocessing import LabelEncoder

# Create LabelEncoder object
label_encoder = LabelEncoder()

# Convert all object (text) columns to numbers
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = label_encoder.fit_transform(df[column])

print("\n========== Encoded Dataset ==========")
print(df.head())

print("\nData Types After Encoding:")
print(df.dtypes)
# ============================================
# Step 9: Split the Dataset
# ============================================

from sklearn.model_selection import train_test_split

# Features (all columns except Churn)
X = df.drop("Churn", axis=1)

# Target (Churn column)
y = df["Churn"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n========== Dataset Split ==========")
print("Training Features :", X_train.shape)
print("Testing Features  :", X_test.shape)
print("Training Labels   :", y_train.shape)
print("Testing Labels    :", y_test.shape)
# ============================================
# Step 10: Train Decision Tree Model
# ============================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create the Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42)

# Train the model
decision_tree.fit(X_train, y_train)

# Make predictions
y_pred_dt = decision_tree.predict(X_test)

# Calculate accuracy
dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("\n========== Decision Tree Results ==========")
print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
# ============================================
# Step 11: Train Random Forest Model
# ============================================

from sklearn.ensemble import RandomForestClassifier

# Create the Random Forest model
random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
random_forest.fit(X_train, y_train)

# Make predictions
y_pred_rf = random_forest.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\n========== Random Forest Results ==========")
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
# ============================================
# Step 12: Train Logistic Regression Model
# ============================================

from sklearn.linear_model import LogisticRegression

# Create the Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
logistic_model.fit(X_train, y_train)

# Make predictions
y_pred_lr = logistic_model.predict(X_test)

# Calculate accuracy
lr_accuracy = accuracy_score(y_test, y_pred_lr)

print("\n========== Logistic Regression Results ==========")
print(f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")
# ============================================
# Step 13: Confusion Matrix
# ============================================

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Create confusion matrix using the Random Forest predictions
cm = confusion_matrix(y_test, y_pred_rf)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")

plt.title("Random Forest - Confusion Matrix")

# Save the image
plt.savefig("output/confusion_matrix.png")

# Show the plot
plt.show()

print("\nConfusion Matrix saved successfully!")
# ============================================
# Step 14: ROC Curve
# ============================================

from sklearn.metrics import roc_curve, auc

# Get prediction probabilities
y_prob = random_forest.predict_proba(X_test)[:, 1]

# Calculate ROC values
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

# Save the figure
plt.savefig("output/roc_curve.png")

# Show the graph
plt.show()

print("ROC Curve saved successfully!")
# ============================================
# Step 15: Feature Importance
# ============================================

# Get feature importance values
importance = random_forest.feature_importances_

# Create a DataFrame
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

# Sort by importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== Feature Importance ==========")
print(feature_importance)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.bar(feature_importance["Feature"], feature_importance["Importance"])
plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

# Save graph
plt.savefig("output/feature_importance.png")

# Show graph
plt.show()

print("Feature Importance graph saved successfully!")
# ============================================
# Step 16: Accuracy Comparison Chart
# ============================================

# Model names
models = [
    "Decision Tree",
    "Random Forest",
    "Logistic Regression"
]

# Accuracy values (convert to percentage)
accuracies = [
    dt_accuracy * 100,
    rf_accuracy * 100,
    lr_accuracy * 100
]

# Create bar chart
plt.figure(figsize=(8, 5))
plt.bar(models, accuracies)

plt.title("Machine Learning Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

# Display accuracy values on top of bars
for i, value in enumerate(accuracies):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.tight_layout()

# Save the chart
plt.savefig("output/model_accuracy.png")

# Show the chart
plt.show()

print("Model Accuracy Comparison chart saved successfully!")
# ============================================
# Step 17: Save the Random Forest Model
# ============================================

import joblib

# Save the trained Random Forest model
joblib.dump(random_forest, "model/random_forest_model.pkl")

print("\n✅ Random Forest model saved successfully!")
print("Location: model/random_forest_model.pkl")