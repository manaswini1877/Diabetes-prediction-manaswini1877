# ==============================================================================
# DIABETES PREDICTION MACHINE LEARNING PIPELINE
# ==============================================================================
# This script loads the Pima Indians Diabetes Dataset, performs exploratory
# data analysis (EDA), handles missing values, visualizes key insights,
# trains two classifiers (Decision Tree & Random Forest), evaluates them,
# plots feature importances, and saves the best model and scaler for deployment.
# ==============================================================================

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set style for visualizations
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ------------------------------------------------------------------------------
# SECTION 1: Load and Explore the Dataset
# ------------------------------------------------------------------------------
print("=== SECTION 1: Loading & Exploring Dataset ===")

# Check if dataset exists
dataset_path = 'diabetes.csv'
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file '{dataset_path}' not found. Please ensure it is in the working directory.")

# Load dataset
df = pd.read_csv(dataset_path)

# Display shape of the dataset
print(f"Dataset shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check for explicit missing/null values
print("\nExplicit null values per column:")
print(df.isnull().sum())

# Display descriptive statistics of the dataset
print("\nDescriptive statistics:")
print(df.describe())


# ------------------------------------------------------------------------------
# SECTION 2: Handle Missing/Zero Values
# ------------------------------------------------------------------------------
print("\n=== SECTION 2: Handling Missing / Zero Values ===")
# In the Pima Indians dataset, missing values are encoded as 0. 
# Zeros in columns like 'Glucose', 'BMI', and 'BloodPressure' are biologically impossible.
# We will replace these 0 values with the column's median value.

cols_to_clean = ['Glucose', 'BMI', 'BloodPressure']

print("Count of 0 values before replacement:")
for col in cols_to_clean:
    zero_count = (df[col] == 0).sum()
    print(f" - {col}: {zero_count} zero(s)")

# Replace zeros with NaN, compute median, and fill NaNs
for col in cols_to_clean:
    # Replace 0 with NaN so they are excluded from the median calculation
    df[col] = df[col].replace(0, np.nan)
    # Calculate the median of the non-zero (non-NaN) values
    col_median = df[col].median()
    # Impute the missing values with the median
    df[col] = df[col].fillna(col_median)
    print(f"Imputed zeros in '{col}' with median: {col_median:.2f}")

print("\nVerify count of 0 values after replacement:")
for col in cols_to_clean:
    zero_count = (df[col] == 0).sum()
    print(f" - {col}: {zero_count} zero(s)")


# ------------------------------------------------------------------------------
# SECTION 3: Visualize Data
# ------------------------------------------------------------------------------
print("\n=== SECTION 3: Visualizing Data ===")

# Plot 1: Correlation Heatmap
# Helps identify relationships between variables and the target variable (Outcome)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar=True)
plt.title('Correlation Heatmap of Diabetes Features', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
heatmap_filename = 'correlation_heatmap.png'
plt.savefig(heatmap_filename, dpi=300)
plt.close()
print(f"Saved correlation heatmap to '{heatmap_filename}'")

# Plot 2: Outcome Distribution
# Shows class balance (Diabetic vs Non-Diabetic)
plt.figure(figsize=(6, 5))
sns.countplot(x='Outcome', data=df, palette='Set2')
plt.title('Outcome Class Distribution', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Outcome (0 = Non-Diabetic, 1 = Diabetic)', fontsize=12)
plt.ylabel('Count', fontsize=12)
# Add counts on top of bars
ax = plt.gca()
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() + 5),
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.tight_layout()
dist_filename = 'outcome_distribution.png'
plt.savefig(dist_filename, dpi=300)
plt.close()
print(f"Saved outcome distribution to '{dist_filename}'")

# Plot 3: Feature Boxplots
# Box plots help inspect range, spread, medians, and potential outliers for features
plt.figure(figsize=(16, 12))
features_to_plot = df.drop(columns=['Outcome'])
for i, col in enumerate(features_to_plot.columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='Outcome', y=col, data=df, palette='Set3')
    plt.title(f'{col} by Outcome', fontsize=12, fontweight='bold')
    plt.xlabel('Outcome')
    plt.ylabel(col)
plt.tight_layout()
boxplot_filename = 'feature_boxplots.png'
plt.savefig(boxplot_filename, dpi=300)
plt.close()
print(f"Saved feature boxplots to '{boxplot_filename}'")


# ------------------------------------------------------------------------------
# SECTION 4: Feature Scaling and Train-Test Split
# ------------------------------------------------------------------------------
print("\n=== SECTION 4: Feature Scaling & Train-Test Split ===")

# Split features (X) and labels (y)
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split data into training (80%) and testing (20%) sets
# Using random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set features shape: {X_train.shape}")
print(f"Testing set features shape:  {X_test.shape}")

# Initialize StandardScaler to normalize feature ranges
scaler = StandardScaler()

# Fit the scaler on the training features only, and transform both sets.
# This prevents data leakage from the test set to the training set.
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the trained scaler so it can be reused to scale new user inputs in the app
scaler_path = 'scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Saved StandardScaler state to '{scaler_path}'")


# ------------------------------------------------------------------------------
# SECTION 5: Train Classifiers
# ------------------------------------------------------------------------------
print("\n=== SECTION 5: Training Classifiers ===")

# Model 1: Decision Tree Classifier
# Hyperparameters set to restrict tree depth to avoid overfitting
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
print("Training Decision Tree Classifier...")
dt_model.fit(X_train_scaled, y_train)

# Model 2: Random Forest Classifier
# Ensemble method using 100 trees for more robust predictions
rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
print("Training Random Forest Classifier...")
rf_model.fit(X_train_scaled, y_train)


# ------------------------------------------------------------------------------
# SECTION 6: Model Evaluation
# ------------------------------------------------------------------------------
print("\n=== SECTION 6: Model Evaluation ===")

# Predict on test set
dt_preds = dt_model.predict(X_test_scaled)
rf_preds = rf_model.predict(X_test_scaled)

# Calculate accuracy
dt_accuracy = accuracy_score(y_test, dt_preds)
rf_accuracy = accuracy_score(y_test, rf_preds)

print("\n--- DECISION TREE CLASSIFIER ---")
print(f"Test Accuracy: {dt_accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_preds))
print("Classification Report:")
print(classification_report(y_test, dt_preds, target_names=['Non-Diabetic', 'Diabetic']))

print("\n--- RANDOM FOREST CLASSIFIER ---")
print(f"Test Accuracy: {rf_accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))
print("Classification Report:")
print(classification_report(y_test, rf_preds, target_names=['Non-Diabetic', 'Diabetic']))


# ------------------------------------------------------------------------------
# SECTION 7: Plot Feature Importance for Random Forest
# ------------------------------------------------------------------------------
print("\n=== SECTION 7: Plotting Feature Importances ===")

# Retrieve importances from the Random Forest model
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = X.columns[indices]

# Plot horizontal bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=sorted_features, palette='viridis')
plt.title('Random Forest Feature Importances', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Relative Importance Value', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
importance_filename = 'feature_importance.png'
plt.savefig(importance_filename, dpi=300)
plt.close()
print(f"Saved feature importances plot to '{importance_filename}'")


# ------------------------------------------------------------------------------
# SECTION 8: Save Best Model
# ------------------------------------------------------------------------------
print("\n=== SECTION 8: Saving the Best Model ===")

# Choose best model based on higher test set accuracy
if rf_accuracy >= dt_accuracy:
    best_model = rf_model
    best_model_name = "Random Forest Classifier"
    best_accuracy = rf_accuracy
else:
    best_model = dt_model
    best_model_name = "Decision Tree Classifier"
    best_accuracy = dt_accuracy

print(f"Best model based on test accuracy: {best_model_name} ({best_accuracy * 100:.2f}%)")

# Save model file
model_path = 'model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

print(f"Successfully saved the best model to '{model_path}'")
print("\nMachine Learning Pipeline execution complete!")
