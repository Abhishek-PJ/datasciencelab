import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
df = pd.read_csv("student_performance.csv")  # Replace with actual dataset file
print("First 5 rows of the dataset:")
print(df.head())  # Display first few rows of the dataset

# Step 2: Prepare Features (X) and Target Variable (y)
# Assuming 'Performance' is the target column, replace it if needed
X = df.drop(columns=['USN','STUDENT NAME'])  # Features (independent variables)
y = df['Result']  # Target variable (dependent variable)

# Convert any remaining categorical features to numeric
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col])  # Encode categorical features

# Step 3: Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Define the hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],      # Number of trees in the forest
    'max_depth': [10, 20, None],          # Maximum depth of trees
    'min_samples_split': [2, 5, 10],      # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],        # Minimum samples required at a leaf node
    'bootstrap': [True, False]            # Whether to use bootstrap samples
}

# Step 5: Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)


# Step 6: Perform Grid Search to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Step 7: Extract the best parameters from GridSearchCV
best_params = grid_search.best_params_
print("Best Parameters Found:", best_params)

# Step 8: Train the model using the best parameters
rf_best = RandomForestClassifier(**best_params, random_state=42)
rf_best.fit(X_train, y_train)

# Step 9: Predict the test set and train set results

# Compute Testing Accuracy
y_test_pred = rf_best.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)


# Compute Training Accuracy
y_train_pred = rf_best.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)


# Print Training and Testing Accuracy
print("\nTraining Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Step 10: Evaluate the Model
# Compute Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
accuracy = accuracy_score(y_test, y_test_pred)

# Display evaluation metrics
print("\nConfusion Matrix:\n", conf_matrix)
print("\nAccuracy Score:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# Step 11: Visualizing Results

# Plot Confusion Matrix as a Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.show()

# Feature Importance Analysis
feature_importance = pd.Series(rf_best.feature_importances_, index=X.columns).sort_values(ascending=False)

# Plot Feature Importance
plt.figure(figsize=(8, 5))
feature_importance.plot(kind='bar', color='teal')
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Feature Importance in Tuned Random Forest Model")
plt.show()
    
