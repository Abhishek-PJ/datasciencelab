import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the dataset
df = pd.read_csv('breastcancer2 lab.csv')
df.head(20)

# Step 3: Drop the 'id' column as it's not useful
df.drop(columns=['id'], inplace=True)

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

df.fillna(df.drop(columns=['diagnosis']).mean(), inplace=True)

# Step 5: Feature Selection (texture_mean, radius_mean) and Target Variable (diagnosis)
X = df[['texture_mean', 'radius_mean']]
y = df['diagnosis']

# Step 6: Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 8: Train KNN Model with different k-values
k_values = range(1, 21)  # Testing k values from 1 to 20
accuracy_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

# Step 9: Plot Accuracy vs. k-values
plt.figure(figsize=(10, 5))
plt.plot(k_values, accuracy_scores, marker='o', linestyle='dashed', color='b')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy for Different k Values')
plt.show()

# Step 10: Select the best k (max accuracy)
best_k = k_values[np.argmax(accuracy_scores)]
print(f"Best k value: {best_k}")

# Step 11: Train the final KNN model with best k value
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = knn_best.predict(X_train_scaled)
y_test_pred = knn_best.predict(X_test_scaled)

# Step 12: Model Performance Metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Step 12: Model Performance Metrics
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# Step 13: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Step 14: Plot Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

sns.heatmap(df.select_dtypes(exclude=['object']).corr())
plt.show()
