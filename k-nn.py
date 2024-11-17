import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
loan_data = pd.read_csv('dataset/loan-10k.lrn.csv')

# Define target (y) and features (X)
y = loan_data['grade']
X = loan_data.drop(columns=['grade'])

# Preprocess categorical columns
cat_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Fill missing values if any
X = X.fillna(X.mean())

# Scale numeric features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode the target variable
label_enc = LabelEncoder()
y = label_enc.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and tune the K-NN model
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

grid_search_knn = GridSearchCV(estimator=KNeighborsClassifier(),
                               param_grid=param_grid,
                               cv=5,
                               scoring='accuracy',
                               verbose=1)

grid_search_knn.fit(X_train, y_train)

# Get the best model
best_knn = grid_search_knn.best_estimator_
print("Best Parameters for K-NN:", grid_search_knn.best_params_)
print("Best Accuracy from GridSearchCV:", grid_search_knn.best_score_)

# Train the best K-NN model
best_knn.fit(X_train, y_train)

# Predict and evaluate
y_pred_knn = best_knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_enc.classes_, yticklabels=label_enc.classes_)
plt.title("Confusion Matrix for K-NN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
