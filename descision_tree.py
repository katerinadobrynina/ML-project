import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load dataset
loan_data = pd.read_csv('dataset/loan-10k.lrn.csv')

# Extract target variable ('grade') before one-hot encoding
y = loan_data['grade']

# Encode target variable
label_enc = LabelEncoder()
y = label_enc.fit_transform(y)

# Impute missing values for numerical columns
num_cols = loan_data.select_dtypes(include=np.number).columns
imputer = SimpleImputer(strategy='mean')
loan_data[num_cols] = imputer.fit_transform(loan_data[num_cols])

# Impute and encode categorical features (exclude 'grade')
cat_cols = loan_data.select_dtypes(include=['object']).columns
cat_cols = [col for col in cat_cols if col != 'grade']  # Exclude 'grade'
imputer_cat = SimpleImputer(strategy='most_frequent')
loan_data[cat_cols] = imputer_cat.fit_transform(loan_data[cat_cols])
loan_data = pd.get_dummies(loan_data, columns=cat_cols, drop_first=True)

# Features (X)
X = loan_data.drop(columns=['grade'])  # Drop the target column

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(40, 20))  # Increase the figure size
plot_tree(clf, feature_names=X.columns, class_names=label_enc.classes_, filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.savefig("decision_tree_high_res.png", dpi=300)  # Save as high-resolution image
from sklearn.tree import export_text

# Export decision tree as a readable text representation
tree_rules = export_text(clf, feature_names=list(X.columns))
print(tree_rules)

feature_importance = pd.Series(clf.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).head(10).plot(kind='bar', figsize=(10, 6))
plt.title("Top 10 Feature Importances")
plt.show()