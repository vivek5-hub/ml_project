# Student Grant Recommendation Prediction Pipeline with Pickle + SMOTE Balancing + Label Saving

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# 1. Load dataset
data = pd.read_csv('student_records.csv')
print("First 5 rows:")
print(data.head())

# 2. Basic data info
print("\nDataset Info:")
print(data.info())
print("\nMissing values before cleaning:")
print(data.isnull().sum())

# 3. Map OverallGrade letters to numeric scores
grade_mapping = {'A': 90, 'B': 80, 'C': 70, 'D': 60, 'E': 50, 'F': 40}
data['OverallGrade'] = data['OverallGrade'].map(grade_mapping)

# 4. Handle missing numeric data
num_imputer = SimpleImputer(strategy='mean')
data[['OverallGrade', 'ResearchScore', 'ProjectScore']] = num_imputer.fit_transform(
    data[['OverallGrade', 'ResearchScore', 'ProjectScore']]
)

# 5. Handle missing categorical data
cat_imputer = SimpleImputer(strategy='most_frequent')
data[['Obedient', 'Recommend']] = cat_imputer.fit_transform(
    data[['Obedient', 'Recommend']]
)

# 6. Encode categorical variables
label_enc = LabelEncoder()
data['Obedient'] = label_enc.fit_transform(data['Obedient'])
data['Recommend'] = label_enc.fit_transform(data['Recommend'])

# 7. Drop 'Name' column
data = data.drop(columns=['Name'])

# 8. Split features and target
X = data.drop(columns=['Recommend'])
y = data['Recommend']

# 9. Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("\nBalanced dataset class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# 10. Train-test split (after balancing)
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Verify balance in train and test
print("\nTrain set class distribution:")
print(pd.Series(y_train).value_counts())
print("\nTest set class distribution:")
print(pd.Series(y_test).value_counts())

# 11. Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 12. Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 13. Save model, scaler, and label encoder using pickle
with open("nb_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label.pkl", "wb") as f:
    pickle.dump(label_enc, f)

print("\nModel, scaler, and label encoder saved successfully!")

# 14. Predictions and evaluation
y_pred = model.predict(X_test_scaled)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 15. Load model, scaler, and label encoder for prediction
with open("nb_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    loaded_scaler = pickle.load(f)
with open("label.pkl", "rb") as f:
    loaded_label = pickle.load(f)

# 16. Predict for a new student
new_student = pd.DataFrame({
    'OverallGrade': [85],   # numeric after mapping
    'Obedient': [1],        # 1 = Yes, 0 = No
    'ResearchScore': [90],
    'ProjectScore': [88]
})
new_student_scaled = loaded_scaler.transform(new_student)
recommendation = loaded_model.predict(new_student_scaled)
print("\nGrant Recommendation (1=Yes, 0=No):", recommendation[0])
print(data['Recommend'].value_counts())
