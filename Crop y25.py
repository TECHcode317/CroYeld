
import pandas as pd
import numpy as np
import joblib  
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


uploaded = files.upload()


data = pd.read_csv('data_season.csv')


data = data.drop(columns=['Year', 'Location', 'Crops', 'price', 'yeilds'])


label_encoders = {}
for col in ['Soil type', 'Irrigation', 'Season']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le


X = data[['Rainfall', 'Temperature', 'soil_moisture', 'Soil type', 'Humidity', 'Season', 'Area']]
y = data['Irrigation']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


joblib.dump(model, 'smart_irrigation_model.pkl')


train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

plt.figure(figsize=(6, 4))
plt.bar(['Training Accuracy', 'Testing Accuracy'], [train_score, test_score], color=['blue', 'green'])
plt.xlabel("Dataset")
plt.ylabel("Accuracy Score")
plt.title("Training vs Testing Accuracy")
plt.ylim(0, 1)
plt.show()


feature_importances = model.feature_importances_
plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importances, y=X.columns)
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Feature Importance in Irrigation Prediction")
plt.show()


conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=label_encoders['Irrigation'].classes_, yticklabels=label_encoders['Irrigation'].classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


plt.figure(figsize=(6, 4))
plt.hist([y_test, y_pred], bins=5, color=['orange', 'blue'], label=['Actual', 'Predicted'], alpha=0.7)
plt.xlabel("Irrigation Labels")
plt.ylabel("Frequency")
plt.title("Predicted vs Actual Irrigation Distribution")
plt.legend()
plt.show()

print(f'Model Accuracy: {accuracy:.2f}')
print("Classification Report:\n", classification_report(y_test, y_pred))
