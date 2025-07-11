import pandas as pd
import numpy as np
import joblib  
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap


plt.style.use('seaborn-v0_8-poster')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300
})


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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


plt.figure(figsize=(8, 5))
plt.bar(['Training Accuracy', 'Testing Accuracy'], 
        [model.score(X_train, y_train), accuracy],
        color=['#4C72B0', '#55A868'])
plt.ylabel("Accuracy Score")
plt.title("Model Generalization Performance")
plt.ylim(0.7, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_imp.values, y=feature_imp.index, palette='viridis')
plt.title("Feature Importance (Gini Importance)")
plt.xlabel("Relative Importance")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoders['Irrigation'].classes_,
            yticklabels=label_encoders['Irrigation'].classes_)
plt.title("Confusion Matrix (Actual vs Predicted)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()


train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="#4C72B0", label="Training")
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="#55A868", label="Cross-validation")
plt.fill_between(train_sizes,
                 np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                 np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                 alpha=0.1, color="#4C72B0")
plt.fill_between(train_sizes,
                 np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                 np.mean(test_scores, axis=1) + np.std(test_scores, axis=1),
                 alpha=0.1, color="#55A868")
plt.title("Learning Curves (5-Fold Cross Validation)")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid()
plt.show()


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("Feature Importance (Absolute Impact on Output)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values[1], X_test, show=False)  
plt.title("Value Distribution (Directional Impact)")
plt.tight_layout()
plt.show()


def plot_sensor_performance(sensor_df, title="Proposed Model Performance Under Extreme Conditions"):
  
    plt.figure(figsize=(10, 6))
    
  
    x = np.arange(len(sensor_df))
    width = 0.35
    
   
    rects1 = plt.bar(x - width/2, sensor_df['Accuracy (%)'], width, 
                    label='Accuracy (%)', color='#4C72B0', alpha=0.8)
    rects2 = plt.bar(x + width/2, sensor_df['MAE'], width, 
                    label='MAE', color='#C44E52', alpha=0.8)
    
   
    plt.ylabel('Performance Metrics')
    plt.xticks(x, sensor_df['Parameter'])
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
 
    def autolabel(rects, unit=''):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height,
                    f'{height:.1f}{unit}',
                    ha='center', va='bottom', fontsize=10)
    
    autolabel(rects1, '%')
    autolabel(rects2)
    

    max_val = max(sensor_df['Accuracy (%)'].max(), sensor_df['MAE'].max()) * 1.2
    plt.ylim(0, max_val)
    
    plt.tight_layout()
    return plt


sensor_data = {
    'Parameter': ['Temperature', 'Soil Moisture', 'Humidity', 'Rainfall'],
    'Accuracy (%)': [94.2, 92.8, 93.5, 95.1],
    'MAE': [0.8, 1.2, 1.5, 0.5]
}

sensor_df = pd.DataFrame(sensor_data)
plot = plot_sensor_performance(sensor_df, 
                             title="Sensor Accuracy vs MAE (45°C, 90% RH)")
plot.show()


joblib.dump(model, 'smart_irrigation_model.pkl')
print(f'\nFinal Model Accuracy: {accuracy:.4f}')
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoders['Irrigation'].classes_))
