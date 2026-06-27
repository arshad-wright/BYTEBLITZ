# ==========================================
# IMPORT LIBRARIES
# ==========================================
import joblib
import matplotlib
matplotlib.use('Agg')   # ← ADD THIS LINE
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("table_fan_dataset_v3.csv")

print("\nFIRST 5 ROWS")
print(df.head())

# ==========================================
# DATASET INFORMATION
# ==========================================

print("\nDATASET INFO")
print(df.info())

print("\nHEALTH STATUS COUNTS")
print(df['health_status'].value_counts())

print("\nFAILURE MODE COUNTS")
print(df['failure_mode'].value_counts())

# ==========================================
# SCATTER PLOT
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(
    df['temperature_C'],
    df['vibration_g'],
    c=df['health_status'].astype('category').cat.codes
)

plt.xlabel("Temperature (C)")
plt.ylabel("Vibration (g)")
plt.title("Temperature vs Vibration")

plt.savefig("scatter_plot.png")
plt.close()

# ==========================================
# FEATURE ENGINEERING
# ==========================================

df['temp_current_ratio'] = (
    df['temperature_C'] /
    df['current_A']
)

df['noise_vibration'] = (
    df['noise_dB'] *
    df['vibration_g']
)

df['stress_index'] = (
      0.4 * df['temperature_C']
    + 0.3 * df['noise_dB']
    + 0.3 * (df['vibration_g'] * 50)
)

# ==========================================
# FEATURES
# ==========================================

X = df[
[
    'temperature_C',
    'noise_dB',
    'vibration_g',
    'rpm',
    'current_A',
    'voltage_V',
    'power_W',
    'humidity_pct',
    'temp_current_ratio',
    'noise_vibration',
    'stress_index'
]
]

# ==========================================
# TARGET = FAILURE MODE
# ==========================================

y = df['failure_mode']

# ==========================================
# ENCODE LABELS
# ==========================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)
joblib.dump(encoder, "label_encoder_random_forest.pkl")
print("\nFAILURE MODE MAPPING")

for i, cls in enumerate(encoder.classes_):
    print(i, "=", cls)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTRAIN SIZE:", len(X_train))
print("TEST SIZE :", len(X_test))

# ==========================================
# RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

# ==========================================
# TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)
joblib.dump(model, "fan_model_random_forest.pkl")
print("\nMODEL TRAINED SUCCESSFULLY")

# ==========================================
# PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nACCURACY =", accuracy)

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nCONFUSION MATRIX")
print(cm)
plt.savefig("feature_importance.png")
plt.savefig("confusion_matrix.png")
plt.close()
# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        predictions
    )
)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame(
{
    'Feature': X.columns,
    'Importance': model.feature_importances_
}
)

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance)

# ==========================================
# FEATURE IMPORTANCE GRAPH
# ==========================================

plt.figure(figsize=(10,5))

plt.bar(
    importance['Feature'],
    importance['Importance']
)

plt.xticks(rotation=45)

plt.title("Feature Importance")

plt.savefig("feature_importance.png")
plt.close()

# ==========================================
# CUSTOM FAN TEST
# ==========================================

sample = pd.DataFrame(
[
{
    'temperature_C':85,
    'noise_dB':65,
    'vibration_g':1.2,
    'rpm':310,
    'current_A':0.55,
    'voltage_V':230,
    'power_W':85,
    'humidity_pct':60
}
]
)

# engineered features

sample['temp_current_ratio'] = (
    sample['temperature_C'] /
    sample['current_A']
)

sample['noise_vibration'] = (
    sample['noise_dB'] *
    sample['vibration_g']
)

sample['stress_index'] = (
      0.4 * sample['temperature_C']
    + 0.3 * sample['noise_dB']
    + 0.3 * (sample['vibration_g'] * 50)
)

# prediction

prediction = model.predict(sample)

failure_mode = encoder.inverse_transform(prediction)

print("\nCUSTOM FAN TEST")
print("Predicted Failure Mode =", failure_mode[0])

# ==========================================
# RECOMMENDATIONS
# ==========================================

recommendations = {

    "Healthy":
    "Fan operating normally.",

    "Bearing_Wear":
    "Inspect and lubricate bearings.",

    "Blade_Imbalance":
    "Check blade balance and alignment.",

    "Electrical_Overheat":
    "Inspect motor winding and cooling.",

    "Loose_Casing":
    "Tighten fan housing and screws.",

    "Mechanical_Looseness":
    "Inspect shaft and mounting assembly.",

    "Motor_Stress":
    "Check electrical load and current.",

    "Severe_Failure":
    "Immediate shutdown and maintenance required."
}

print(
    "\nRECOMMENDED ACTION:"
)

print(
    recommendations.get(
        failure_mode[0],
        "No recommendation available."
    )
)