import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # ← ADD THIS LINE
import matplotlib.pyplot as plt
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("table_fan_dataset_v3.csv")
print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

# =====================================================
# FEATURE ENGINEERING
# =====================================================

df["temp_current_ratio"] = (
    df["temperature_C"] /
    (df["current_A"] + 0.0001)
)

df["noise_vibration"] = (
    df["noise_dB"] *
    df["vibration_g"]
)

df["stress_index"] = (
      0.4 * df["temperature_C"]
    + 0.3 * df["noise_dB"]
    + 0.3 * (df["vibration_g"] * 50)
)

# =====================================================
# FEATURES
# =====================================================

features = [

    "temperature_C",
    "noise_dB",
    "vibration_g",
    "rpm",
    "current_A",
    "voltage_V",
    "power_W",
    "humidity_pct",

    "temp_current_ratio",
    "noise_vibration",
    "stress_index"

]

X = df[features]

# =====================================================
# TARGET
# =====================================================

y = df["failure_mode"]

# =====================================================
# LABEL ENCODING
# =====================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)
joblib.dump(encoder, "label_encoder_xgboost.pkl")

print("\nFAILURE MODE MAPPING")

for i, name in enumerate(encoder.classes_):
    print(i, "=", name)

# =====================================================
# SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTRAIN SIZE =", len(X_train))
print("TEST SIZE  =", len(X_test))

# =====================================================
# GPU XGBOOST MODEL
# =====================================================

model = XGBClassifier(

    objective="multi:softmax",

    num_class=len(encoder.classes_),

    n_estimators=500,

    max_depth=8,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    tree_method="hist",

    device="cpu",

    random_state=42
)

# =====================================================
# TRAIN
# =====================================================

print("\nTRAINING ON INTEL UHD...")

start = time.time()

model.fit(
    X_train,
    y_train
)
joblib.dump(model, "fan_model_xgboost.pkl")
end = time.time()

training_time = end - start

print("\nMODEL TRAINED")

print(
    "TRAINING TIME =",
    round(training_time,2),
    "seconds"
)

# =====================================================
# PREDICT
# =====================================================

predictions = model.predict(X_test)

# =====================================================
# ACCURACY
# =====================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nACCURACY =", round(accuracy,4))

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nCONFUSION MATRIX")

print(cm)

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        predictions
    )
)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = pd.DataFrame({

    "Feature": features,

    "Importance":
    model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")

print(importance)

# =====================================================
# IMPORTANCE GRAPH
# =====================================================

plt.figure(figsize=(10,5))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xticks(rotation=45)

plt.title(
    "XGBoost Feature Importance"
)

plt.tight_layout()

plt.show()

# =====================================================
# CUSTOM FAN TEST
# =====================================================

sample = pd.DataFrame([{

    "temperature_C":85,

    "noise_dB":65,

    "vibration_g":1.2,

    "rpm":310,

    "current_A":0.55,

    "voltage_V":230,

    "power_W":85,

    "humidity_pct":60

}])

sample["temp_current_ratio"] = (
    sample["temperature_C"] /
    (sample["current_A"] + 0.0001)
)

sample["noise_vibration"] = (
    sample["noise_dB"] *
    sample["vibration_g"]
)

sample["stress_index"] = (
      0.4 * sample["temperature_C"]
    + 0.3 * sample["noise_dB"]
    + 0.3 * (sample["vibration_g"] * 50)
)

prediction = model.predict(sample)

failure_mode = encoder.inverse_transform(
    prediction.astype(int)
)

print("\nCUSTOM FAN TEST")

print(
    "PREDICTED FAILURE MODE =",
    failure_mode[0]
)

# =====================================================
# RECOMMENDATION
# =====================================================

recommendations = {

    "Healthy":
    "Fan operating normally.",

    "Bearing_Wear":
    "Inspect and lubricate bearings.",

    "Blade_Imbalance":
    "Check blade balancing.",

    "Electrical_Overheat":
    "Inspect motor winding and cooling.",

    "Loose_Casing":
    "Tighten housing and screws.",

    "Mechanical_Looseness":
    "Inspect shaft and mounting assembly.",

    "Motor_Stress":
    "Check electrical load and current draw.",

    "Severe_Failure":
    "Immediate shutdown and maintenance required."
}

print("\nRECOMMENDED ACTION")

print(
    recommendations.get(
        failure_mode[0],
        "No recommendation available."
    )
)