import streamlit as st

import pandas as pd
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Fan Predictive Maintenance",
    page_icon="🛠",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("fan_model_random_forest.pkl")
encoder = joblib.load("label_encoder_random_forest.pkl")
st.write(type(model))
st.write(model.feature_names_in_)
st.write(model.n_features_in_)
# =====================================================
# TITLE
# =====================================================

st.title("🛠 AI Based Fan Predictive Maintenance Dashboard")
mode = st.sidebar.selectbox(
    "Choose Prediction Mode",
    [
        "Manual Prediction",
        "CSV Prediction"
    ]
)

##############################################################
#
# MODE 1
#
##############################################################

if mode == "Manual Prediction":

    st.header("Sensor Inputs")

    col1,col2=st.columns(2)

    with col1:

        temperature=st.number_input(
            "Temperature",
            value=35.0
        )

        noise=st.number_input(
            "Noise",
            value=45.0
        )

        vibration=st.number_input(
            "Vibration",
            value=2.5
        )

        rpm=st.number_input(
            "RPM",
            value=1450.0
        )

    with col2:

        current=st.number_input(
            "Current",
            value=0.45
        )

        voltage=st.number_input(
            "Voltage",
            value=230.0
        )

        power=st.number_input(
            "Power",
            value=90.0
        )

        humidity=st.number_input(
            "Humidity",
            value=60.0
        )

    if st.button("Predict"):

        temp_current_ratio=temperature/(current+0.0001)

        noise_vibration=noise*vibration

        stress_index=(
            0.4*temperature+
            0.3*noise+
            0.3*(vibration*50)
        )

        sample=pd.DataFrame([{

            "temperature_C":temperature,
            "noise_dB":noise,
            "vibration_g":vibration,
            "rpm":rpm,
            "current_A":current,
            "voltage_V":voltage,
            "power_W":power,
            "humidity_pct":humidity,

            "temp_current_ratio":
            temp_current_ratio,

            "noise_vibration":
            noise_vibration,

            "stress_index":
            stress_index

        }])

        pred=model.predict(sample)

        result = encoder.inverse_transform([int(pred[0])])[0]

        probs=model.predict_proba(sample)

        prob_df=pd.DataFrame({

            "Failure Mode":
            encoder.classes_,

            "Probability":
            probs[0]*100

        })

        prob_df=prob_df.sort_values(
            by="Probability",
            ascending=False
        )

        st.success(
            "Predicted Failure : "+ result
        )

        st.dataframe(prob_df)

        st.bar_chart(
            prob_df.set_index(
                "Failure Mode"
            )
        )

##############################################################
#
# MODE 2
#
##############################################################

else:

    st.header("CSV Fan Analysis")

    uploaded = st.file_uploader(
        "Upload Fan CSV",
        type="csv"
    )

    if uploaded is not None:

        df = pd.read_csv(uploaded)

        st.subheader("CSV Preview")
        st.dataframe(df.head())

        required_columns = [

            "temperature_C",
            "noise_dB",
            "vibration_g",
            "rpm",
            "current_A",
            "voltage_V",
            "power_W",
            "humidity_pct"

        ]

        missing = []

        for c in required_columns:
            if c not in df.columns:
                missing.append(c)

        if len(missing) > 0:

            st.error("Missing Columns : " + str(missing))

        else:

            # -------------------------
            # Feature Engineering
            # -------------------------

            df["temp_current_ratio"] = (
                df["temperature_C"] /
                (df["current_A"] + 0.0001)
            )

            df["noise_vibration"] = (
                df["noise_dB"] *
                df["vibration_g"]
            )

            df["stress_index"] = (
                0.4 * df["temperature_C"] +
                0.3 * df["noise_dB"] +
                0.3 * (df["vibration_g"] * 50)
            )

            # -------------------------
            # Select ONLY model features
            # -------------------------

            X = df[list(model.feature_names_in_)].copy()

            # -------------------------
            # Prediction
            # -------------------------

            preds = model.predict(X)

            df["Prediction"] = encoder.inverse_transform(
                preds.astype(int)
            )

            # -------------------------
            # Probability (optional)
            # -------------------------

            if hasattr(model, "predict_proba"):

                probs = model.predict_proba(X)

                confidence = probs.max(axis=1) * 100

                df["Confidence (%)"] = confidence.round(2)

            # -------------------------
            # Results
            # -------------------------

            st.subheader("Prediction Results")

            st.dataframe(df)

            # -------------------------
            # Distribution
            # -------------------------

            counts = df["Prediction"].value_counts()

            st.subheader("Failure Distribution")

            st.bar_chart(counts)

            # -------------------------
            # Overall Condition
            # -------------------------

            majority = counts.idxmax()

            st.success(
                "Overall Fan Condition : " + majority
            )

            # -------------------------
            # Download
            # -------------------------

            csv = df.to_csv(index=False)

            st.download_button(

                "Download Results",

                csv,

                "Predicted_Fan.csv",

                "text/csv"

            )