import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="centered")


# ---------------------------------------------------------------
# Load the trained model, scaler and feature columns (cached)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("random_forest_model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_columns = joblib.load("feature_columns.joblib")
    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Model file not found: {e.filename}. Keep random_forest_model.joblib, "
        "scaler.joblib and feature_columns.joblib in the same folder as this app."
    )
    st.stop()


# ---------------------------------------------------------------
# Title
# ---------------------------------------------------------------
st.title("Fraud Detection Prediction App")
st.write(
    "Enter the transaction details below to predict whether a transaction "
    "is fraudulent or not."
)

# ---------------------------------------------------------------
# Inputs
# These are exactly the raw features the model was trained on:
# type, amount, oldbalanceOrg, oldbalanceDest
# (newbalanceOrig / newbalanceDest were deliberately dropped in the
#  notebook, so they are NOT asked here.)
# ---------------------------------------------------------------
st.header("Enter Transaction Information")

transaction_type = st.selectbox(
    "Transaction Type",
    ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
    help="Type of the transaction. In this dataset, fraud occurs mainly in "
         "TRANSFER and CASH_OUT.",
)

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=1000.0,
    step=100.0,
    format="%.2f",
)

oldbalanceOrg = st.number_input(
    "Sender's Old Balance (before the transaction)",
    min_value=0.0,
    value=5000.0,
    step=100.0,
    format="%.2f",
)

oldbalanceDest = st.number_input(
    "Receiver's Old Balance (before the transaction)",
    min_value=0.0,
    value=2000.0,
    step=100.0,
    format="%.2f",
)

# Only shown if the model was trained with this column
is_flagged = 0
if "isFlaggedFraud" in feature_columns:
    flagged_choice = st.selectbox(
        "Flagged by system rule (isFlaggedFraud)?",
        ["No", "Yes"],
        help="Whether the bank's rule-based system already flagged this transaction.",
    )
    is_flagged = 1 if flagged_choice == "Yes" else 0

# Simple sanity warning (does not block the prediction)
if transaction_type != "CASH_IN" and amount > oldbalanceOrg:
    st.warning(
        "The amount is larger than the sender's old balance. "
        "This is unusual and can itself be a fraud signal."
    )


# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
if st.button("Predict Fraud"):

    # 1. Raw input
    input_data = pd.DataFrame({
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "oldbalanceDest": [oldbalanceDest],
        "type": [transaction_type],
    })

    if "isFlaggedFraud" in feature_columns:
        input_data["isFlaggedFraud"] = is_flagged

    # 2. amount_range feature (same bins/labels as the notebook)
    bins = [0, 1000, 10000, 100000, 1000000, float("inf")]
    labels = ["0–1K", "1K–10K", "10K–100K", "100K–1M", "1M+"]

    input_data["amount_range"] = pd.cut(
        input_data["amount"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    # 3. transaction_amount_category (same bins/labels as the notebook)
    input_data["transaction_amount_category"] = pd.cut(
        input_data["amount"],
        bins=[0, 100, 1000, 10000, np.inf],
        labels=["low", "medium", "high", "very high"],
    )

    # 4. Dummy variables
    input_data = pd.get_dummies(
        input_data,
        columns=["type", "amount_range", "transaction_amount_category"],
        dtype=int,
    )

    # 5. Match training feature columns and order
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)

    # 6. Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    fraud_probability = None
    if hasattr(model, "predict_proba"):
        fraud_probability = float(model.predict_proba(input_scaled)[0][1])

    # 7. Show result
    st.header("Fraud Detection Result")

    if prediction == 1:
        st.error("YES! This Transaction is FRAUDULENT.")
    else:
        st.success("NO! This Transaction is NOT FRAUDULENT.")

    if fraud_probability is not None:
        st.metric("Fraud Probability", f"{fraud_probability * 100:.2f}%")
        st.progress(min(max(fraud_probability, 0.0), 1.0))

        if fraud_probability >= 0.7:
            st.write("Risk level: **High** – recommend blocking / manual review.")
        elif fraud_probability >= 0.3:
            st.write("Risk level: **Medium** – recommend manual review.")
        else:
            st.write("Risk level: **Low**.")

    with st.expander("Show details used by the model"):
        st.write("Input summary")
        st.write(
            {
                "type": transaction_type,
                "amount": amount,
                "oldbalanceOrg": oldbalanceOrg,
                "oldbalanceDest": oldbalanceDest,
            }
        )
        st.write("Encoded features (before scaling), non-zero only")
        st.dataframe(input_data.loc[:, (input_data != 0).any(axis=0)])

st.caption(
    "This prediction is made by a machine-learning model and should support, "
    "not replace, human review."
)