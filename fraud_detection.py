import joblib
import streamlit as st

from fraud_model import known_types, predict_one

MODEL_PATH = "fraud_detection_pipeline.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("Fraud Detection Prediction App")

try:
    model = load_model()
except FileNotFoundError:
    st.error(f"`{MODEL_PATH}` not found. Run `python train_model.py` (needs the Kaggle CSV) first.")
    st.stop()
except Exception as exc:  # typically a scikit-learn version mismatch when unpickling
    st.error(
        "Could not load the saved model. It was trained with scikit-learn 1.6.1; "
        "install the versions in requirements.txt or retrain with `python train_model.py`."
    )
    st.caption(f"{type(exc).__name__}: {exc}")
    st.stop()

st.markdown("Please enter the transaction details and use the predict button")

st.divider()

transaction_type = st.selectbox("Transaction Type", known_types(model))
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

if st.button("Predict"):
    prediction = predict_one(model, transaction_type, amount, oldbalanceOrg, newbalanceOrig,
                             oldbalanceDest, newbalanceDest)

    st.subheader(f"Prediction : '{prediction}'")

    if prediction == 1:
        st.error("This transaction may be fraudulent")
    else:
        st.success("This transaction looks like it is not a fraud")
    st.caption("Demo model (logistic regression on a simulated dataset); not for real fraud decisions.")
