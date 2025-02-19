import pandas as pd
import streamlit as st
import pickle


model = pickle.load(open("models/lgb_clf.pkl", "rb"))


def preprocess(
    customerID,
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    MonthlyCharges,
    TotalCharges,
):
    df = pd.DataFrame(
        {
            "customerID": [customerID],
            "gender": [gender],
            "SeniorCitizen": [SeniorCitizen],
            "Partner": [Partner],
            "Dependents": [Dependents],
            "tenure": [tenure],
            "PhoneService": [PhoneService],
            "MultipleLines": [MultipleLines],
            "InternetService": [InternetService],
            "OnlineSecurity": [OnlineSecurity],
            "OnlineBackup": [OnlineBackup],
            "DeviceProtection": [DeviceProtection],
            "TechSupport": [TechSupport],
            "StreamingTV": [StreamingTV],
            "StreamingMovies": [StreamingMovies],
            "Contract": [Contract],
            "PaperlessBilling": [PaperlessBilling],
            "PaymentMethod": [PaymentMethod],
            "MonthlyCharges": [MonthlyCharges],
            "TotalCharges": [TotalCharges],
        }
    )

    df = df.drop(["gender", "customerID"], axis=1)
    df["SeniorCitizen"] = df["SeniorCitizen"].apply(lambda x: 1 if x == "Yes" else 0)
    df = pd.get_dummies(df, drop_first=True, dtype=float)
    print(df)

    return df


def predict(data):
    prediction = model.predict(data)
    return prediction


# Interface do Strealit
st.title("Churn Prediction")

# Criação de campos para entrada de dados
customerID = st.text_input("CustomerID", "0000-AAAAA")
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("SeniorCitizen", ["Yes", "No"])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure", min_value=1, max_value=100, value=1)
PhoneService = st.selectbox("PhoneService", ["Yes", "No"])
MultipleLines = st.selectbox("MultipleLines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("InternetService", ["Fiber optic", "DSL", "No"])
OnlineSecurity = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox(
    "DeviceProtection", ["Yes", "No", "No internet service"]
)
TechSupport = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "PaymentMethod",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)
MonthlyCharges = st.number_input(
    "MonthlyCharges", min_value=10.0, max_value=200.0, value=10.0
)
TotalCharges = st.number_input(
    "TotalCharges", min_value=10.0, max_value=11000.0, value=10.0
)

if st.button("Situação"):
    data = preprocess(
        customerID,
        gender,
        SeniorCitizen,
        Partner,
        Dependents,
        tenure,
        PhoneService,
        MultipleLines,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        StreamingTV,
        StreamingMovies,
        Contract,
        PaperlessBilling,
        PaymentMethod,
        MonthlyCharges,
        TotalCharges,
    )
    prediction = predict(data)

    st.write("Situação:", "Churn" if prediction[0] == 1 else "No Churn")
