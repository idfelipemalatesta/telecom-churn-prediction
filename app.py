import pandas as pd
import streamlit as st
import pickle
import warnings

warnings.filterwarnings("ignore")


def load_model(path="models/lgb_clf.pkl"):
    with open(path, "rb") as file:
        return pickle.load(file)


def load_features(path="models/features.pkl"):
    with open(path, "rb") as file:
        features = list(pickle.load(file))
    return features


model = load_model()
expected_features = load_features()


def preprocess(input_data):
    print("preprocess")
    # Cria o DataFrame a partir do dicionário de dados
    df = pd.DataFrame(input_data, index=[0])
    df = df.drop(["gender", "customerID"], axis=1)
    df["SeniorCitizen"] = df["SeniorCitizen"].apply(lambda x: 1 if x == "Yes" else 0)
    df = pd.get_dummies(df)
    # Reindexa com as colunas esperadas (completa com 0 se alguma estiver faltando)
    df = df.reindex(columns=expected_features, fill_value=0)

    # print(df.columns)
    return df


def get_user_input():
    return {
        "customerID": st.text_input("CustomerID", "0000-AAAAA"),
        "gender": st.selectbox("Gender", ["Male", "Female"]),
        "SeniorCitizen": st.selectbox("SeniorCitizen", ["Yes", "No"]),
        "Partner": st.selectbox("Partner", ["Yes", "No"]),
        "Dependents": st.selectbox("Dependents", ["Yes", "No"]),
        "tenure": st.number_input("Tenure", min_value=1, max_value=100, value=1),
        "PhoneService": st.selectbox("PhoneService", ["Yes", "No"]),
        "MultipleLines": st.selectbox(
            "MultipleLines", ["Yes", "No", "No phone service"]
        ),
        "InternetService": st.selectbox(
            "InternetService", ["Fiber optic", "DSL", "No"]
        ),
        "OnlineSecurity": st.selectbox(
            "OnlineSecurity", ["Yes", "No", "No internet service"]
        ),
        "OnlineBackup": st.selectbox(
            "OnlineBackup", ["Yes", "No", "No internet service"]
        ),
        "DeviceProtection": st.selectbox(
            "DeviceProtection", ["Yes", "No", "No internet service"]
        ),
        "TechSupport": st.selectbox(
            "TechSupport", ["Yes", "No", "No internet service"]
        ),
        "StreamingTV": st.selectbox(
            "StreamingTV", ["Yes", "No", "No internet service"]
        ),
        "StreamingMovies": st.selectbox(
            "StreamingMovies", ["Yes", "No", "No internet service"]
        ),
        "Contract": st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        ),
        "PaperlessBilling": st.selectbox("PaperlessBilling", ["Yes", "No"]),
        "PaymentMethod": st.selectbox(
            "PaymentMethod",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        ),
        "MonthlyCharges": st.number_input(
            "MonthlyCharges", min_value=10.0, max_value=200.0, value=10.0
        ),
        "TotalCharges": st.number_input(
            "TotalCharges", min_value=10.0, max_value=11000.0, value=10.0
        ),
    }


def predict(data):
    print("predict")
    prediction = model.predict(data)
    return prediction


# Interface do Streamlit
st.title("Churn Prediction")
user_data = get_user_input()

if st.button("Situação"):
    df = preprocess(user_data)
    prediction = predict(df)
    st.write("Situação:", "Churn" if prediction[0] == 1 else "No Churn")
