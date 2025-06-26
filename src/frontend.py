import streamlit as st
import requests

# Função para enviar dados para a API e receber a previsão
def get_previsao(customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges):
    url = 'http://127.0.0.1:8000/prever_churn'  # Endereço da API FastAPI
    data = {
        "customerID": customerID,
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    response = requests.post(url, json=data)
    print("Status Code:", response.status_code)
    print("Response Body:", response.text)  # Print or log the raw response
    try:
        return response.json()  # Attempt to parse JSON
    except ValueError:
        # Handle the case where parsing JSON fails
        print("Failed to decode JSON from response:")
        print(response.text)
        return None


# Interface do usuário no Streamlit
st.title('Previsão de Churn de Clientes')


# Entrada de dados pelo usuário
customerID = st.text_input("CustomerID", "0000-AAAAA")
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.number_input("SeniorCitizen", min_value=0, max_value=1, value=0)
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure", min_value=1, max_value=100, value=1)
PhoneService = st.selectbox("PhoneService", ["Yes", "No"])
MultipleLines = st.selectbox("MultipleLines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("InternetService", ["Fiber optic", "DSL", "No"])
OnlineSecurity = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"])
PaymentMethod = st.selectbox("PaymentMethod",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
MonthlyCharges = st.number_input("MonthlyCharges", min_value=10.0, max_value=200.0, value=10.0)
TotalCharges = st.number_input("TotalCharges", min_value=10.0, max_value=11000.0, value=10.0)

# Botão para fazer a previsão
if st.button('Situação'):
    resposta = get_previsao(customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges)
    if resposta is not None and 'churn' in resposta:
        st.success(f'A situação do cliente é: {"Churn" if resposta["churn"] else "No Churn"}')
    else:
        st.error('Erro ao obter previsão.')
