import pandas as pd
from fastapi import FastAPI, responses
import joblib
from pydantic import BaseModel,  PositiveInt, PositiveFloat
import pickle
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # substitua 4 pelo número de núcleos físicos da sua máquina

app = FastAPI()

modelo = joblib.load("../models/lgb_clf.pkl")

def load_features(path="../models/features.pkl"):
    with open(path, "rb") as file:
        features = list(pickle.load(file))
    return features

features = load_features()

class EspecificacoesCliente(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: PositiveInt
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: PositiveFloat
    TotalCharges: PositiveFloat

class EspecificacoesClienteResponse(BaseModel):
    churn: int

def preprocess(input_data):
    df = pd.DataFrame(input_data, index=[0])
    df = df.drop(["gender", "customerID"], axis=1)
    df = pd.get_dummies(df)
    # Reindexa com as colunas esperadas (completa com 0 se alguma estiver faltando)
    df = df.reindex(columns=features, fill_value=0)
    return df

@app.get("/")
def read_root():
    return {"message": "API de Previsão de Churn de Clientes"}

@app.post("/prever_churn", response_model=EspecificacoesClienteResponse)
def prever_churn(especificacoes: EspecificacoesCliente):

    dados_entrada = especificacoes.model_dump()

    # Preprocessa os dados
    dados_prep = preprocess(dados_entrada)
    
    # Faz a previsão
    pred = modelo.predict(dados_prep)
    response = EspecificacoesClienteResponse(churn=int(pred))
    return response




#print(prever_churn('5575-GNVDE','Male',0,'No','No',34,'Yes','No',
#                   'DSL','Yes','No','Yes','No','No','No','One year',
#                   'No','Mailed check',56.95,1889.5))


'''
Para garantir que o pd.get_dummies sempre retorna todas as colunas esperadas (inclusive as que não aparecem no input),
 você deve reindexar o DataFrame resultante com a lista completa de colunas que o modelo espera, preenchendo com 0 onde faltar.

Você pode salvar a lista de colunas usadas no treinamento (por exemplo, em um arquivo features.pkl)
e carregá-la para usar na reindexação. Aqui está como ajustar sua função preprocess:

Observação:

Certifique-se de salvar a lista de colunas usadas no treinamento em features.pkl usando
joblib.dump(list(df.columns), "features.pkl") após o treinamento.

Assim, mesmo que uma categoria não esteja presente no input, a coluna correspondente será criada com valor 0,
garantindo compatibilidade com o modelo.

'''
