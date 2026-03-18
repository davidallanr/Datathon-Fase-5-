import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Risco Educacional", layout="centered")

st.title("📊 Previsão de Risco Educacional")
st.write("Modelo baseado nos indicadores reais da base Passos Mágicos")

# =========================
# CARREGAR DADOS
# =========================
import os

arquivos = os.listdir()

st.write("Arquivos disponíveis:", arquivos)  # debug

arquivo = [f for f in arquivos if f.endswith(".xlsx")][0]

df = pd.read_excel(arquivo)

# limpar colunas
df.columns = df.columns.str.strip()

# =========================
# SELECIONAR COLUNAS CORRETAS (DO SEU DATASET)
# =========================
colunas = ["IDA", "IEG", "IPS", "IPV", "INDE"]

df_modelo = df[colunas].dropna()

# =========================
# CRIAR TARGET
# =========================
df_modelo["RISCO"] = (df_modelo["INDE"] < 5).astype(int)

X = df_modelo[["IDA", "IEG", "IPS", "IPV", "INDE"]]
y = df_modelo["RISCO"]

# =========================
# TREINAR MODELO
# =========================
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X, y)

# =========================
# INPUT USUÁRIO
# =========================
st.subheader("Insira os indicadores do aluno")

ida = st.slider("IDA - Desempenho Acadêmico", 0.0, 10.0, 5.0)
ieg = st.slider("IEG - Engajamento", 0.0, 10.0, 5.0)
ips = st.slider("IPS - Psicossocial", 0.0, 10.0, 5.0)
ipv = st.slider("IPV - Ponto de Virada", 0.0, 10.0, 5.0)
inde = st.slider("INDE - Índice Educacional", 0.0, 10.0, 5.0)

# =========================
# PREVISÃO
# =========================
if st.button("🔍 Prever risco"):
    entrada = [[ida, ieg, ips, ipv, inde]]

    pred = modelo.predict(entrada)[0]
    prob = modelo.predict_proba(entrada)[0][1]

    st.subheader("Resultado")

    st.write(f"Probabilidade de risco: **{prob:.2%}**")

    if pred == 1:
        st.error("⚠️ Aluno em risco educacional")
    else:
        st.success("✅ Aluno com desenvolvimento adequado")

    # =========================
    # IMPORTÂNCIA
    # =========================
    importancias = pd.Series(modelo.feature_importances_, index=X.columns)

    fig = px.bar(
        x=importancias.index,
        y=importancias.values,
        title="Importância dos Indicadores"
    )

    st.plotly_chart(fig)

# =========================
# DASHBOARD
# =========================
st.subheader("📈 Visão geral")

col1, col2 = st.columns(2)

col1.metric("Total de alunos", len(df_modelo))
col2.metric("Alunos em risco", int(df_modelo["RISCO"].sum()))
