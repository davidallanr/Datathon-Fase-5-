import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Risco Educacional", layout="centered")

st.title("📊 Previsão de Risco Educacional")
st.write("Modelo baseado nos indicadores educacionais dos alunos.")

# =========================
# CARREGAR DADOS
# =========================
df = pd.read_excel("dados_limpos base de dados.xlsx")

# limpar nomes
df.columns = df.columns.str.strip()

# deixar tudo maiúsculo (padronizar)
df.columns = df.columns.str.upper()

# =========================
# MAPEAR COLUNAS AUTOMATICAMENTE
# =========================
colunas_map = {
    "IDA": [col for col in df.columns if "IDA" in col],
    "IEG": [col for col in df.columns if "IEG" in col],
    "IPS": [col for col in df.columns if "IPS" in col],
    "IPP": [col for col in df.columns if "IPP" in col],
    "IPV": [col for col in df.columns if "IPV" in col],
    "INDE": [col for col in df.columns if "INDE" in col],
}

# pegar a primeira correspondência de cada
colunas = [colunas_map[k][0] for k in colunas_map if colunas_map[k]]

df = df[colunas]
df.columns = ["IDA", "IEG", "IPS", "IPP", "IPV", "INDE"]

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
ips = st.slider("IPS - Aspectos Psicossociais", 0.0, 10.0, 5.0)
ipp = st.slider("IPP - Indicador Psicopedagógico", 0.0, 10.0, 5.0)
ipv = st.slider("IPV - Ponto de Virada", 0.0, 10.0, 5.0)
inde = st.slider("INDE - Índice Educacional", 0.0, 10.0, 5.0)

# =========================
# PREVISÃO
# =========================
if st.button("🔍 Prever risco educacional"):
    entrada = [[ida, ieg, ips, ipp, ipv, inde]]
    
    pred = modelo.predict(entrada)[0]
    prob = modelo.predict_proba(entrada)[0][1]

    st.subheader("Resultado da previsão")

    st.write(f"Probabilidade de risco: **{prob:.2%}**")

    if pred == 1:
        st.error("⚠️ Aluno em risco educacional")
    else:
        st.success("✅ Aluno com desenvolvimento adequado")

    # =========================
    # IMPORTÂNCIA DAS FEATURES
    # =========================
    importancias = pd.Series(modelo.feature_importances_, index=colunas)

    fig = px.bar(
        x=importancias.index,
        y=importancias.values,
        title="Indicadores que mais influenciam o risco"
    )

    st.plotly_chart(fig)
