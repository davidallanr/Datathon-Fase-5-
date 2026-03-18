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

# limpar nomes das colunas (muito importante)
df.columns = df.columns.str.strip()

# debug (pode apagar depois)
# st.write(df.columns)

# =========================
# DEFINIR COLUNAS (AJUSTADO)
# =========================
colunas = ["IDA", "IEG", "IPS", "IPP", "IPV", "INDE"]

# garantir que existem
df = df[colunas]

# =========================
# CRIAR TARGET (RISCO)
# =========================
df["RISCO"] = (df["INDE"] < 5).astype(int)

X = df[colunas]
y = df["RISCO"]

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
