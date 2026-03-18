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

# limpar nomes das colunas
df.columns = df.columns.str.strip().str.upper()

# =========================
# IDENTIFICAR COLUNAS AUTOMATICAMENTE
# =========================
def encontrar_coluna(nome):
    for col in df.columns:
        if nome in col:
            return col
    return None

col_ida = encontrar_coluna("IDA")
col_ieg = encontrar_coluna("IEG")
col_ips = encontrar_coluna("IPS")
col_ipp = encontrar_coluna("IPP")
col_ipv = encontrar_coluna("IPV")
col_inde = encontrar_coluna("INDE")

colunas_reais = [col_ida, col_ieg, col_ips, col_ipp, col_ipv, col_inde]

# validar
if None in colunas_reais:
    st.error("Erro: Não foi possível identificar todas as colunas automaticamente.")
    st.write("Colunas disponíveis no dataset:")
    st.write(df.columns)
    st.stop()

# criar dataframe padronizado
df_modelo = df[colunas_reais].copy()
df_modelo.columns = ["IDA", "IEG", "IPS", "IPP", "IPV", "INDE"]

# =========================
# TRATAR DADOS
# =========================
df_modelo = df_modelo.dropna()

# =========================
# TARGET (RISCO)
# =========================
df_modelo["RISCO"] = (df_modelo["INDE"] < 5).astype(int)

X = df_modelo[["IDA", "IEG", "IPS", "IPP", "IPV", "INDE"]]
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
    importancias = pd.Series(modelo.feature_importances_, index=X.columns)

    fig = px.bar(
        x=importancias.index,
        y=importancias.values,
        title="Importância dos Indicadores"
    )

    st.plotly_chart(fig)

# =========================
# DASHBOARD SIMPLES
# =========================
st.subheader("📈 Visão geral dos dados")

col1, col2 = st.columns(2)

col1.metric("Total de alunos", len(df_modelo))
col2.metric("Alunos em risco", int(df_modelo["RISCO"].sum()))
