import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# carregar dados
df = pd.read_excel("dados_limpos base de dados.xlsx")

# Features (ajuste se necessário)
X = df[["IDA_2022", "IEG_2022", "IPS_2022", "IPP_2022", "IPV_2022", "INDE_2022"]]

# Criar variável alvo (exemplo)
y = (df["INDE_2022"] < 5).astype(int)

# Treinar modelo
modelo = RandomForestClassifier()
modelo.fit(X, y)

# Inputs do usuário
st.title("Previsão de Risco Educacional")

ida = st.slider("IDA", 0.0, 10.0, 5.0)
ieg = st.slider("IEG", 0.0, 10.0, 5.0)
ips = st.slider("IPS", 0.0, 10.0, 5.0)
ipp = st.slider("IPP", 0.0, 10.0, 5.0)
ipv = st.slider("IPV", 0.0, 10.0, 5.0)
inde = st.slider("INDE", 0.0, 10.0, 5.0)

# Previsão
if st.button("Prever risco"):
    entrada = [[ida, ieg, ips, ipp, ipv, inde]]
    pred = modelo.predict(entrada)[0]

    if pred == 1:
        st.error("Aluno em risco educacional")
    else:
        st.success("Aluno com desenvolvimento adequado")

st.set_page_config(page_title="Datathon Passos Mágicos", layout="wide")

st.title("📊 Datathon Passos Mágicos")
st.subheader("Análise e Previsão de Risco Educacional")

# menu lateral
pagina = st.sidebar.selectbox(
    "Escolha uma página",
    ["Dashboard", "Previsão de Risco", "Análise do Modelo"]
)

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if pagina == "Dashboard":

    st.header("Visão geral dos dados")

    col1, col2 = st.columns(2)

    col1.metric("Total de alunos", len(df))

    if "risco" in df.columns:
        col2.metric("Alunos em risco", df["risco"].sum())

    st.subheader("Distribuição do desempenho")

    df["IDA_2022"].hist(ax=ax)

# --------------------------------------------------
# PREVISÃO
# --------------------------------------------------

if pagina == "Previsão de Risco":

    st.header("Previsão de risco educacional")

    ida = st.slider("IDA - Desempenho Acadêmico", 0.0, 10.0)
    ieg = st.slider("IEG - Engajamento", 0.0, 10.0)
    ips = st.slider("IPS - Aspectos Psicossociais", 0.0, 10.0)
    ipp = st.slider("IPP - Indicador Psicopedagógico", 0.0, 10.0)
    ipv = st.slider("IPV - Ponto de Virada", 0.0, 10.0)
    inde = st.slider("INDE - Índice Educacional", 0.0, 10.0)

    dados = pd.DataFrame(
        [[ida, ieg, ips, ipp, ipv, inde]],
        columns=[
            "IDA_2022",
            "IEG_2022",
            "IPS_2022",
            "IPP_2022",
            "IPV_2022",
            "INDE_2022",
        ],
    )

    if st.button("Prever risco"):

        resultado = modelo.predict(dados)

        prob = modelo.predict_proba(dados)

        risco = prob[0][1] * 100

        st.write(f"Probabilidade de risco: {risco:.2f}%")

        st.progress(int(risco))

        if resultado[0] == 1:
            st.error("Aluno em risco educacional")
        else:
            st.success("Aluno com desenvolvimento adequado")

# --------------------------------------------------
# ANÁLISE DO MODELO
# --------------------------------------------------

if pagina == "Análise do Modelo":

    st.header("Importância das variáveis")

    importancias = modelo.feature_importances_

    features = [
        "IDA",
        "IEG",
        "IPS",
        "IPP",
        "IPV",
        "INDE",
    ]

    df_imp = pd.DataFrame(
        {"Indicador": features, "Importância": importancias}
    )

    df_imp = df_imp.sort_values("Importância", ascending=True)

    st.bar_chart(df_imp.set_index("Indicador"))
