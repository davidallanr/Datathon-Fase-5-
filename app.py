import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# carregar dados
df = pd.read_csv("DadosNormalizados-Final.csv")

# carregar modelo
modelo = pickle.load(open("modelo_risco.pkl", "rb"))

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

    fig, ax = plt.subplots()

    df["IDA_2022"].hist(ax=ax)

    st.pyplot(fig)

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
