import streamlit as st
import pandas as pd

st.set_page_config(page_title="Risco Educacional")

# =========================
# TÍTULO
# =========================
st.title("📊 Previsão de Risco Educacional")
st.write("Modelo baseado nos indicadores reais da base Passos Mágicos")

# =========================
# CARREGAR BASE
# =========================
try:
    df = pd.read_excel("dados_limpos base de dados.xlsx")
except:
    st.error("Erro ao carregar a base de dados.")
    st.stop()

# =========================
# LIMPAR COLUNAS
# =========================
df.columns = df.columns.str.strip().str.upper()

def find_col(keyword):
    for col in df.columns:
        if keyword in col:
            return col
    return None

col_ida = find_col("IDA")
col_ieg = find_col("IEG")
col_ips = find_col("IPS")
col_ipv = find_col("IPV")

if None in [col_ida, col_ieg, col_ips, col_ipv]:
    st.error("Erro: não foi possível encontrar as colunas necessárias.")
    st.write("Colunas disponíveis:", df.columns)
    st.stop()

# =========================
# PREPARAR DADOS
# =========================
df_modelo = df[[col_ida, col_ieg, col_ips, col_ipv]].copy()
df_modelo.columns = ["IDA", "IEG", "IPS", "IPV"]

# converter para número
for col in df_modelo.columns:
    df_modelo[col] = (
        df_modelo[col]
        .astype(str)
        .str.replace(",", ".")
        .str.strip()
    )
    df_modelo[col] = pd.to_numeric(df_modelo[col], errors="coerce")

df_modelo = df_modelo.dropna()

# normalizar
df_norm = (df_modelo - df_modelo.min()) / (df_modelo.max() - df_modelo.min())

# score de risco
df_modelo["RISCO"] = 1 - df_norm.mean(axis=1)

# =========================
# INTERFACE
# =========================
st.subheader("Insira os indicadores do aluno")

ida = st.slider("IDA - Desempenho Acadêmico", 0.0, 10.0, 5.0)
ieg = st.slider("IEG - Engajamento", 0.0, 10.0, 5.0)
ips = st.slider("IPS - Aspectos Psicossociais", 0.0, 10.0, 5.0)
ipv = st.slider("IPV - Ponto de Virada", 0.0, 10.0, 5.0)

# =========================
# PREDIÇÃO
# =========================
if st.button("Prever risco educacional"):

    entrada = pd.DataFrame([{
        "IDA": ida,
        "IEG": ieg,
        "IPS": ips,
        "IPV": ipv
    }])

    entrada_norm = (entrada - df_modelo[["IDA","IEG","IPS","IPV"]].min()) / (
        df_modelo[["IDA","IEG","IPS","IPV"]].max() - df_modelo[["IDA","IEG","IPS","IPV"]].min()
    )

    risco = 1 - entrada_norm.mean(axis=1)[0]

    # =========================
    # RESULTADO
    # =========================
    st.subheader("Resultado da previsão")

    st.write(f"📉 Probabilidade de risco educacional: **{risco*100:.2f}%**")

    st.progress(float(risco))

    # interpretação
    st.subheader("📌 Interpretação")

    if risco > 0.7:
        st.warning("Risco alto: o aluno pode estar enfrentando dificuldades significativas.")
    elif risco > 0.4:
        st.info("Risco moderado: atenção recomendada para acompanhamento.")
    else:
        st.success("Baixo risco: o aluno apresenta bom desenvolvimento.")

    # gráfico dos inputs
    st.subheader("📊 Indicadores informados")
    st.bar_chart(entrada.T)
