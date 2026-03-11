# 🪄 Análise de Dados e Modelo Preditivo: Associação Passos Mágicos (2022-2024)

## 📌 Sobre a Associação

A **Associação Passos Mágicos** atua há 32 anos transformando a trajetória de crianças e jovens de baixa renda, proporcionando melhores oportunidades de vida por meio da educação e apoio psicossocial.

## 🎯 Objetivos do Projeto

Este trabalho visa extrair inteligência dos dados da associação para otimizar o acompanhamento dos alunos, focando em quatro pilares:

1. **ETL & Análise Exploratória:** Limpeza e tratamento dos dados referentes aos anos de 2022, 2023 e 2024.
2. **Inteligência de Negócio:** Responder às dores e dúvidas estratégicas relatadas nos documentos de apoio da instituição.
3. **Ciência de Dados:** Desenvolvimento de um **modelo preditivo** para identificar a probabilidade de um aluno entrar em risco de defasagem escolar.
4. **Deployment:** Criação de uma aplicação interativa em **Streamlit** para consulta do modelo e visualização dos indicadores.

---

## 🛠️ Tratamento de Dados (ETL)

A base original (PEDE) foi desmembrada em três conjuntos anuais (`2022.xlsx`, `2023.xlsx` e `2024.xlsx`) para garantir a integridade histórica. Aplicamos as seguintes padronizações:

* **Gênero:** Unificação para o padrão *Feminino/Masculino*.
* **Fase Escolar:** Padronização de formatos heterogêneos para a nomenclatura oficial (*ALFA / FASE 1* até *FASE 8*).
* **Correção de Tipografia:** Ajustes em nomes de pedras (ex: *Agata* → *Ágata*) e limpeza de registros inconsistentes.
* **Renomeação de Colunas:** Simplificação técnica para facilitar a codificação (ex: *Matem* → `Mat`, *Portug* → `Por`, *Defas* → `Defasagem`).
* **Categorização Institucional:** Consolidação dos tipos de escola em: *Pública, Privada, Privada (Bolsa), Privada (Empresa)* e *Concluído*.
* **Conversão de Tipos:** Transformação do INDE 2024 de texto para formato numérico (`float`) para cálculos estatísticos.

---

## 🚀 Tecnologias Utilizadas

* **Linguagem:** Python
* **Manipulação de Dados:** Pandas / NumPy
* **Visualização:** Plotly / Matplotlib / Seaborn
* **Machine Learning:** Scikit-Learn
* **Interface Web:** Streamlit

---

## 📂 Como Executar o Projeto

1. Clone este repositório:
```bash
git clone https://github.com/seu-usuario/passos-magicos-analise.git

```


2. Instale as dependências:
```bash
pip install -r requirements.txt

```


3. Execute o dashboard:
```bash
streamlit run app.py

```





