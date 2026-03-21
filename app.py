import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.funcoes import (
    carregar_dados,
    carregar_modelo,
    grafico_evolucao_ian,
    prever_risco,
    grafico_evolucao_ian,
    grafico_evolucao_ida,
    grafico_heatmap,
    grafico_pairplot,
    grafico_iaa_ida_ieg,
    grafico_ips_delta,
    grafico_ipp_ian,
    grafico_corr_ipv,
    grafico_corr_inde,
    ranking_risco
)

# ============================================================
# CONFIGURAÇÃO DO APP
# ============================================================

st.set_page_config(
    page_title="Passos Mágicos – Datathon",
    layout="wide"
)

st.title("📊 Plataforma Analítica – Passos Mágicos")
st.markdown("### Datathon 2024 – Análises, Insights e Predição de Risco")

# ============================================================
# CARREGAR DADOS E MODELO
# ============================================================

df = carregar_dados()
modelo = carregar_modelo()

tabs = st.tabs(["📈 Dashboard", "🔍 Análises", "🤖 Modelo Preditivo", "🚨 Alunos em Risco"])

# ============================================================
# 1. DASHBOARD
# ============================================================

with tabs[0]:
    st.header("📈 Dashboard Geral")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Evolução do IAN ao longo dos anos")
        st.pyplot(grafico_evolucao_ian(df))

    with col2:
        st.subheader("Evolução do IDA (2022–2024)")
        st.pyplot(grafico_evolucao_ida(df))

    st.markdown("---")

    st.subheader("Correlação entre Indicadores")
    st.pyplot(grafico_heatmap(df))

# ============================================================
# 2. ANÁLISES DETALHADAS
# ============================================================

with tabs[1]:
    st.header("🔍 Análises Detalhadas")

    pergunta = st.selectbox(
        "Selecione a análise:",
        [
            "Evolução do IAN ao longo dos anos",
            "Evolução do IDA",
            "Relação IEG x IDA x IPV",
            "IAA x IDA x IEG",
            "IPS como preditor de queda",
            "IPP confirma IAN?",
            "Correlação com IPV",
            "INDE é multidimensional?"
        ]
    )

    if pergunta == "Evolução do IAN ao longo dos anos":
        st.pyplot(grafico_evolucao_ian(df))

    elif pergunta == "Evolução do IDA":
        st.pyplot(grafico_evolucao_ida(df))

    elif pergunta == "Relação IEG x IDA x IPV":
        st.pyplot(grafico_pairplot(df))

    elif pergunta == "IAA x IDA x IEG":
        st.pyplot(grafico_iaa_ida_ieg(df))

    elif pergunta == "IPS como preditor de queda":
        st.pyplot(grafico_ips_delta(df))

    elif pergunta == "IPP confirma IAN?":
        st.pyplot(grafico_ipp_ian(df))

    elif pergunta == "Correlação com IPV":
        st.pyplot(grafico_corr_ipv(df))

    elif pergunta == "INDE é multidimensional?":
        st.pyplot(grafico_corr_inde(df))

# ============================================================
# 3. MODELO PREDITIVO
# ============================================================

with tabs[2]:
    st.header("🤖 Previsão de Risco Individual")

    st.markdown("Insira os indicadores do aluno:")

    col1, col2 = st.columns(2)

    with col1:
        IAN = st.number_input("IAN", 0.0, 10.0, 5.0)
        IDA = st.number_input("IDA", 0.0, 10.0, 5.0)
        IEG = st.number_input("IEG", 0.0, 10.0, 5.0)
        IAA = st.number_input("IAA", 0.0, 10.0, 5.0)

    with col2:
        IPS = st.number_input("IPS", 0.0, 10.0, 5.0)
        IPP = st.number_input("IPP", 0.0, 10.0, 5.0)
        IPV = st.number_input("IPV", 0.0, 10.0, 5.0)
        INDE = st.number_input("INDE_2024", 0.0, 10.0, 5.0)

    if st.button("Prever Risco"):
        pred = prever_risco(modelo, IAN, IDA, IEG, IAA, IPS, IPP, IPV, INDE)

        if pred == 1:
            st.error("🚨 O aluno está em RISCO de defasagem.")
        else:
            st.success("✅ O aluno NÃO está em risco.")

# ============================================================
# 4. ALUNOS EM RISCO
# ============================================================

with tabs[3]:
    st.header("🚨 Ranking de Alunos em Risco")

    df_risco = ranking_risco(df)
    st.dataframe(df_risco)