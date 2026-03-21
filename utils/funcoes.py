import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ============================================================
# 1. CARREGAMENTO DE DADOS E MODELO
# ============================================================

def carregar_dados(caminho="dados_limpos.csv"):
    """Carrega o CSV limpo usado pelo Streamlit."""
    df = pd.read_csv(caminho)
    df["ANO"] = df["ANO"].astype(int)
    df["IDA"] = pd.to_numeric(df["IDA"], errors="coerce")
    return df

def carregar_modelo(caminho="modelo_passos_magicos.pkl"):
    """Carrega o modelo treinado."""
    with open(caminho, "rb") as f:
        model = pickle.load(f)
    return model

# ============================================================
# 2. FUNÇÃO DE PREVISÃO
# ============================================================

def prever_risco(modelo, IAN, IDA, IEG, IAA, IPS, IPP, IPV, INDE):
    """Recebe os indicadores e retorna 0 (sem risco) ou 1 (risco)."""
    entrada = np.array([[IAN, IDA, IEG, IAA, IPS, IPP, IPV, INDE]])
    pred = modelo.predict(entrada)[0]
    return int(pred)

# ============================================================
# 3. GRÁFICOS PRINCIPAIS
# ============================================================

def grafico_evolucao_ian(df):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.lineplot(data=df, x="ANO", y="IAN", estimator="mean", ci=95, marker="o", ax=ax)
    ax.set_title("Evolução do IAN ao longo dos anos")
    ax.set_xlabel("Ano")
    ax.set_ylabel("IAN médio")
    ax.grid(True, alpha=0.3)
    return fig

def grafico_evolucao_ida(df):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.lineplot(data=df, x="ANO", y="IDA", estimator="mean", ci=None, ax=ax)
    ax.set_title("Evolução do IDA (2022–2024)")
    return fig

def grafico_heatmap(df):
    indicadores = ["IAN","IDA","IEG","IAA","IPS","IPP","IPV","INDE_2024"]
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df[indicadores].corr(), annot=True, cmap="viridis", ax=ax)
    ax.set_title("Correlação entre Indicadores")
    return fig

def grafico_pairplot(df):
    fig = sns.pairplot(df[["IEG","IDA","IPV"]], diag_kind="kde")
    return fig

def grafico_iaa_ida_ieg(df):
    g = sns.pairplot(
        df[["IAA", "IDA", "IEG"]],
        kind="reg",
        diag_kind="kde",
        plot_kws={"line_kws": {"color": "red"}, "scatter_kws": {"alpha": 0.3}}
    )

    g.figure.suptitle(
        "Relação entre Autopercepção (IAA), Desempenho (IDA) e Engajamento (IEG)",
        y=1.02,
        fontsize=16
    )

    return g


def grafico_ips_delta(df):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(data=df, x="IPS", y="DELTA_IDA", hue="ANO", ax=ax)
    ax.set_title("IPS como preditor de queda de desempenho")
    return fig

def grafico_ipp_ian(df):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(data=df, x="IPP", y="IAN", hue="ANO", ax=ax)
    ax.set_title("IPP x IAN — Confirmação da Adequação")
    return fig

def grafico_corr_ipv(df):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df[["IAN","IDA","IEG","IAA","IPS","IPP","IPV"]].corr(),
                annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlação dos Indicadores com IPV")
    return fig

def grafico_corr_inde(df):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df[["IAN","IDA","IEG","IAA","IPS","IPP","IPV","INDE_2024"]].corr(),
                annot=True, cmap="viridis", ax=ax)
    ax.set_title("INDE é Multidimensional?")
    return fig

# ============================================================
# 4. RANKING DE RISCO
# ============================================================

def ranking_risco(df):
    """Retorna os alunos com maior defasagem."""
    df_risco = df[df["DEFAS_UNICA"] > 0][
        ["ID_ALUNO","DEFAS_UNICA","IDA","IEG","IPS"]
    ]
    df_risco = df_risco.sort_values(by="DEFAS_UNICA", ascending=False)
    return df_risco