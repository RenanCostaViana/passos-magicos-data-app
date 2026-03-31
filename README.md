# Plataforma Analítica Passos Mágicos – 2024

## Objetivo do Projeto

Este projeto foi desenvolvido para o Datathon Passos Mágicos 2024 com o propósito de:

- analisar indicadores pedagógicos dos alunos (IAN, IDA, IEG, IPS, IPP, IPV, IAA, INDE);
- identificar padrões de desempenho, engajamento e percepção;
- unificar a defasagem escolar entre anos diferentes;
- construir um modelo preditivo de risco de defasagem;
- disponibilizar uma plataforma interativa em Streamlit para visualização e tomada de decisão.

O resultado é uma solução que combina análise exploratória, machine learning e visualização interativa.

---

## Principais Entregas

- Dashboard interativo com gráficos e indicadores.
- Análises detalhadas respondendo às perguntas do Datathon.
- Modelo preditivo (Random Forest) para risco de defasagem.
- Ranking de alunos em risco.
- Previsão individual de risco com base nos indicadores.
- Base de dados padronizada e limpa para uso no Streamlit.

---

## Estrutura do Projeto
```text
projeto/
│
├── app.py                     # Aplicação Streamlit
├── utils/
│     └── funcoes.py           # Funções auxiliares (gráficos, modelo, dados)
│
├── dados_limpos.csv           # Base final usada pelo Streamlit
├── modelo_passos_magicos.pkl  # Modelo treinado
│
├── requirements.txt           # Dependências do projeto
├── .gitignore                 # Arquivos ignorados pelo Git
└── README.md                  # Este arquivo
```

---

## Como Executar o Projeto

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Executar o Streamlit

```bash
streamlit run app.py
```

A aplicação abrirá automaticamente no navegador.

---

## Sobre os Dados

Os dados originais foram fornecidos pela Passos Mágicos e incluem:

- indicadores pedagógicos (IAN, IDA, IEG, IAA, IPS, IPP, IPV);
- INDE (Índice de Desenvolvimento);
- defasagem escolar (com nomes diferentes entre anos);
- informações de identificação e ano.

### Padronizações realizadas

- criação da coluna DEFAS_UNICA (unificando DEFAS e DEFASAGEM);
- criação de DELTA_IDA (variação do desempenho);
- conversão de indicadores para valores numéricos;
- remoção de inconsistências e valores inválidos.

O arquivo final utilizado pelo Streamlit é dados_limpos.csv.

---

## Modelo Preditivo

O modelo utilizado é um Random Forest Classifier, treinado com:

- IAN
- IDA
- IEG
- IAA
- IPS
- IPP
- IPV
- INDE_2024

### Target
DEFAS_UNICA > 0
(1 = aluno em risco de defasagem, 0 = aluno adequado)

### Balanceamento
Foi aplicado SMOTE para lidar com desbalanceamento entre classes.

---

## Funcionalidades do Streamlit

### Dashboard
- Defasagem por ano
- Evolução do IDA
- Heatmap de correlação

### Análises
- Relação entre indicadores
- Coerência entre autoavaliação e desempenho
- Preditores de queda
- Multidimensionalidade do INDE

### Modelo Preditivo
- Previsão individual de risco

### Alunos em Risco
- Ranking ordenado por defasagem
- Indicadores relevantes para intervenção

---

## Impacto Esperado

A plataforma permite:

- identificar rapidamente alunos em risco;
- apoiar decisões pedagógicas baseadas em dados;
- visualizar padrões de engajamento e desempenho;
- acompanhar evolução dos indicadores ao longo dos anos;
- fortalecer a atuação da Passos Mágicos com insights acionáveis.

---

## Autor

Projeto desenvolvido por Renan Costa Viana para o Datathon da Pós Tech de Data Analytics 2025
