import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime, timedelta
import base64
from io import BytesIO
from PIL import Image
import os

st.set_page_config(layout="wide")


logo_path = "./logo_transparente_kn.png"

def load_image_as_base64(image_path):
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Carregar logo como base64 para usar em marca d'água
img_str = load_image_as_base64(logo_path)



# Definição das cores da marca
brand_colors = {
    "VERDE_TEXTO": "#d2b589",
    "VERDE_PRINCIPAL": "#2d63b2",
    "VERDE_DETALHES": "#7cb9f2",
    "CREME": "#d5d5d5",
    "CINZA": "#5c6972",
    "PRETO": "#1B1B1B"
}

# Parâmetros para cálculo do VaR
z_95 = 1.96
z_99 = 2.33

# Lista de períodos disponíveis (preset: "1 ano" – índice 2)
periodos_disponiveis = ["60 dias", "6 meses", "1 ano", "5 anos", "Base Total"]

def carregar_dados():
    xls = pd.ExcelFile("History Cot.xlsx")
    ativos = {}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=2)
        df.columns.values[0] = 'Data'
        df.columns.values[1] = 'Preço'
        df.set_index('Data', inplace=True)
        df.sort_index(inplace=True)
        ativos[sheet_name] = df
    return ativos

# Carrega os ativos reais a partir da base
ativos = carregar_dados()

def filtrar_por_periodo(df, periodo):
    hoje = datetime.today()
    if periodo == "60 dias":
        inicio = hoje - timedelta(days=60)
    elif periodo == "6 meses":
        inicio = hoje - timedelta(days=6 * 30)
    elif periodo == "1 ano":
        inicio = hoje - timedelta(days=365)
    elif periodo == "5 anos":
        inicio = hoje - timedelta(days=5 * 365)
    else:
        return df
    return df[df.index >= inicio]

@st.cache_data
def treinar_e_prever(dados_acao):
    # Seleciona os últimos 100 registros
    dados_acao = dados_acao[['Preço']].tail(100)
    cotacao = dados_acao['Preço'].to_numpy().reshape(-1, 1)
    tamanho_treino = int(len(cotacao) * 0.8)
    
    escalador = MinMaxScaler(feature_range=(0, 1))
    dados_normalizados = escalador.fit_transform(cotacao)
    
    # Preparação dos dados para treinamento
    treino_x, treino_y = [], []
    for i in range(60, tamanho_treino):
        treino_x.append(dados_normalizados[i-60:i, 0])
        treino_y.append(dados_normalizados[i, 0])
    treino_x = np.array(treino_x)
    treino_y = np.array(treino_y)
    treino_x = treino_x.reshape(treino_x.shape[0], treino_x.shape[1], 1)
    
    # Criação e treinamento do modelo LSTM
    modelo = Sequential([
        LSTM(50, return_sequences=True, input_shape=(treino_x.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    modelo.compile(optimizer="adam", loss="mean_squared_error")
    modelo.fit(treino_x, treino_y, batch_size=1, epochs=1, verbose=0)
    
    # Preparação dos dados para teste e predição
    dados_teste = dados_normalizados[tamanho_treino - 60:]
    teste_x = np.array([dados_teste[i-60:i, 0] for i in range(60, len(dados_teste))])
    teste_x = teste_x.reshape(teste_x.shape[0], teste_x.shape[1], 1)
    
    predicoes = modelo.predict(teste_x)
    predicoes = escalador.inverse_transform(predicoes)
    
    return predicoes, dados_acao['Preço'].iloc[tamanho_treino:].values

@st.cache_data
def get_direcao_projetada(ativos):
    direcoes = {}
    for ativo, dados in ativos.items():
        predicoes, _ = treinar_e_prever(dados)
        if len(predicoes) > 0:
            valor_projetado = predicoes[-1][0]
            ultimo_ajuste = dados['Preço'].iloc[-1]
            var_perc = (valor_projetado - ultimo_ajuste) / ultimo_ajuste * 100
            direcao = "Alta" if var_perc > 0 else "Baixa"
        else:
            direcao = "Predição indisponível"
        direcoes[ativo] = direcao
    return direcoes

# Interface do usuário
st.title("Predição e Volatilidade")

# Seletor de ativo e período (preset "1 ano" – índice 2)
ativo_selecionado = st.selectbox("Selecione o ativo:", list(ativos.keys()))
periodo_selecionado = st.selectbox("Selecione o período:", periodos_disponiveis, index=2)
st.write("Visualizar Médias Móveis:")
exibir_medias_moveis = st.checkbox("Exibir Médias Móveis (50, 100, 200 períodos)", value=True)
st.divider()

# Filtragem dos dados do ativo selecionado
df_ativo_original = ativos[ativo_selecionado]
df_ativo_filtrado = filtrar_por_periodo(df_ativo_original, periodo_selecionado)

# Cálculo das médias móveis (se selecionado) – calculadas com base no DataFrame original
if exibir_medias_moveis:
    df_ativo_filtrado['Media_50'] = df_ativo_original['Preço'].rolling(window=50).mean()
    df_ativo_filtrado['Media_100'] = df_ativo_original['Preço'].rolling(window=100).mean()
    df_ativo_filtrado['Media_200'] = df_ativo_original['Preço'].rolling(window=200).mean()

# Análise de predição utilizando LSTM
direcoes_mercado = get_direcao_projetada(ativos)
if ativo_selecionado in direcoes_mercado:
    st.title(f"A predição para o ativo {ativo_selecionado} é: {direcoes_mercado[ativo_selecionado]}")

# Cálculo das volatilidades (janelas de 7, 14 e 21 dias)
variacao = df_ativo_filtrado['Preço'].pct_change().dropna()
desvio_7d = variacao.rolling(window=7).std()
desvio_14d = variacao.rolling(window=14).std()
desvio_21d = variacao.rolling(window=21).std()

# Média dos desvios (último valor)
media_desvios = pd.concat([desvio_7d, desvio_14d, desvio_21d], axis=1).mean(axis=1).tail(1).values[0]
volatilidade_atual = (pd.concat([desvio_7d, desvio_14d, desvio_21d], axis=1).mean(axis=1).iloc[-1]) * 100
volatilidade_atual_decimal = volatilidade_atual / 100
var_95_diario = z_95 * volatilidade_atual_decimal
var_99_diario = z_99 * volatilidade_atual_decimal
var_95_percentual = var_95_diario * 100
var_99_percentual = var_99_diario * 100

st.metric("Volatilidade Média (7, 14, 21 dias) (%)", f"{volatilidade_atual:.3f}%")
st.write(f"Value at Risk Diário: {var_95_percentual:.3f}% (95%) / {var_99_percentual:.3f}% (99%)")

# Configuração do gráfico
fig = go.Figure()

# Linha do preço do ativo
fig.add_trace(go.Scatter(
    x=df_ativo_filtrado.index, 
    y=df_ativo_filtrado['Preço'],
    mode='lines', 
    name='Preço',
    line=dict(color=brand_colors['VERDE_DETALHES'])
))

# Linhas de médias móveis (se selecionado)
if exibir_medias_moveis:
    fig.add_trace(go.Scatter(
        x=df_ativo_filtrado.index, y=df_ativo_filtrado['Media_50'],
        mode='lines', name='Média Móvel 50',
        line=dict(color=brand_colors['VERDE_PRINCIPAL'], dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=df_ativo_filtrado.index, y=df_ativo_filtrado['Media_100'],
        mode='lines', name='Média Móvel 100',
        line=dict(color=brand_colors['CINZA'], dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=df_ativo_filtrado.index, y=df_ativo_filtrado['Media_200'],
        mode='lines', name='Média Móvel 200',
        line=dict(color=brand_colors['CREME'], dash="dot")
    ))

# Cálculo dos "desvios" (linhas de volatilidade)
dp = 8
preco_atual = df_ativo_filtrado['Preço'].iloc[-1]
desvios = [preco_atual + i * media_desvios * preco_atual for i in range(1, dp)]
desvios_neg = [preco_atual - i * media_desvios * preco_atual for i in range(1, dp)]
desvios_percentuais = [(desvio - preco_atual) / preco_atual * 100 for desvio in desvios]
desvios_neg_percentuais = [(desvio - preco_atual) / preco_atual * 100 for desvio in desvios_neg]

# Adiciona os traços de volatilidade (linhas "desvios") no gráfico
# Usamos os últimos 15 registros para o eixo X (apenas para exibir as linhas de forma fixa)
for i, (up, down, perc_up, perc_down) in enumerate(zip(desvios, desvios_neg, desvios_percentuais, desvios_neg_percentuais), start=1):
    # Linha de desvio positivo
    fig.add_trace(go.Scatter(
         x=df_ativo_filtrado.index[-15:], 
         y=[up] * len(df_ativo_filtrado.index[-15:]),
         mode='lines+text',
         line=dict(dash='dash', color=brand_colors['VERDE_TEXTO']),
         text=[''] * (len(df_ativo_filtrado.index[-15:]) - 1) + [f'+{i} DP: {up:.2f} ({perc_up:+.2f}%)'],
         textposition="top center",
         textfont=dict(color="#FFFCF5", size=12),
         showlegend=(i == 1),
         legendgroup="Desvios",
         name="Desvios" if i == 1 else None,
         opacity=0.8
    ))
    # Linha de desvio negativo
    fig.add_trace(go.Scatter(
         x=df_ativo_filtrado.index[-15:], 
         y=[down] * len(df_ativo_filtrado.index[-15:]),
         mode='lines+text',
         line=dict(dash='dash', color=brand_colors['VERDE_TEXTO']),
         text=[''] * (len(df_ativo_filtrado.index[-15:]) - 1) + [f'-{i} DP: {down:.2f} ({perc_down:+.2f}%)'],
         textposition="bottom center",
         textfont=dict(color="#FFFCF5", size=12),
         showlegend=False,
         legendgroup="Desvios",
         opacity=0.8
    ))

# Atualiza o layout do gráfico, definindo o eixo Y à direita
fig.update_layout(
    title=f"Gráfico do Ativo {ativo_selecionado} ({periodo_selecionado})",
    xaxis_title="Data",
    yaxis_title="Preço",
    template="plotly_white",
    yaxis=dict(side='right', showgrid=False)
)

# Logo centralizado no gráfico
fig.add_layout_image(
    dict(
        source=f'data:image/png;base64,{img_str}',
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        sizex=0.8, sizey=0.8,
        xanchor="center", yanchor="middle",
        opacity=0.6,
        layer="below"
    )
)

st.plotly_chart(fig, use_container_width=True)
