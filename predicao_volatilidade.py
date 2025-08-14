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
        df['Data'] = pd.to_datetime(df['Data'])
        df.set_index('Data', inplace=True)
        df.sort_index(inplace=True)
        ativos[sheet_name] = df
    return ativos

# Carrega os ativos reais a partir da base
ativos = carregar_dados()

def filtrar_por_periodo(df, periodo, data_final):
    # garante índice datetime e ordenado
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # recorta até a data_final (inclusivo)
    data_final = pd.Timestamp(data_final)
    df = df.loc[:data_final]

    if periodo == "Base Total":
        return df

    if periodo == "60 dias":
        inicio = data_final - pd.Timedelta(days=60)
    elif periodo == "6 meses":
        inicio = data_final - pd.Timedelta(days=6 * 30)
    elif periodo == "1 ano":
        inicio = data_final - pd.Timedelta(days=365)
    elif periodo == "5 anos":
        inicio = data_final - pd.Timedelta(days=5 * 365)
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
ativos_keys = list(ativos.keys())
default_index = ativos_keys.index("WDOFUT") if "WDOFUT" in ativos_keys else 0
col1, col2 = st.columns([2, 1])  # 3x mais espaço para a primeira coluna

# Primeira coluna - ativo, período e médias móveis
with col1:
    ativo_selecionado = st.selectbox(
        "Selecione o ativo:", ativos_keys, index=default_index
    )
    periodo_selecionado = st.selectbox(
        "Selecione o período:", periodos_disponiveis, index=2
    )
    st.write("Visualizar Médias Móveis:")
    exibir_medias_moveis = st.checkbox(
        "Exibir Médias Móveis (50, 100, 200 períodos)", value=True
    )

# Segunda coluna - data final
with col2:
    hoje_date = datetime.today().date()
    data_final = st.date_input(
        "Data final",
        value=hoje_date,
        max_value=hoje_date,
        format="DD/MM/YYYY"
    )
    data_final_ts = pd.Timestamp(data_final)

st.divider()

# Filtragem dos dados do ativo selecionado
df_ativo_original = ativos[ativo_selecionado]
df_ativo_filtrado = filtrar_por_periodo(df_ativo_original, periodo_selecionado, data_final_ts)

if df_ativo_filtrado.empty:
    st.warning("Sem dados no período selecionado até a data final escolhida.")
    st.stop()
    
# Cálculo das médias móveis (se selecionado) – calculadas com base no DataFrame original
if exibir_medias_moveis:
    mm50 = df_ativo_original['Preço'].rolling(window=50).mean()
    mm100 = df_ativo_original['Preço'].rolling(window=100).mean()
    mm200 = df_ativo_original['Preço'].rolling(window=200).mean()

    df_ativo_filtrado = df_ativo_filtrado.copy()
    df_ativo_filtrado['Media_50'] = mm50.reindex(df_ativo_filtrado.index)
    df_ativo_filtrado['Media_100'] = mm100.reindex(df_ativo_filtrado.index)
    df_ativo_filtrado['Media_200'] = mm200.reindex(df_ativo_filtrado.index)

try:
    preds, y_real = treinar_e_prever(df_ativo_filtrado)
    if len(preds) > 0:
        valor_proj = float(preds[-1][0])
        ultimo = float(df_ativo_filtrado['Preço'].iloc[-1])
        var_perc = (valor_proj - ultimo) / ultimo * 100
        direcao = "Alta" if var_perc > 0 else "Baixa"
        st.title(f"A predição para o ativo {ativo_selecionado} é: {direcao}")
    else:
        st.title("Predição indisponível para o recorte selecionado.")
except Exception as e:
    st.warning(f"Predição indisponível: {e}")

# ==============================
# Volatilidades (7, 14, 21)
# ==============================
variacao = df_ativo_filtrado['Preço'].pct_change().dropna()
desvio_7d = variacao.rolling(window=7).std()
desvio_14d = variacao.rolling(window=14).std()
desvio_21d = variacao.rolling(window=21).std()

media_desvios = pd.concat([desvio_7d, desvio_14d, desvio_21d], axis=1).mean(axis=1).tail(1).values[0]
media_desvios = media_desvios / 2
volatilidade_atual = (pd.concat([desvio_7d, desvio_14d, desvio_21d], axis=1).mean(axis=1).iloc[-1]) * 100
volatilidade_atual_decimal = volatilidade_atual / 100
var_95_diario = z_95 * volatilidade_atual_decimal
var_99_diario = z_99 * volatilidade_atual_decimal
var_95_percentual = var_95_diario * 100
var_99_percentual = var_99_diario * 100

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
dp = 6
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
         text=[''] * (len(df_ativo_filtrado.index[-15:]) - 1) + [f'+{i} VOL: {up:.2f} ({perc_up:+.2f}%)'],
         textposition="top center",
         textfont=dict(color="#FFFCF5", size=12),
         showlegend=(i == 0.5),
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
         showlegend=(i == 0.5),
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
        sizex=1.2, sizey=1.2,
        xanchor="center", yanchor="middle",
        opacity=0.5,
        layer="below"
    )
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# Métricas (todas já respeitando data_final)
# ==============================
mes_atual_inicio = datetime(data_final_ts.year, data_final_ts.month, 1)
df_mes_atual = df_ativo_filtrado[df_ativo_filtrado.index >= mes_atual_inicio]

preco_atual = df_ativo_filtrado['Preço'].iloc[-1]
preco_ontem = df_ativo_filtrado['Preço'].iloc[-2] if len(df_ativo_filtrado) > 1 else np.nan
variacao_dia = ((preco_atual - preco_ontem) / preco_ontem) * 100 if pd.notna(preco_ontem) else 0.0


if not df_mes_atual.empty:
    preco_inicio_mes = df_mes_atual['Preço'].iloc[0]  # Primeiro preço do mês
    variacao_mensal = ((preco_atual - preco_inicio_mes) / preco_inicio_mes) * 100
else:
    preco_inicio_mes = preco_atual  # Caso não haja dados para o mês atual
    variacao_mensal = 0.0

preco_atual = df_ativo_filtrado['Preço'].iloc[-1]
preço_ontem = df_ativo_filtrado['Preço'].iloc[-2] if len(df_ativo_filtrado) > 1 else preco_atual
maximo = df_ativo_filtrado['Preço'].max()
minimo = df_ativo_filtrado['Preço'].min()
variacao_dia = ((preco_atual - preço_ontem) / preço_ontem) * 100 if preço_ontem else 0
volatilidade = df_ativo_filtrado['Preço'].pct_change().std() * np.sqrt(len(df_ativo_filtrado)) * 100


desvios = [preco_atual + i * media_desvios * preco_atual for i in range(1, dp)]
desvios_percentuais = [(desvio - preco_atual) / preco_atual * 100 for desvio in desvios]
desvios_neg = [preco_atual - i * media_desvios * preco_atual for i in range(1, dp)]
desvios_neg_percentuais = [(desvio - preco_atual) / preco_atual * 100 for desvio in desvios_neg]


preco_inicio_base = df_ativo_filtrado['Preço'].iloc[0]  # Primeiro preço da base filtrada
variacao_base_filtrada = ((preco_atual - preco_inicio_base) / preco_inicio_base) * 100

# Filtrar o primeiro dia do mês atual na base
mes_atual_inicio = datetime(datetime.today().year, datetime.today().month, 1)
df_mes_atual = df_ativo_filtrado[df_ativo_filtrado.index >= mes_atual_inicio]


df_ativo_filtrado['Variação'] = df_ativo_filtrado['Preço'].pct_change()

# Inicializar variáveis para sequências
dias_seq_positiva = []
dias_seq_negativa = []

current_seq_positiva = []
current_seq_negativa = []

max_seq_positiva = 0
max_seq_negativa = 0


dias_var_positiva = df_ativo_filtrado[df_ativo_filtrado['Variação'] > (var_95_percentual/100)].index
dias_var_negativa = df_ativo_filtrado[df_ativo_filtrado['Variação'] < -(var_95_percentual/100)].index

# Loop para identificar sequências e armazenar datas
for idx, var in enumerate(df_ativo_filtrado['Variação']):
    if var > 0:
        current_seq_positiva.append(df_ativo_filtrado.index[idx])
        current_seq_negativa = []
    elif var < 0:
        current_seq_negativa.append(df_ativo_filtrado.index[idx])
        current_seq_positiva = []
    else:
        current_seq_positiva = []
        current_seq_negativa = []
    
    if len(current_seq_positiva) > max_seq_positiva:
        max_seq_positiva = len(current_seq_positiva)
        dias_seq_positiva = current_seq_positiva[:]
    
    if len(current_seq_negativa) > max_seq_negativa:
        max_seq_negativa = len(current_seq_negativa)
        dias_seq_negativa = current_seq_negativa[:]


seq_atual = 0
alta_ou_baixa = "neutra"

if not df_ativo_filtrado['Variação'].empty:
    ultima_var = df_ativo_filtrado['Variação'].iloc[-1]

    if ultima_var > 0:
        alta_ou_baixa = "↑"
        for var in reversed(df_ativo_filtrado['Variação']):
            if var > 0:
                seq_atual += 1
            else:
                break
    elif ultima_var < 0:
        alta_ou_baixa = "↓"
        for var in reversed(df_ativo_filtrado['Variação']):
            if var < 0:
                seq_atual += 1
            else:
                break

df_ativo_filtrado['Media_50'] = df_ativo_filtrado['Preço'].rolling(window=50).mean()
df_ativo_filtrado['Media_100'] = df_ativo_filtrado['Preço'].rolling(window=100).mean()
df_ativo_filtrado['Media_200'] = df_ativo_filtrado['Preço'].rolling(window=200).mean()

# Calcular os spreads entre o preço e as médias móveis
df_ativo_filtrado['Spread_MM50'] = df_ativo_filtrado['Preço'] - df_ativo_filtrado['Media_50']
df_ativo_filtrado['Spread_MM100'] = df_ativo_filtrado['Preço'] - df_ativo_filtrado['Media_100']
df_ativo_filtrado['Spread_MM200'] = df_ativo_filtrado['Preço'] - df_ativo_filtrado['Media_200']

df_ativo_ultimo_spread_mm50 = df_ativo_filtrado['Spread_MM50'][-1]
df_ativo_ultimo_spread_mm100 = df_ativo_filtrado['Spread_MM100'][-1]
df_ativo_ultimo_spread_mm200 = df_ativo_filtrado['Spread_MM200'][-1]

# Calcular o desvio padrão dos spreads
desvio_spread_100 = df_ativo_filtrado['Spread_MM100'].std()
desvio_spread_200 = df_ativo_filtrado['Spread_MM200'].std()

# Limites de 95% de confiança para os spreads
limite_inferior_100 = df_ativo_filtrado['Spread_MM100'].mean() - z_95 * desvio_spread_100
limite_superior_100 = df_ativo_filtrado['Spread_MM100'].mean() + z_95 * desvio_spread_100

limite_inferior_200 = df_ativo_filtrado['Spread_MM200'].mean() - z_95 * desvio_spread_200
limite_superior_200 = df_ativo_filtrado['Spread_MM200'].mean() + z_95 * desvio_spread_200

# Filtrar os maiores valores registrados dos spreads
maiores_spreads_100 = df_ativo_filtrado[df_ativo_filtrado['Spread_MM100'] > limite_superior_100]
maiores_spreads_200 = df_ativo_filtrado[df_ativo_filtrado['Spread_MM200'] > limite_superior_200]

st.title(f"Dados Quantitativos para o {ativo_selecionado}")

cols = st.columns(5)

# Linha 1
cols[0].metric("Último Ajuste", f"{preco_atual:,.3f}")
cols[1].metric("Variação do Dia (%)", f"{variacao_dia:.2f}%")
cols[2].metric("Volatilidade Média (7, 14, 21 dias) (%)", f"{volatilidade_atual:.3f}%")
cols[3].metric("Value at Risk (%)", f"{var_95_percentual:,.3f}% (95%)")
cols[4].metric("Var % Base Filtrada", f"{variacao_base_filtrada:.2f}%")

# Linha 2
cols = st.columns(6)
cols[0].metric("Var % Mensal", f"{variacao_mensal:.2f}%")
cols[1].metric("Máx. Seq. Negativa", f"{max_seq_negativa} dias")
cols[2].metric("Seq. Atual", f"{seq_atual} {alta_ou_baixa}")
cols[3].metric("Máx. Seq. Positiva", f"{max_seq_positiva} dias")
cols[4].metric("Mínima do Período", f"{minimo:,.3f}")
cols[5].metric("Máxima do Período", f"{maximo:,.3f}")


# Linha 3 (condicional: médias móveis)
if exibir_medias_moveis:
    cols = st.columns(5)
    cols[0].metric("Spread MM50", f"{df_ativo_ultimo_spread_mm50:.3f}")    
    cols[1].metric("Spread MM100", f"{df_ativo_ultimo_spread_mm100:.3f}")
    cols[2].metric("Spread MM200", f"{df_ativo_ultimo_spread_mm200:.3f}")
    cols[3].metric("Máx. Spread MM100 (VaR 95%)", f"{limite_superior_200:.3f}")
    cols[4].metric("Máx. Spread MM200 (VaR 95%)", f"{limite_superior_100:.3f}")

# Rodapé
st.markdown("---")
st.markdown("**Desenvolvido por Kauan Nunes - Trader QUANT**")




