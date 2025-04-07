import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import base64
from io import BytesIO
from PIL import Image
import os

logo_path = "./logo_transparente_kn.png"

def load_image_as_base64(image_path):
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Carregar logo como base64 para usar em marca d'água
img_str = load_image_as_base64(logo_path)

st.set_page_config(page_title="Relatório Quant - Cambirela", layout="wide")
st.title("📊 Relatório de Risco e Predição - Cambirela Quant")

col10, col11 = st.columns([2, 2])
with col10:
    # Exibir a figura interativa no Streamlit
    st.markdown("""
    Este relatório apresenta uma análise quantitativa de ativos com foco em:
    - **Indicadores Técnicos** (RSI, Bandas de Bollinger, Drawdown)
    - **Z-score com base na MM100**
    - **Predição com LSTM**
    - **Exponente de Hurst**
    - **Volatilidade Condicional via TGARCH**
    """)
    

with col11:
    st.markdown("""
    🧠 **Como interpretar**:
    - Z-score > 2 → Ativo sobrecomprado  
    - Z-score < -2 → Ativo sobrevendido  
    - Hurst > 0.5 → Tendência persistente  
    - Hurst < 0.5 → Comportamento aleatório  
    - RSI > 70 → Sobrecompra  
    - RSI < 30 → Sobrevenda  
    - MAPE: Indica o erro percentual médio entre o valor previsto e o valor real. Quanto menor, melhor.  
    - MAPE < 5% → Excelente  
    - 5% a 10% → Muito bom  
    - 10% a 20% → Aceitável  
    - > 20% → Atenção: erro elevado
    """)

cores = {
    "primarias": ["#7cb9f2", "#2d63b2", "#050835", "#010207"],
    "secundarias": ["#eee5d2", "#d2b589", "#b79347", "#d5d5d5", "#b7b7b7", "#5c6972"]
}

periodos_disponiveis = ["60 dias", "6 meses", "1 ano", "5 anos", "Base Total"]

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
def carregar_dados():
    xls = pd.ExcelFile("History Cot.xlsx")
    ativos = {}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=2)
        df.columns.values[0] = 'Data'
        df.columns.values[1] = 'Preço'
        df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
        df.dropna(subset=['Data', 'Preço'], inplace=True)
        df.set_index('Data', inplace=True)
        df.sort_index(inplace=True)
        ativos[sheet_name] = df
    return ativos

@st.cache_data
def treinar_e_prever(dados_acao):
    dados_acao = dados_acao[['Preço']].dropna().tail(100)
    if len(dados_acao) < 70:
        return pd.Series(dtype=float)

    cotacao = dados_acao['Preço'].to_numpy().reshape(-1, 1)
    tamanho_treino = int(len(cotacao) * 0.8)

    escalador = MinMaxScaler()
    dados_normalizados = escalador.fit_transform(cotacao)

    treino_x, treino_y = [], []
    for i in range(60, tamanho_treino):
        treino_x.append(dados_normalizados[i-60:i, 0])
        treino_y.append(dados_normalizados[i, 0])

    if len(treino_x) == 0:
        return pd.Series(dtype=float)

    treino_x = np.array(treino_x).reshape(-1, 60, 1)
    treino_y = np.array(treino_y)

    modelo = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])
    modelo.compile(optimizer='adam', loss='mean_squared_error')
    modelo.fit(treino_x, treino_y, batch_size=1, epochs=1, verbose=0)

    dados_teste = dados_normalizados[tamanho_treino - 60:]
    teste_x = np.array([dados_teste[i-60:i, 0] for i in range(60, len(dados_teste))])
    teste_x = teste_x.reshape(-1, 60, 1)

    predicoes = modelo.predict(teste_x)
    predicoes = escalador.inverse_transform(predicoes)
    datas_pred = dados_acao.index[tamanho_treino:]
    return pd.Series(predicoes.flatten(), index=datas_pred)


def calcular_volatilidade_tgarch(df, alpha=0.05, omega=0.000001, gamma=0.1, beta=0.85):
    """
    Calcula a volatilidade condicional com base no modelo TGARCH.
    df: DataFrame contendo a coluna 'Retornos'
    """
    retornos = df['Retornos'].dropna()
    sigma2 = np.zeros_like(retornos)
    sigma2[0] = np.var(retornos)

    for t in range(1, len(retornos)):
        retorno_anterior = retornos.iloc[t-1]
        impacto_negativo = gamma * retorno_anterior**2 if retorno_anterior < 0 else 0
        sigma2[t] = omega + alpha * retorno_anterior**2 + impacto_negativo + beta * sigma2[t-1]

    return pd.Series(np.sqrt(sigma2), index=retornos.index)


def calcular_rsi(series, period=14):
    delta = series.diff()
    ganho = delta.where(delta > 0, 0)
    perda = -delta.where(delta < 0, 0)
    media_ganho = ganho.rolling(period).mean()
    media_perda = perda.rolling(period).mean()
    rs = media_ganho / media_perda
    return 100 - (100 / (1 + rs))


def calcular_hurst(df, janela=120):
    hurst_vals = []
    datas = []
    for i in range(janela, len(df)):
        janela_df = df.iloc[i-janela:i]
        ts = janela_df['Preço'].dropna().values
        if len(ts) < janela:
            hurst_vals.append(np.nan)
            datas.append(janela_df.index[-1])
            continue

        lags = range(2, 20)
        tau = []
        valid_lags = []

        for lag in lags:
            if lag < len(ts):
                diff = ts[lag:] - ts[:-lag]
                std = np.std(diff)
                if std > 0:
                    tau.append(std)
                    valid_lags.append(lag)

        if len(tau) < 5:
            hurst_vals.append(np.nan)
        else:
            hurst = np.polyfit(np.log(valid_lags), np.log(tau), 1)[0]
            hurst_vals.append(hurst)
        datas.append(janela_df.index[-1])

    return pd.Series(hurst_vals, index=datas)

def calcular_rsi(series, period=14):
    delta = series.diff()
    ganho = delta.where(delta > 0, 0)
    perda = -delta.where(delta < 0, 0)
    media_ganho = ganho.rolling(period).mean()
    media_perda = perda.rolling(period).mean()
    rs = media_ganho / media_perda
    return 100 - (100 / (1 + rs))

# Execução principal
ativos = carregar_dados()
ativo_selecionado = st.selectbox("📌 Selecione o Ativo:", list(ativos.keys()), index=list(ativos.keys()).index("WDOFUT"))
periodo_escolhido = st.selectbox("📅 Período de Análise:", periodos_disponiveis, index=periodos_disponiveis.index("1 ano"))

df = ativos[ativo_selecionado].copy()
df = filtrar_por_periodo(df, periodo_escolhido)

df_ativo_original = ativos[ativo_selecionado]
df_ativo_filtrado = filtrar_por_periodo(df_ativo_original, periodo_escolhido)
if len(df) < 70:
    st.warning(f"⚠️ Apenas {len(df)} registros após o filtro. Resultados limitados.")
else:
    df['Retornos'] = df['Preço'].pct_change()
    df['RSI'] = calcular_rsi(df['Preço'])
    df['MM20'] = df['Preço'].rolling(20).mean()
    df['Desvio'] = df['Preço'].rolling(20).std()
    df['Bollinger Sup'] = df['MM20'] + 2 * df['Desvio']
    df['Bollinger Inf'] = df['MM20'] - 2 * df['Desvio']
    df['Pico'] = df['Preço'].cummax()
    df['Drawdown'] = df['Preço'] / df['Pico'] - 1
    df['MM100'] = df['Preço'].rolling(100).mean()
    df['Desvio100'] = df['Preço'].rolling(100).std()
    df['Z-score'] = (df['Preço'] - df['MM100']) / df['Desvio100']
    hurst = calcular_hurst(df)
    max_drawdown = df['Drawdown'].min()
    data_max_dd = df['Drawdown'].idxmin()

    df_pred = treinar_e_prever(df)
    df['Predito'] = df_pred if not df_pred.empty else np.nan
    

    
    # Cálculo dos retornos e da volatilidade condicional
    df_ativo_filtrado['Retornos'] = df_ativo_filtrado['Preço'].pct_change()
    df_ativo_filtrado['Vol_TGARCH'] = calcular_volatilidade_tgarch(df_ativo_filtrado)

    # Último valor da volatilidade estimada
    vol_atual = df_ativo_filtrado['Vol_TGARCH'].dropna().iloc[-1]
    preco_atual = df_ativo_filtrado['Preço'].iloc[-1]

    # Faixas de desvio padrão baseadas na volatilidade TGARCH
    dp = 8
    desvios = [preco_atual + i * vol_atual * preco_atual for i in range(1, dp)]
    desvios_neg = [preco_atual - i * vol_atual * preco_atual for i in range(1, dp)]
    desvios_percentuais = [(desvio - preco_atual) / preco_atual * 100 for desvio in desvios]
    desvios_neg_percentuais = [(desvio - preco_atual) / preco_atual * 100 for desvio in desvios_neg]
    # Gráfico
    fig_tgarch = go.Figure()
    fig_tgarch.add_trace(go.Scatter(
        x=df.index,
        y=df['Preço'],
        mode='lines',
        name='Preço',
        line=dict(color=cores['primarias'][0])
    ))

    for i, (up, down, perc_up, perc_down) in enumerate(zip(desvios, desvios_neg, desvios_percentuais, desvios_neg_percentuais), start=1):
        fig_tgarch.add_trace(go.Scatter(
            x=df.index[-15:], y=[up] * 15,
            mode='lines+text',
            line=dict(dash='dash', color=cores['primarias'][1]),
            text=[''] * 14 + [f'+{i} DP: {up:.2f} ({perc_up:+.2f}%)'],
            textposition="top center",
            textfont=dict(size=12),
            showlegend=(i == 1),
            legendgroup="TGARCH",
            name="Volatilidade TGARCH" if i == 1 else None,
            opacity=0.8
        ))
        fig_tgarch.add_trace(go.Scatter(
            x=df.index[-15:], y=[down] * 15,
            mode='lines+text',
            line=dict(dash='dash', color=cores['primarias'][1]),
            text=[''] * 14 + [f'-{i} DP: {down:.2f} ({perc_down:+.2f}%)'],
            textposition="bottom center",
            textfont=dict(size=12),
            showlegend=False,
            legendgroup="TGARCH",
            opacity=0.8
        ))

    fig_tgarch.update_layout(
        title=f"Volatilidade Condicional (TGARCH) - {ativo_selecionado} ({periodo_escolhido})",
        xaxis_title="Data",
        yaxis_title="Preço",
        template="plotly_white",
        yaxis=dict(side='right', showgrid=False)
    )
    
    fig_tgarch.add_layout_image(
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

    st.plotly_chart(fig_tgarch, use_container_width=True)


    st.subheader("📈 Preço, Bandas de Bollinger e Predição")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Preço'], name="Preço", line=dict(color=cores['primarias'][0])))
    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger Sup'], name="Bollinger Sup", line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger Inf'], name="Bollinger Inf", line=dict(dash='dot')))
    if not df_pred.empty:
        fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred.values, name="Predição", line=dict(color="green", dash="dot")))
    fig.update_layout(height=400, yaxis=dict(side="right"))
    
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

    st.subheader("📊 Z-score com MM100")
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(x=df.index, y=df['Z-score'], name="Z-score", line=dict(color=cores['primarias'][1])))
    fig_z.add_hline(y=2, line=dict(color='red', dash='dot'))
    fig_z.add_hline(y=-2, line=dict(color='green', dash='dot'))
    fig_z.update_layout(height=300, yaxis_title="Z-score", yaxis=dict(side="right"))
    fig_z.add_layout_image(
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
    st.plotly_chart(fig_z, use_container_width=True)

    st.subheader("🔄 RSI - Índice de Força Relativa")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color=cores['primarias'][1])))
    fig_rsi.add_hline(y=70, line=dict(color="red", dash="dot"), annotation_text="🔴 Sobrecompra (RSI > 70)", annotation_position="top right")
    fig_rsi.add_hline(y=30, line=dict(color="green", dash="dot"), annotation_text="🟢 Sobrevenda (RSI < 30)", annotation_position="bottom right")
    fig_rsi.update_layout(height=300, yaxis_range=[0, 100], yaxis_title="RSI", yaxis=dict(side="right"))
    fig_rsi.add_layout_image(
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
    st.plotly_chart(fig_rsi, use_container_width=True)

    st.subheader("📉 Drawdown no período")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=df.index, y=df['Drawdown'], name="Drawdown", line=dict(color=cores['primarias'][1])))
    fig_dd.add_trace(go.Scatter(x=[data_max_dd], y=[max_drawdown],
        mode='markers+text',
        marker=dict(size=10, color="red"),
        text=[f"🟥 Máx DD: {max_drawdown:.2%}"],
        textposition="top center",
        name="Máximo DD"
    ))
    fig_dd.update_layout(height=300, yaxis_title="Drawdown", yaxis=dict(side="right"))
    fig_dd.add_layout_image(
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
    st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("📏 Exponente de Hurst (Rolling)")
    df['Hurst Rolling'] = calcular_hurst(df)

    fig_hurst = go.Figure()
    fig_hurst.add_trace(go.Scatter(x=df['Hurst Rolling'].index, y=df['Hurst Rolling'], name="Hurst Rolling",
                                line=dict(color="#7cb9f2", dash="solid")))
    fig_hurst.update_layout(height=300, yaxis_range=[0, 1], yaxis_title="Hurst", yaxis=dict(side="right"))
    fig_hurst.add_layout_image(
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
    st.plotly_chart(fig_hurst, use_container_width=True)

    # Exibir último valor como métrica
    ultimo_valor = df['Hurst Rolling'].dropna().iloc[-1] if df['Hurst Rolling'].dropna().size > 0 else np.nan
    if not np.isnan(ultimo_valor):
        st.metric("📌 Último Hurst Rolling", f"{ultimo_valor:.2f}")
    else:
        st.metric("📌 Último Hurst Rolling", "Dados insuficientes")

    st.subheader("📌 Avaliação de Predição (MAPE)")
    df_valid = df.dropna(subset=['Predito'])
    if not df_valid.empty:
        df_valid['ErroAbs'] = np.abs((df_valid['Preço'] - df_valid['Predito']) / df_valid['Preço']) * 100
        mape = df_valid['ErroAbs'].mean()
        st.metric("MAPE da Predição", f"{mape:.2f}%")

        fig_mape = go.Figure()
        fig_mape.add_trace(go.Scatter(x=df_valid.index, y=df_valid['ErroAbs'], name="Erro (%)", line=dict(color=cores['primarias'][1])))
        fig_mape.update_layout(height=300, yaxis_title="Erro (%)", yaxis=dict(side="right"))
        fig_mape.add_layout_image(
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
        st.plotly_chart(fig_mape, use_container_width=True)
    else:
        st.info("Sem valores suficientes para calcular MAPE.")
        
st.markdown("---")
st.markdown("**Desenvolvido por Kauan Nunes - Trader QUANT**")
