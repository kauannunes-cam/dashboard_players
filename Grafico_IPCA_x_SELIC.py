import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import base64

# Função para carregar a logo como base64
def load_image_as_base64(image_path):
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Caminho da logo
logo_path = "logo_transparente.png"
img_str = load_image_as_base64(logo_path)

# Definindo as cores da marca
brand_colors = {
    "VERDE_TEXTO": "#1F4741",
    "VERDE_PRINCIPAL": "#2B6960",
    "VERDE_DETALHES": "#49E2B1",
    "CREME": "#FFFCF5",
    "CINZA": "#76807D",
    "PRETO": "#1B1B1B"
}

# Configurando o layout do Streamlit
st.set_page_config(page_title="IPCA vs Selic - Dashboard", layout="wide")
st.title("IPCA Acumulado 12 Meses vs Taxa Selic")

# Função para obter dados de uma série temporal do Banco Central
def fetch_data_from_bcb(api_url):
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
            df['valor'] = pd.to_numeric(df['valor'])
            return df
        else:
            st.error(f"Erro ao buscar dados: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro de conexão: {e}")
        return pd.DataFrame()

# URLs das séries temporais
url_ipca = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=json"  # IPCA mensal
url_selic = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1178/dados?formato=json"  # Selic mensal

# Carregando os dados
df_ipca = fetch_data_from_bcb(url_ipca)
df_selic = fetch_data_from_bcb(url_selic)

if df_ipca.empty or df_selic.empty:
    st.error("Falha ao carregar os dados. Verifique sua conexão com a internet.")
else:
    # Cálculo do IPCA acumulado 12 meses
    df_ipca['ipca_acumulado_12m'] = df_ipca['valor'].rolling(window=12).sum()

    # Mesclando dados de IPCA e Selic
    df_merged = pd.merge(df_ipca, df_selic, on='data', suffixes=('_ipca', '_selic'))

    # Input de seleção de período
    period_dict = {
        "1 ano": 12,
        "2 anos": 24,
        "5 anos": 60,
        "10 anos": 120,
        "Base Total": None
    }
    selected_period = st.selectbox("Selecione o período:", list(period_dict.keys()))

    # Filtrando os dados com base no período
    if period_dict[selected_period]:
        df_filtered = df_merged[df_merged['data'] >= df_merged['data'].max() - pd.DateOffset(months=period_dict[selected_period])]
    else:
        df_filtered = df_merged

    # Obtendo o último ponto para exibir a tag apenas no valor final
    last_ipca_point = df_filtered.iloc[-1]
    last_selic_point = df_filtered.iloc[-1]

    # Criando o gráfico
    fig = go.Figure()

    # Linha de IPCA acumulado 12 meses com tag de último valor
    fig.add_trace(go.Scatter(x=df_filtered['data'], y=df_filtered['ipca_acumulado_12m'],
                             mode='lines+markers', name='IPCA Acumulado 12 Meses',
                             line=dict(color=brand_colors['VERDE_DETALHES'], width=2)))

    fig.add_trace(go.Scatter(x=[last_ipca_point['data']], y=[last_ipca_point['ipca_acumulado_12m']],
                             mode='text',
                             text=[f"{last_ipca_point['ipca_acumulado_12m']:.2f}%"],
                             textposition="top right",
                             name='IPCA Tag',
                             showlegend=False))

    # Linha de Taxa Selic com tag de último valor
    fig.add_trace(go.Scatter(x=df_filtered['data'], y=df_filtered['valor_selic'],
                             mode='lines+markers', name='Taxa Selic',
                             line=dict(color=brand_colors['VERDE_PRINCIPAL'], width=2)))

    fig.add_trace(go.Scatter(x=[last_selic_point['data']], y=[last_selic_point['valor_selic']],
                             mode='text',
                             text=[f"{last_selic_point['valor_selic']:.2f}%"],
                             textposition="top right",
                             name='Selic Tag',
                             showlegend=False))

    # Layout do gráfico
    fig.update_layout(
        title="",
        xaxis=dict(tickformat="%Y-%m", nticks=20),
        yaxis=dict(title="%", side="right", showgrid=False, zeroline=False),
        template="plotly_white",
        font=dict(family="Arial", size=12, color=brand_colors['VERDE_DETALHES']),
        legend=dict(x=0.01, y=0.99),
    )

    # Adicionando a logo ao gráfico
    fig.add_layout_image(
        dict(
            source=f'data:image/png;base64,{img_str}',
            xref="paper", yref="paper",
            x=0.5, y=1.1,
            sizex=0.4, sizey=0.4,
            xanchor="center", yanchor="middle",
            opacity=0.7,
            layer="below"
        )
    )

    # Exibir o gráfico no Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Rodapé
st.markdown("---")
st.markdown("**Desenvolvido por Kauan Nunes - Cambirela Educa**")
