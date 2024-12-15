import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import base64


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
st.set_page_config(page_title="Preços - IPCA e Meta", layout="wide")
st.title("IPCA e Meta de Inflação por Presidente")

# Função para obter os dados reais do Banco Central
def get_ipca_data():
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
        df['valor'] = pd.to_numeric(df['valor'])
        return df
    else:
        st.error("Erro ao obter os dados do IPCA do Banco Central")
        return pd.DataFrame()

# Função para obter as metas de inflação do Boletim Focus
def get_focus_meta():
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.13522/dados?formato=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
        df['valor'] = pd.to_numeric(df['valor'])
        return df
    else:
        st.error("Erro ao obter os dados do Boletim Focus")
        return pd.DataFrame()

# Obter os dados reais do IPCA acumulado 12 meses (ou similar)
def get_ipca_acumulado():
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4449/dados?formato=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
        df['valor'] = pd.to_numeric(df['valor'])
        return df
    else:
        st.error("Erro ao obter os dados do IPCA Acumulado")
        return pd.DataFrame()

# Input do Usuário: seleção do período
period_dict = {
    "1 ano": 12,
    "5 anos": 60,
    "10 anos": 120,
    "20 anos": 240,
    "30 anos": 360,
    "Base Total": None
}

selected_period = st.selectbox("Selecione o período:", list(period_dict.keys()))


# Checkbox para visualizar faixas dos presidentes
show_presidents = st.checkbox("Visualizar Presidentes", value=True)


# Obter os dados reais
df_ipca = get_ipca_data()
df_focus = get_focus_meta()
df_ipca['ipca_acumulado'] = df_ipca['valor'].rolling(window=12, min_periods=1).sum()
df_ipca['valor_meta'] = 3.00



# Adicionar banda da meta de inflação
df_focus['meta_superior'] = df_ipca['valor_meta'][0] + 1.5
df_focus['meta_inferior'] = df_ipca['valor_meta'][0] - 1.5

# Filtrar os dados com base no input do usuário
df = df_ipca.sort_values(by='data')

if period_dict[selected_period]:
    df = df[df['data'] >= df['data'].max() - pd.DateOffset(months=period_dict[selected_period])]
    df_ipca = df_ipca[df_ipca['data'] >= df['data'].max() - pd.DateOffset(months=period_dict[selected_period])]


df_ipca_6d = df_ipca.tail(6)
df_focus_6d = df_focus.tail(6)

# Criando o gráfico com Plotly
fig = go.Figure()

# Gráfico de barras do IPCA ocorrido
fig.add_trace(go.Bar(x=df['data'], y=df['valor'],
                     name='IPCA Mensal',
                     marker=dict(color=brand_colors['VERDE_PRINCIPAL'], opacity=0.3),
                     yaxis='y'))

# Gráfico de linha para IPCA acumulado 12 meses
fig.add_trace(go.Scatter(x=df_ipca['data'], y=df_ipca['ipca_acumulado'],
                         mode='lines', name='IPCA Acuml',
                         line=dict(color=brand_colors['VERDE_DETALHES'], width=2),
                         yaxis='y2'))

# Linha da Meta para a inflação
fig.add_trace(go.Scatter(x=df_ipca_6d['data'], y=df_ipca_6d['valor_meta'],
                         mode='lines', name='Meta IPCA',
                         line=dict(color=brand_colors['CINZA'], width=2),
                         yaxis='y2'))

# Linha para banda superior da Meta
fig.add_trace(go.Scatter(x=df_focus_6d['data'], y=df_focus_6d['meta_superior'],
                         mode='lines', name='Banda Superior',
                         line=dict(color=brand_colors['CINZA'], dash='dash'),
                         yaxis='y2'))

# Linha para banda inferior da Meta
fig.add_trace(go.Scatter(x=df_focus_6d['data'], y=df_focus_6d['meta_inferior'],
                         mode='lines', name='Banda Inferior',
                         line=dict(color=brand_colors['CINZA'], dash='dash'),
                         yaxis='y2'))

# Lista de presidentes e seus períodos de mandato
presidentes = [
    {"nome": "José Sarney", "inicio": "1985-03-15", "fim": "1990-03-15", "cor": "rgba(105, 159, 255, 0.2)"},
    {"nome": "Fernando Collor", "inicio": "1990-03-15", "fim": "1992-12-29", "cor": "rgba(255, 179, 102, 0.2)"},
    {"nome": "Itamar Franco", "inicio": "1992-12-29", "fim": "1995-01-01", "cor": "rgba(102, 255, 179, 0.2)"},
    {"nome": "Fernando Henrique Cardoso", "inicio": "1995-01-01", "fim": "2003-01-01", "cor": "rgba(255, 102, 102, 0.2)"},
    {"nome": "Luiz Inácio Lula da Silva", "inicio": "2003-01-01", "fim": "2011-01-01", "cor": "rgba(102, 102, 255, 0.2)"},
    {"nome": "Dilma Rousseff", "inicio": "2011-01-01", "fim": "2016-08-31", "cor": "rgba(255, 102, 204, 0.2)"},
    {"nome": "Michel Temer", "inicio": "2016-08-31", "fim": "2019-01-01", "cor": "rgba(102, 204, 255, 0.2)"},
    {"nome": "Jair Bolsonaro", "inicio": "2019-01-01", "fim": "2023-01-01", "cor": "rgba(255, 153, 51, 0.2)"},
    {"nome": "Luiz Inácio Lula da Silva", "inicio": "2023-01-01", "fim": "2026-12-31", "cor": "rgba(102, 102, 255, 0.2)"}
]

# Filtrar presidentes com base no período selecionado
data_min = df_ipca['data'].min()
data_max = df_ipca['data'].max()

if show_presidents:
    for presidente in presidentes:
        inicio = pd.to_datetime(presidente["inicio"])
        fim = pd.to_datetime(presidente["fim"])
        if inicio <= df['data'].max() and fim >= df['data'].min():
            x0 = max(inicio, df['data'].min())
            x1 = min(fim, df['data'].max())
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor="rgba(105, 159, 255, 0.1)", opacity=0.2, line_width=0,
                annotation_text=presidente["nome"],
                annotation_position="top left",
                annotation_font=dict(size=10, color=brand_colors["CREME"]),
                layer="below"
            )

# Layout do gráfico
fig.update_layout(
    title="",
    xaxis=dict(
    tickformat="%d-%m-%Y",
    nticks=28,
    tickangle=-45),
    yaxis=dict(title="IPCA Mensal (%)", side="left", showgrid=False, tickformat=".2f"),
    yaxis2=dict(title="IPCA Acumulado / Meta (%)", overlaying="y", side="right", showgrid=False, tickformat=".2f"),
    barmode='overlay',
    template="plotly_white",
    font=dict(family="Arial", size=12, color=brand_colors['PRETO']),
    legend=dict(x=0.01, y=0.93, traceorder="normal"),
)




fig.add_layout_image(
    dict(
        source=f'data:image/png;base64,{img_str}',
        xref="paper", yref="paper",
        x=0.5, y=1.15,
        sizex=0.4, sizey=0.4,
        xanchor="center", yanchor="middle",
        opacity=0.6,
        layer="below"
    )
)
# Exibir o gráfico no Streamlit
st.plotly_chart(fig, use_container_width=True)

# Rodapé
st.markdown("---")
st.markdown("**Desenvolvido por Kauan Nunes - Cambirela Educa**")
