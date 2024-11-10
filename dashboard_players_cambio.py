import streamlit as st
import base64
from io import BytesIO
from PIL import Image
import plotly.graph_objs as go
import matplotlib.ticker as ticker
import yfinance as yf
import plotly.subplots as sp
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import locale
import os

um_dia = datetime.timedelta(days=1)
hoje = datetime.datetime.today()
data_final = hoje.strftime('%Y-%m-%d')


players = ['estrangeiro', 'flocal', 'bancos', 'pj', 'pf']
ativos = ['wdo', 'dol', 'ddi', 'swap']
valor_ativo = [10000, 50000, 50000, -50000]

# Configurações de cores da marca
brand_colors = {
    "VERDE_TEXTO": "#1F4741",
    "VERDE_PRINCIPAL": "#2B6960",
    "VERDE_DETALHES": "#49E2B1",
    "CREME": "#FFFCF5",
    "CINZA": "#76807D",
    "PRETO": "#1B1B1B"
}
colors = [brand_colors["VERDE_DETALHES"], "#3498DB", "#F1C40F", "#E74C3C", "#8E44AD"]


# Caminho para os arquivos Excel
file_path_saldos = 'History dollar B3.xlsx'
file_path_uc1 = 'History Cot.xlsx'

# Carrega todas as planilhas de saldos em um dicionário de dataframes
excel_data_saldos = pd.read_excel(file_path_saldos, sheet_name=None, skiprows=2)

# Carrega o arquivo Excel para o UC1
xls = pd.ExcelFile(file_path_uc1)
ativos = {}
for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=2)
    df.columns.values[0] = 'Data'
    df.columns.values[1] = 'Preço'
    df_selected = df[['Data', 'Preço']]
    df_selected.set_index('Data', inplace=True)
    df_sorted = df_selected.sort_index()
    ativos[sheet_name] = df_sorted

# Seleciona o UC1
selected_assets = ['UC1']
ativos_selected = {key: ativos[key] for key in selected_assets if key in ativos}
uc1_df = ativos_selected['UC1']



# Caminho para o arquivo Excel
file_path = 'History dollar B3.xlsx'

# Carrega todas as planilhas em um dicionário de dataframes
excel_data = pd.read_excel(file_path, sheet_name=None, skiprows=2)

# Listas de players e ativos
players = ['estrangeiro', 'flocal', 'bancos', 'pj', 'pf']
ativos = ['dolarcheio', 'dolarmini', 'ddi', 'swap']
valor_ativo = {'dolarcheio': 1, 'dolarmini': 1, 'ddi': 1, 'swap': 1}

# Função para ajustar cada dataframe
def ajustar_dataframe(sheet_name, data):
    # Definir a segunda coluna como índice no formato de data
    data[data.columns[0]] = pd.to_datetime(data[data.columns[0]])
    data.set_index(data.columns[0], inplace=True)
    # Identificar o ativo a partir do nome da planilha
    for ativo in ativos:
        if ativo in sheet_name:
            data.columns = [ativo]  # Renomeia a coluna com o nome do ativo
            data[ativo] = data[ativo] * valor_ativo[ativo]  # Converter valores em U$$
            break
    return data

# Itera sobre cada planilha e ajusta o dataframe
all_dataframes = {}
for sheet_name, data in excel_data.items():
    dataframe_name = 'df_' + sheet_name.replace(' ', '_').replace('-', '_')
    adjusted_data = ajustar_dataframe(sheet_name, data)
    globals()[dataframe_name] = adjusted_data
    # Agrupa os dataframes ajustados por player
    player = sheet_name.split('_')[-1]
    if player not in all_dataframes:
        all_dataframes[player] = []
    all_dataframes[player].append(adjusted_data)

# Concatena os dataframes de cada player e calcula saldo e variação líquida diária
for player, dfs in all_dataframes.items():
    combined_df = pd.concat(dfs, axis=1)
    combined_df['saldo'] = combined_df.sum(axis=1)
    combined_df['var_liq_diaria'] = combined_df['saldo'].diff()
    globals()[f'df_{player}'] = combined_df




# Função para carregar a imagem e converter para base64
def load_image_as_base64(image_path):
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Caminho da logo (ajuste conforme necessário)
logo_path = "./logo_transparente.png"


# Carregar logo como base64 para usar em marca d'água
img_str = load_image_as_base64(logo_path)

# Função para plotar gráfico combinado de linhas (saldo e UC1) e barras (composição de contratos)
def plot_combined_chart(player_df, uc1_df, player_name):
    fig = go.Figure()

    # Exibir apenas os últimos 60 dias no preview
    recent_df = player_df[-60:]

    # Gráfico de linhas de saldo e UC1
    fig.add_trace(go.Scatter(
        x=recent_df.index,
        y=recent_df['saldo'],
        mode='lines',
        name=f"{player_name} Saldo",
        line=dict(color=brand_colors["CREME"], shape='spline'),
        yaxis="y2",
        showlegend=True
    ))

    uc1_df_filtered = uc1_df.loc[uc1_df.index.isin(recent_df.index)]
    fig.add_trace(go.Scatter(
        x=uc1_df_filtered.index,
        y=uc1_df_filtered['Preço'],
        mode='lines',
        name='UC1',
        line=dict(color=brand_colors["CINZA"], shape='spline'),
        yaxis="y3",
        showlegend=True
    ))

    # Gráfico de barras de composição de contratos
    for idx, column in enumerate(recent_df.columns[:-2]):  # Exclui saldo e variação
        fig.add_trace(go.Bar(
            x=recent_df.index,
            y=recent_df[column],
            name=column,
            marker=dict(color=colors[idx % len(colors)]),
            yaxis="y1",
            showlegend=True
        ))


    # Configurações de layout e estilo com coloração dos textos dos eixos Y
    fig.update_layout(
        title=f"Saldos dos Contratos dos {player_name} x Dólar",
        yaxis=dict(
            title='Contratos (U$$)',
            side='left',
            showgrid=False,
            titlefont=dict(color=brand_colors["CREME"]),  # Coloração para o eixo das barras
            tickfont=dict(color=brand_colors["CREME"])
        ),
        yaxis2=dict(
            title='Saldo de Contratos (U$$)',
            overlaying='y',
            side='right',
            showgrid=False,
            titlefont=dict(color=brand_colors["CREME"]),  # Coloração para o eixo do saldo
            tickfont=dict(color=brand_colors["CREME"])
        ),
        yaxis3=dict(
            overlaying='y',
            side='right',
            position=0.5,
            showgrid=True,
            titlefont=dict(color=brand_colors["CINZA"]),  # Coloração para o eixo UC1
            tickfont=dict(color=brand_colors["CINZA"])
        ),
        barmode='stack',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            font=dict(size=10),
            orientation="h",
            x=0.5,
            y=-0.3,
            xanchor="center",
            yanchor="top",
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor="White",
            borderwidth=1
        )
    )
    fig.add_layout_image(
        dict(
            source=f'data:image/png;base64,{img_str}',
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            sizex=0.9, sizey=0.9,
            xanchor="center", yanchor="middle",
            opacity=0.2,
            layer="below"
        )
    )
    return fig

# Configurações de layout do dashboard com Streamlit
st.set_page_config(layout="wide")

# Exibir título do dashboard
st.title("Análise de Posições dos Players")

# Primeira linha - Estrangeiros e Fundo Local
col1, col2 = st.columns(2)
with col1:
    st.subheader("Estrangeiros")
    st.plotly_chart(plot_combined_chart(df_estrangeiro, uc1_df, "Estrangeiro"), use_container_width=True)

with col2:
    st.subheader("Fundo Local")
    st.plotly_chart(plot_combined_chart(df_flocal, uc1_df, "Fundo Local"), use_container_width=True)

# Segunda linha - Bancos, Pessoas Jurídicas e Pessoas Físicas
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Bancos")
    st.plotly_chart(plot_combined_chart(df_bancos, uc1_df, "Bancos"), use_container_width=True)

with col2:
    st.subheader("Pessoas Jurídicas")
    st.plotly_chart(plot_combined_chart(df_pj, uc1_df, "Pessoas Jurídicas"), use_container_width=True)

with col3:
    st.subheader("Pessoas Físicas")
    st.plotly_chart(plot_combined_chart(df_pf, uc1_df, "Pessoas Físicas"), use_container_width=True)

# Ajuste de estilo para transparência do fundo e espaçamento
st.markdown(f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: 1600px;
            padding: 1rem 2rem;
            background-color: rgba(0, 0, 0, 0);
        }}
        .css-1lcbmhc {{
            margin-bottom: -10px;
        }}
    </style>
""", unsafe_allow_html=True)

# Primeira linha - Estrangeiros e Fundo Local
col1, col2, col3, col4, col5 = st.columns(5)

