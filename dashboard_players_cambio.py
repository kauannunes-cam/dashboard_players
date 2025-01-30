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
cinco_dia = datetime.timedelta(days=5)
hoje = datetime.datetime.today()
data_final = hoje.strftime('%Y-%m-%d')
data_inicial = datetime.date(2024, 8, 1)  # Define o início em 2024
data_minima = datetime.date(2010, 1, 1)  # Define o início em 2024
data_menor_inicial = hoje - cinco_dia # Define o início em 2024


players = ['estrangeiro', 'flocal', 'bancos', 'pj', 'pf']
ativos = ['wdo', 'dol', 'ddi', 'swap']
valor_ativo = [10000, 50000, 50000, -50000]

# Configurações de cores da marca
brand_colors = {
    "VERDE_TEXTO": "#1F4741",
    "VERDE_PRINCIPAL": "#2B6960",
    "VERDE_DETALHES": "#49E2B1",
    "CREME": "#FFFCF5",
    "CINZA": "#969B91",
    "PRETO": "#1B1B1B"
}



# Definição das cores ajustadas para as barras
colors = [
    "#3498DB",  # DOLAR_CHEIO (Verde Detalhes)
    "#F1C40F",  # DOLAR_MINI (Verde Principal)
    "#E74C3C",  # DDI (Um tom contrastante, já em uso)
    "#8E44AD"   # SWAP (Amarelo, já em uso e harmoniza)
]




st.set_page_config(layout="wide")


# Configuração de estilo para centralizar os elementos
st.markdown("""
    <style>
        .centered {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .block-container {
            max-width: 100%;
            padding: 1rem 2rem;
            margin: 0 auto;
        }
        .custom-header {
            text-align: center;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

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
    combined_df['var_liq'] = combined_df['saldo'].diff()
    combined_df['var_%'] = combined_df['saldo'].pct_change() * 100  # Calcula a variação percentual diária
    globals()[f'df_{player}'] = combined_df

# Função para formatar números no estilo brasileiro
def formatar_moeda(valor):
    return f'R$ {valor:,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.')




# Configuração do título com filtro de data ao lado
col1, col2 = st.columns([2,2])
with col1:
    st.title("Análise de Posições dos Players")

with col2:
    # Criando colunas para organizar as datas e o checkbox
    date_col1, date_col2 = st.columns([2, 2])
    
    # Filtro de data inicial
    with date_col1:
        start_date = st.date_input("Data Inicial", value=data_inicial, min_value=data_minima, max_value=hoje)
        visualizar_tabela = st.checkbox("Visualizar Tabelas", key="Visualizar")

    # Filtro de data final e checkbox para definir "Hoje"
    with date_col2:
        end_date = st.date_input("Data Final", value=hoje, min_value=data_minima, max_value=hoje)
        # Checkbox para definir Data Final como Hoje, posicionado abaixo
        set_end_today = st.checkbox("Hoje", key="end_today")
        if set_end_today:
            end_date = hoje  # Redefine a data final para hoje se o checkbox estiver marcado

        
st.divider()


# Função para carregar a imagem e converter para base64
def load_image_as_base64(image_path):
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Caminho da logo (ajuste conforme necessário)
logo_path = "logo_transparente_cambirela.png"


# Carregar logo como base64 para usar em marca d'água
img_str = load_image_as_base64(logo_path)

# Função para plotar gráfico combinado de linhas (saldo e UC1) e barras (composição de contratos)
def plot_combined_chart(player_df, uc1_df, player_name):
    fig = go.Figure()


    player_df.index = pd.to_datetime(player_df.index)
    # Exibir apenas os últimos 60 dias no preview
    recent_df = player_df.loc[(player_df.index >= pd.to_datetime(start_date)) & (player_df.index <= pd.to_datetime(end_date))]

    # Gráfico de linhas de saldo e UC1
    fig.add_trace(go.Scatter(
        x=recent_df.index,
        y=recent_df['saldo'],
        mode='lines',
        name=f"{player_name} Saldo",
        line=dict(color="#76807d", shape='spline'),
        yaxis="y2",
        showlegend=True
    ))

    uc1_df_filtered = uc1_df.loc[uc1_df.index.isin(recent_df.index)]
    fig.add_trace(go.Scatter(
        x=uc1_df_filtered.index,
        y=uc1_df_filtered['Preço'],
        mode='lines',
        name='UC1',
        line=dict(color="#3DF2C1", shape='spline'),
        yaxis="y3",
        showlegend=True
    ))

    # Gráfico de barras de composição de contratos (somente ddi, swap, dolarmini, dolarcheio)
    columns_to_plot = ['ddi', 'swap', 'dolarmini', 'dolarcheio']
    for idx, column in enumerate(columns_to_plot):
        if column in recent_df.columns:
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
            tickfont=dict(color="#3DF2C1")
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
            x=0.5, y=1.15,
            sizex=0.2, sizey=0.2,
            xanchor="center", yanchor="middle",
            opacity=1,
            layer="below"
        )
    )
    return fig


def formatar_numero(valor): 
    return f'U$${valor:,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.')


def calcular_variacoes(df):
    # Calcula a variação percentual diária para cada ativo
    for ativo in ['dolarcheio', 'dolarmini', 'swap', 'ddi']:
        if ativo in df.columns:
            df[f'var_$_{ativo}'] = df[ativo].diff()
            df[f'var_%_{ativo}'] = df[ativo].pct_change() * 100
    # Calcula o saldo e variação percentual do saldo
    df['var_liq_saldo'] = df['saldo'].diff()
    df['var_%_saldo'] = df['saldo'].pct_change() * 100

    # Reorganiza as colunas conforme solicitado
    ordered_columns = []
    for ativo in ['dolarcheio', 'var_$_dolarcheio', 'var_%_dolarcheio',
                  'dolarmini', 'var_$_dolarmini', 'var_%_dolarmini',
                  'swap', 'var_$_swap', 'var_%_swap',
                  'ddi', 'var_$_ddi', 'var_%_ddi']:
        if ativo in df.columns:
            ordered_columns.append(ativo)
    ordered_columns.extend(['saldo', 'var_liq_saldo', 'var_%_saldo'])
    return df[ordered_columns]

# Aplica a função para cada tabela (df_player)
dfs = {'Estrangeiro': df_estrangeiro, 'Fundo Local': df_flocal, 'Bancos': df_bancos, 'Pessoas Jurídicas': df_pj, 'Pessoas Físicas': df_pf}
dfs_variados = {player: calcular_variacoes(df) for player, df in dfs.items()}



# Ajustar o DataFrame para incluir a data, ordenar, e formatar
dfs_formatados = {}
for player_name, df in dfs_variados.items():
    # Inclui a data como coluna e verifica o nome
    df = df.reset_index()

    # Identifica dinamicamente o nome da coluna de data
    date_column = df.columns[0]  # A primeira coluna será a data original (antes do reset_index)
    df[date_column] = pd.to_datetime(df[date_column]).dt.date  # Formata como apenas data

    # Ordena pela coluna de data
    df.sort_values(by=date_column, ascending=False, inplace=True)

    
    # Formatar apenas as colunas numéricas
    df_formatado = df.copy()
    for col in df.columns[1:]:  # Ignora a coluna de índice ao formatar
        if "var_%" in col:  # Verifica se a coluna é de variação percentual
            df_formatado[col] = df[col].apply(lambda x: f"{x:.2f}%")
        else:  # Formata normalmente as colunas numéricas
            df_formatado[col] = df[col].apply(formatar_numero)
    
    dfs_formatados[player_name] = df_formatado



for player_name, df in dfs_formatados.items():
    st.subheader(player_name)
    st.plotly_chart(plot_combined_chart(dfs_variados[player_name], uc1_df, player_name), use_container_width=True)
    st.divider()

    if visualizar_tabela:
        try:
            col1, col2 = st.columns([1, 3])  # Define proporção das colunas

            with col1:
                st.write("")
                st.markdown("### Selecione o período da tabela:")

            with col2:
                quantidade_dias = st.selectbox(
                    "",
                    options=[5, 10, 15, 30, 60, 90, 180, 360],  # Opções no dropdown
                    index=0,  # Valor padrão é o primeiro (5 dias)
                    key=f"{player_name}_dias"
                )
            # Remove a primeira coluna (geralmente o índice resetado)
            #df = df.drop(columns=df.columns[0])  # `columns` especifica o nome ou índice da coluna
            
            # Adiciona CSS para centralizar o conteúdo da tabela
            st.markdown(
                """
                <style>
                    .dataframe {
                        width: 80%; /* Define largura da tabela como 100% */
                    }
                    .dataframe tbody td {
                        text-align: center; /* Centraliza os valores dentro das células */
                        font-size: 12px; /* Define o tamanho da fonte para o corpo */
                    }
                    .dataframe thead th {
                        text-align: center; /* Centraliza os cabeçalhos */
                        font-size: 12px; /* Define o tamanho da fonte para os cabeçalhos */
                    }
                    .reportview-container .main .block-container {
                        max-width: 1600px;
                        padding: 1rem 2rem;
                        background-color: rgba(0, 0, 0, 0);
                    }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Renderiza a tabela como HTML com estilo centralizado
            st.write(
                df.head(quantidade_dias).to_html(index=False, justify='center', classes='dataframe'),  # Converte DataFrame para HTML
                unsafe_allow_html=True
            )
            st.divider()
        except Exception as e:
            st.error(f"Erro ao renderizar tabela para {player_name}: {e}")



# Rodapé
st.markdown("---")
st.markdown("**Desenvolvido por Kauan Nunes - Cambirela Educa**")

