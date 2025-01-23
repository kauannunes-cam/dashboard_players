##################################################################################
#############################  BIBLIOTECAS  ######################################
##################################################################################

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import datetime, timedelta
import base64
from io import BytesIO
from PIL import Image
import os
import numpy as np


# Configurações de cores da marca
brand_colors = {
    "VERDE_TEXTO": "#1F4741",
    "VERDE_PRINCIPAL": "#2B6960",
    "VERDE_DETALHES": "#49E2B1",
    "CREME": "#FFFCF5",
    "CINZA": "#76807D",
    "PRETO": "#1B1B1B"
}


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


st.title("Análise de Posições do Estrangeiro em Ibov Spot e Índice Futuro")
##################################################################################
#############################  BANCO DE DADOS  ###################################
##################################################################################

@st.cache_data
def carregar_dados_cot():
    xls = pd.ExcelFile("History Cot.xlsx")
    ativos = {}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=2)
        df.columns.values[0] = 'Data'
        df.columns.values[1] = 'Preço'
        df.set_index('Data', inplace=True)
        df.sort_index(inplace=True, ascending=True)
        ativos[sheet_name] = df
    return ativos



# Carrega os dados
ativos = carregar_dados_cot()

ibovespa_cotacao = ativos['Ibovespa']
dolar_cotacao = ativos['UC1']

ibovespa_cotacao_dolar = (ibovespa_cotacao['Preço'] / dolar_cotacao['Preço']).dropna()


@st.cache_data
def carregar_dados_fluxo():
    xls = pd.ExcelFile("Fluxo estrangeiro b3.xlsx")
    fluxo = {}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=2)
        df.columns.values[0] = 'Data'
        df.columns.values[1] = 'Fluxo Estrangeiro'
        df['Data'] = pd.to_datetime(df['Data'], format="%d/%m/%Y")
        df.set_index('Data', inplace=True)
        df.sort_index(inplace=True, ascending=True)
        fluxo[sheet_name] = df
    return fluxo



# Carrega os dados
fluxo = carregar_dados_fluxo()


fluxo_ibovespa = fluxo['Ibov Spot']
fluxo_indfut = fluxo['Ind Fut']
fluxo_winfut = fluxo['Win Fut']


fluxo_indice_futuro = fluxo_indfut.add(fluxo_winfut, fill_value=0)

# Alinha os índices de todos os dataframes e preenche valores ausentes com 0
all_dates = fluxo_ibovespa.index.union(fluxo_indice_futuro.index).union(ibovespa_cotacao.index)
fluxo_ibovespa = fluxo_ibovespa.reindex(all_dates, fill_value=0)
fluxo_indices_futuro = fluxo_indice_futuro.reindex(all_dates, fill_value=0)
ibovespa_cotacao = ibovespa_cotacao.reindex(all_dates, fill_value=0)


# Cria o dataframe final com as 4 colunas
final_df = pd.DataFrame({
    'Fluxo IBOV Spot': fluxo_ibovespa['Fluxo Estrangeiro'],
    'Fluxo WINFUT + INDFUT': fluxo_indice_futuro['Fluxo Estrangeiro'],
    'Ibovespa': ibovespa_cotacao['Preço']
}, index=all_dates)


periodos = {
    "60 dias": 60,
    "180 dias": 180,
    "1 ano": 365,
    "5 anos": 1825,
    "Base total": None
}

# Seleciona o período desejado
periodo_escolhido = st.selectbox("Período", list(periodos.keys()))

if periodos[periodo_escolhido]:
    data_inicio = pd.Timestamp.today() - pd.Timedelta(days=periodos[periodo_escolhido])
    ultimas_datas = fluxo_ibovespa.loc[fluxo_ibovespa.index >= data_inicio].index
else:
    ultimas_datas = fluxo_ibovespa.index

# Cria o dataframe unificado
fluxo_ibov_dolarizado = pd.DataFrame({
    'Fluxo Estrangeiro': fluxo_ibovespa.reindex(ultimas_datas, fill_value=0)['Fluxo Estrangeiro'],
    'Ibovespa em Dólar': ibovespa_cotacao_dolar.reindex(ultimas_datas, fill_value=0)
})


fluxo_ibov_dolarizado = fluxo_ibov_dolarizado.sort_index(ascending=False)

df_mensal = pd.DataFrame()

# Calcula o acumulado mensal
df_mensal['Fluxo Total'] = final_df['Fluxo IBOV Spot'] + final_df['Fluxo WINFUT + INDFUT']
df_mensal['Mes'] = final_df.index.to_period('M')

# Agrupa por mês e soma
acumulado_mensal = df_mensal.groupby('Mes')['Fluxo Total'].sum().reset_index()


# Configurações de período mensal
periodo_mensal = {
    "24 meses": 24,
    "60 meses": 60,
    "120 meses": 120,
    "Base total": None
}


df_anual = pd.DataFrame()


df_anual['Fluxo Total'] = final_df['Fluxo IBOV Spot'] + final_df['Fluxo WINFUT + INDFUT']
df_anual['Ano'] = final_df.index.to_period('A')

# Agrupa por mês e soma
acumulado_anual = df_anual.groupby('Ano')['Fluxo Total'].sum().reset_index()




##################################################################################
#################################  GRAFICOS  #####################################
##################################################################################
# Cria o gráfico
gr1 = go.Figure()

# Adiciona barras para o Fluxo Estrangeiro IBOV Spot
gr1.add_trace(go.Bar(
    x=all_dates,
    y=fluxo_ibov_dolarizado['Fluxo Estrangeiro'],
    name='Fluxo IBOV Spot',
    marker_color=[
        brand_colors["VERDE_PRINCIPAL"] if v < 0 else brand_colors["VERDE_DETALHES"]
        for v in fluxo_ibov_dolarizado['Fluxo Estrangeiro']
    ]
))

# Adiciona linha para a Cotação IBOVESPA em dólar no eixo secundário
gr1.add_trace(go.Scatter(
    x=all_dates,
    y=fluxo_ibov_dolarizado['Ibovespa em Dólar'],
    name='Cotação IBOVESPA (USD)',
    mode='lines',
    line=dict(color=brand_colors["CREME"]),
    yaxis="y2"
))

# Configura layout do gráfico
gr1.update_layout(
    title="Fluxo Estrangeiro IBOV Spot e Cotação IBOVESPA em Dólar",
    xaxis_title="",
    yaxis=dict(
        title="Fluxo Estrangeiro (IBOV Spot)",
        titlefont=dict(color=brand_colors["VERDE_DETALHES"]),
        tickfont=dict(color=brand_colors["VERDE_DETALHES"]),
        showgrid=False
    ),
    yaxis2=dict(
        title="Cotação IBOVESPA (USD)",
        titlefont=dict(color=brand_colors["CREME"]),
        tickfont=dict(color=brand_colors["CREME"]),
        overlaying="y",
        side="right",
        showgrid=False
    ),
    legend=dict(orientation="h", y=1.05),
    plot_bgcolor="rgba(0,0,0,0)"
)

gr1.add_layout_image(
    dict(
        source=f'data:image/png;base64,{img_str}',
        xref="paper", yref="paper",
        x=0.03, y=0.8,
        sizex=0.2, sizey=0.2,
        xanchor="center", yanchor="middle",
        opacity=0.7,
        layer="below"
    )
)

#________________________________________________________________________________


# Cria o gráfico gr2
gr3 = go.Figure()

# Adiciona barras com o acumulado mensal
gr3.add_trace(go.Bar(
x=acumulado_anual['Ano'].dt.strftime('%Y-%m'),
y=acumulado_anual['Fluxo Total'],
name='Acumulado Anual',
marker_color=[
    brand_colors["VERDE_PRINCIPAL"] if v < 0 else brand_colors["VERDE_DETALHES"]
    for v in acumulado_anual['Fluxo Total']
]
))

# Configura layout do gráfico
gr3.update_layout(
title="Acumulado Anual (Fluxo IBOV Spot + WINFUT + INDFUT)",
yaxis_title="Fluxo Acumulado",
xaxis=dict(tickangle=-45),
yaxis=dict(showgrid=False),  # Corrigido de yaxix para yaxis
plot_bgcolor="rgba(0,0,0,0)"
)

gr3.add_layout_image(
dict(
    source=f'data:image/png;base64,{img_str}',
    xref="paper", yref="paper",
    x=0.03, y=0.8,
    sizex=0.2, sizey=0.2,
    xanchor="center", yanchor="middle",
    opacity=0.7,
    layer="below"
)
)    
        
##################################################################################
##################################  LAYOUT  ######################################
##################################################################################
# Título principal


col1, col2 = st.columns([16, 6])

with col1:
        st.plotly_chart(gr1, use_container_width=True)
        st.markdown("---")       
        col3, col4 = st.columns([5, 5])
        with col3:
            periodo_escolhido_mensal = st.selectbox("Período Mensal", list(periodo_mensal.keys()))
            if periodo_mensal[periodo_escolhido_mensal]:
                data_inicio_mensal = pd.Timestamp.today().to_period('M') - periodo_mensal[periodo_escolhido_mensal]
                acumulado_mensal = df_mensal[df_mensal['Mes'] >= data_inicio_mensal].groupby('Mes')['Fluxo Total'].sum().reset_index()
            else:
                acumulado_mensal = final_df.groupby('Mes')['Fluxo Total'].sum().reset_index()
                
            # Cria o gráfico gr2
            gr2 = go.Figure()

            # Adiciona barras com o acumulado mensal
            gr2.add_trace(go.Bar(
                x=acumulado_mensal['Mes'].dt.strftime('%Y-%m'),
                y=acumulado_mensal['Fluxo Total'],
                name='Acumulado Mensal',
                marker_color=[
                    brand_colors["VERDE_PRINCIPAL"] if v < 0 else brand_colors["VERDE_DETALHES"]
                    for v in acumulado_mensal['Fluxo Total']
                ]
            ))

            # Configura layout do gráfico
            gr2.update_layout(
                title="Acumulado Mensal (Fluxo IBOV Spot + WINFUT + INDFUT)",
                yaxis_title="Fluxo Acumulado",
                xaxis=dict(tickangle=-45),
                yaxis=dict(showgrid=False),  # Corrigido de yaxix para yaxis
                plot_bgcolor="rgba(0,0,0,0)"
            )

            gr2.add_layout_image(
                dict(
                    source=f'data:image/png;base64,{img_str}',
                    xref="paper", yref="paper",
                    x=0.03, y=0.8,
                    sizex=0.2, sizey=0.2,
                    xanchor="center", yanchor="middle",
                    opacity=0.7,
                    layer="below"
                )
            )                
            st.plotly_chart(gr2, use_container_width=True)
            st.markdown("---")  
        
        
        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.plotly_chart(gr3, use_container_width=True)
            st.markdown("---")  
            
        # Configuração para o gráfico gr4
        periodos = {
            "60 dias": 60,
            "180 dias": 180,
            "1 ano": 365,
            "5 anos": 1825,
            "Base total": None
        }

        periodo_escolhido_gr4 = st.selectbox("Períodos", list(periodos.keys()))

        if periodos[periodo_escolhido_gr4]:
            data_inicio_gr4 = pd.Timestamp.today() - pd.Timedelta(days=periodos[periodo_escolhido_gr4])
            df_gr4 = final_df[final_df.index >= data_inicio_gr4]
        else:
            df_gr4 = final_df

        # Cria o gráfico gr4
        gr4 = go.Figure()

        # Adiciona barras para IBOV Spot
        gr4.add_trace(go.Bar(
            x=df_gr4.index,
            y=df_gr4['Fluxo IBOV Spot'],
            name='IBOV Spot',
            marker_color=brand_colors["VERDE_PRINCIPAL"]
        ))

        # Adiciona barras para fluxo_indice_futuro
        gr4.add_trace(go.Bar(
            x=df_gr4.index,
            y=df_gr4['Fluxo WINFUT + INDFUT'],
            name='Fluxo WINFUT + INDFUT',
            marker_color=brand_colors["VERDE_DETALHES"]
        ))

        # Configura layout do gráfico
        gr4.update_layout(
            title="Distribuição Diária: IBOV Spot e Fluxo Índice Futuro",
            barmode='stack',
            xaxis_title="Data",
            yaxis_title="Fluxo Diário",
            yaxis=dict(showgrid=False),
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.05)
        )
        gr4.add_layout_image(
            dict(
                source=f'data:image/png;base64,{img_str}',
                xref="paper", yref="paper",
                x=0.03, y=0.8,
                sizex=0.2, sizey=0.2,
                xanchor="center", yanchor="middle",
                opacity=0.7,
                layer="below"
            )
        ) 

        # Exibe o gráfico gr4
        st.plotly_chart(gr4, use_container_width=True)
        
with col2:
        st.table(final_df.tail(24))
        
# Rodapé
st.markdown("---")
st.markdown("**Desenvolvido por Kauan Nunes - Cambirela Educa**")
