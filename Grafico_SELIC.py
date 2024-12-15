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
st.set_page_config(page_title="Taxa Selic - Gráfico", layout="wide")
st.title("Taxa de Juros Selic por Presidente")

# Função para obter os dados da Taxa Selic
def get_selic_data():
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1178/dados?formato=json"  # Selic acumulada
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
        df['valor'] = pd.to_numeric(df['valor'])
        return df
    else:
        st.error("Erro ao obter os dados da Taxa Selic")
        return pd.DataFrame()

# Input do usuário: seleção do período
period_dict = {
    "1 ano": 12,
    "5 anos": 60,
    "10 anos": 120,
    "20 anos": 240,
    "30 anos": 360,
    "Base Total": None
}

# Checkbox para visualizar faixas dos presidentes


selected_period = st.selectbox("Selecione o período:", list(period_dict.keys()), index=1)
show_presidents = st.checkbox("Visualizar Presidentes", value=True)

# Obter os dados reais
df_selic = get_selic_data()

df_selic["valor"] = df_selic["valor"] + 0.10

if not df_selic.empty:
    # Calcular a Selic acumulada nos últimos 12 meses
    df_selic['selic_acumulada'] = df_selic['valor'].rolling(window=12, min_periods=1).sum()

    # Calcular a diferença entre as taxas
    df_selic['diferenca'] = df_selic['valor'].diff().fillna(0)
    df_diferenca = df_selic[df_selic['diferenca'] != 0]

    # Filtrar os dados com base no input do usuário
    df_selic = df_selic.sort_values(by='data')
    if period_dict[selected_period]:
        df_selic = df_selic[df_selic['data'] >= df_selic['data'].max() - pd.DateOffset(months=period_dict[selected_period])]
        df_diferenca = df_diferenca[df_diferenca['data'] >= df_diferenca['data'].max() - pd.DateOffset(months=period_dict[selected_period])]

    # Criando o gráfico com Plotly
    fig = go.Figure()

    # Gráfico de barras para a diferença entre as taxas
    fig.add_trace(go.Bar(x=df_diferenca['data'], y=df_diferenca['diferenca'].round(2),
                         name='Diferença entre Taxas',
                         marker=dict(color=brand_colors['VERDE_PRINCIPAL'], opacity=0.3),
                         yaxis='y'))


    # Gráfico de linha para a Selic acumulada 12 meses
    fig.add_trace(go.Scatter(x=df_selic['data'], y=df_selic['valor'].round(2),
                             mode='lines', name='Taxa Selic Mensal',
                             line=dict(color=brand_colors['VERDE_DETALHES'], width=2),
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
    data_min = df_selic['data'].min()
    data_max = df_selic['data'].max()

    if show_presidents:
        for presidente in presidentes:
            inicio = pd.to_datetime(presidente["inicio"])
            fim = pd.to_datetime(presidente["fim"])

            # Verificar se o mandato intercepta o período filtrado
            if inicio <= data_max and fim >= data_min:
                # Ajustar os limites para ficar dentro do período filtrado
                x0 = max(inicio, data_min)
                x1 = min(fim, data_max)

                # Adicionar a faixa do presidente ao gráfico
                fig.add_vrect(
                    x0=x0, x1=x1,
                    fillcolor=presidente["cor"], opacity=0.2, line_width=0,
                    annotation_text=presidente["nome"],
                    annotation_position="top right",
                    annotation_font=dict(size=10, color=brand_colors["CREME"]),
                    layer="below"
                )


    # Layout do gráfico
    fig.update_layout(
        xaxis=dict(
        tickformat="%d-%m-%Y",
        nticks=28,
        tickangle=-45),
        yaxis=dict(title="Reunião COPOM (%)", side="left", showgrid=False, tickformat=".2f"),
        yaxis2=dict(title="Taxa SELIC (%)", overlaying="y", side="right", showgrid=False, tickformat=".2f"),
        barmode='overlay',
        template="plotly_white",
        font=dict(family="Arial", size=12, color=brand_colors['PRETO']),
        legend=dict(x=0.01, y=0.93, traceorder="normal"),
    )

    # Adicionar a logo ao gráfico
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
else:
    st.error("Falha ao carregar os dados. Verifique sua conexão com a internet ou a fonte de dados.")

# Rodapé
st.markdown("---")
st.markdown("**Desenvolvido por Kauan Nunes - Cambirela Educa**")
