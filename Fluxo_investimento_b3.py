import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Cores da Marca
brand_colors = {
    "VERDE_PRINCIPAL": "#2B6960",
    "VERDE_DETALHES": "#49E2B1",
    "CINZA": "#76807D",
    "PRETO": "#1B1B1B"
}

# Função para coletar os dados do site
def coletar_dados_fluxo():
    url = "https://www.dadosdemercado.com.br/fluxo"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    tabela = soup.find('table')
    linhas = tabela.find_all('tr')

    dados = []
    for linha in linhas[1:]:  # Pula o cabeçalho
        colunas = linha.find_all('td')
        dados.append({
            'Data': colunas[0].text.strip(),
            'Estrangeiro': int(float(colunas[1].text.replace('mi', '').replace(',', '').strip()) * 10_000),
            'Institucional': int(float(colunas[2].text.replace('mi', '').replace(',', '').strip()) * 10_000),
            'Pessoa Física': int(float(colunas[3].text.replace('mi', '').replace(',', '').strip()) * 10_000),
            'Inst. Financeira': int(float(colunas[4].text.replace('mi', '').replace(',', '').strip()) * 10_000),
            'Outros': int(float(colunas[5].text.replace('mi', '').replace(',', '').strip()) * 10_000),
        })

    df = pd.DataFrame(dados)
    df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')
    return df

import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Cores da Marca
brand_colors = {
    "VERDE_PRINCIPAL": "#2B6960",
    "VERDE_DETALHES": "#49E2B1",
    "CINZA": "#76807D",
    "PRETO": "#1B1B1B"
}

# Função para coletar os dados do site
def coletar_dados_fluxo():
    url = "https://www.dadosdemercado.com.br/fluxo"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    tabela = soup.find('table')
    linhas = tabela.find_all('tr')

    dados = []
    for linha in linhas[1:]:  # Pula o cabeçalho
        colunas = linha.find_all('td')
        dados.append({
            'Data': colunas[0].text.strip(),
            'Estrangeiro': int(float(colunas[1].text.replace('mi', '').replace(',', '').strip()) * 10_000),
            'Institucional': int(float(colunas[2].text.replace('mi', '').replace(',', '').strip()) * 10_000),
            'Pessoa Física': int(float(colunas[3].text.replace('mi', '').replace(',', '').strip()) * 10_000),
            'Inst. Financeira': int(float(colunas[4].text.replace('mi', '').replace(',', '').strip()) * 10_000),
            'Outros': int(float(colunas[5].text.replace('mi', '').replace(',', '').strip()) * 10_000),
        })

    df = pd.DataFrame(dados)
    df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')
    return df

# Função para calcular acumulados por período
def calcular_acumulados(df, player):
    df['Mes'] = df['Data'].dt.to_period('M')
    df['Trimestre'] = df['Data'].dt.to_period('Q')
    df['Semestre'] = df['Data'].dt.to_period('2Q')
    df['Ano'] = df['Data'].dt.to_period('Y')
    df['PrimeiroDiaMes'] = df['Data'].dt.is_month_start

    df['Acumulado_Mes'] = df.groupby('Mes')[player].cumsum()
    df['Acumulado_Trimestre'] = df.groupby('Trimestre')[player].cumsum()
    df['Acumulado_Semestre'] = df.groupby('Semestre')[player].cumsum()
    df['Acumulado_Ano'] = df.groupby('Ano')[player].cumsum()

    # Zerar o acumulado no início do mês
    df.loc[df['PrimeiroDiaMes'], 'Acumulado_Mes'] = 0
    return df

# Função para criar o gráfico Plotly
def criar_grafico(df, player):
    fig = go.Figure()

    # Gráfico de barras
    fig.add_trace(go.Bar(
        x=df['Data'],
        y=df[player],
        name=f"{player} Diário",
        marker_color=brand_colors['VERDE_DETALHES']
    ))

    # Linha de acumulado
    fig.add_trace(go.Scatter(
        x=df['Data'],
        y=df['Acumulado_Mes'],
        mode='lines',
        name="Acumulado Mensal",
        line=dict(color=brand_colors['VERDE_PRINCIPAL'], width=2, dash='dot'),
        yaxis="y2"
    ))
    
    #Linhas verticais para cada mês
    for mes in df['Mes'].unique():
        mes_inicio = df[df['Mes'] == mes]['Data'].min()
        fig.add_vline(
            x=mes_inicio,
            line=dict(color=brand_colors['CINZA'], width=1, dash='dash'),
            opacity=0.3
        )

    fig.update_layout(
        title=f"Fluxo de Investimentos ({player}) com Acumulado Mensal",
        xaxis=dict(title="Data", showgrid=False),
        yaxis=dict(title="Total (Milhões)", side="left"),
        yaxis2=dict(title="Acumulado Mensal", overlaying="y", side="right", showgrid=False),
        template="simple_white",
        height=600,
        width=1200,
        barmode='overlay'
    )
    return fig

# Função principal do Streamlit
def main():
    st.set_page_config(layout="wide")
    st.title("Fluxo de Investimentos na B3")

    # Coletar dados
    df = coletar_dados_fluxo()
    st.write("Dados coletados até a data:", df['Data'].max().strftime('%d/%m/%Y'))

    # Selecionar participante
    participantes = ['Estrangeiro', 'Institucional', 'Pessoa Física', 'Inst. Financeira', 'Outros']
    player = st.selectbox("Selecione o participante:", participantes)

    # Calcular acumulados
    df = calcular_acumulados(df, player)

    # Criar gráfico
    fig = criar_grafico(df, player)
    st.plotly_chart(fig, use_container_width=True)

    # Exibir DataFrame
    #st.dataframe(df[['Data', player, 'Acumulado_Mes', 'Acumulado_Trimestre', 'Acumulado_Semestre', 'Acumulado_Ano']]
                 #.rename(columns={player: 'Total (Milhões)'}), use_container_width=True)

if __name__ == "__main__":
    main()


# Rodapé
st.markdown("---")
st.markdown("**Desenvolvido por Kauan Nunes - Cambirela Educa**")
