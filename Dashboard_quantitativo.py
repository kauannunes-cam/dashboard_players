import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import datetime
import base64
from io import BytesIO
from PIL import Image
import os
import numpy as np
from datetime import datetime, timedelta

nivel_confianca = 0.975
z_95 = 1.96
z_99 = 2.33

def formatar_moeda(valor):
    valor_str = f"{valor:,.2f}"  # Formata com separador de milhar e duas casas decimais
    return valor_str.replace(",", "X").replace(".", ",").replace("X", ".") 


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
logo_path = "logo_transparente.png"

file_path = 'History Cot.xlsx'

# Carregar logo como base64 para usar em marca d'água
img_str = load_image_as_base64(logo_path)

#############################  BANCO DE DADOS  ###################################
st.set_page_config(layout="wide")

@st.cache_data
def carregar_dados():
    xls = pd.ExcelFile(file_path)
    ativos = {}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=2)
        df.columns.values[0] = 'Data'
        df.columns.values[1] = 'Preço'
        df.set_index('Data', inplace=True)
        df.sort_index(inplace=True)
        ativos[sheet_name] = df
    return ativos

# Carrega os dados
ativos = carregar_dados()

# Lista de ativos
selected_assets = ['BBDXY', 'Bitcoin', 'CL1', 'CO1', 'DM1', 'ES1', 'NQ1', 'ODF26', 'ODF27', 'ODN26', 
                   'T-02', 'T-10', 'UC1', 'UC2', 'USDMXN', 'VIX', 'XAU', 'XB1', 'BCOMGR', 'CDS Brazil']
ativos_selected = {key: ativos[key] for key in selected_assets if key in ativos}



st.title("Análise Quantitativa")

# Função para calcular a variação percentual e sequências com setas coloridas
def processar_dados_com_setas_coloridas(df):
    df = df.copy()
    df['Variação (%)'] = df['Preço'].pct_change() * 100
    df['Trend'] = df['Preço'].diff().apply(lambda x: '⬆️' if x > 0 else '⬇️')  # Adiciona setas
    df['Sequence'] = (df['Trend'] != df['Trend'].shift()).cumsum()
    df['Sequência'] = df.groupby('Sequence').cumcount() + 1
    df.drop(columns=['Sequence'], inplace=True)
    return df.sort_index(ascending=False)

# Função para adicionar a coluna "Preço em R$" quando o ativo for Bitcoin
def adicionar_preco_em_real(df, uc1_df):
    df = df.copy()
    df['Preço em R$'] = df['Preço'] * uc1_df['Preço']
    return df

# Função para estilizar tabela
def estilizar_tabela(dataframe):
    styled_df = dataframe.style.applymap(
        lambda val: 'color: green; font-weight: bold;' if val == '⬆️' else (
            'color: red; font-weight: bold;' if val == '⬇️' else ''),
        subset=['Trend']
    ).format({
        'Preço': lambda x: formatar_moeda(x),  # Formata a coluna Preço em moeda
        'Variação (%)': "{:.2f}%",  # Formata a variação percentual
        'Preço em R$': lambda x: formatar_moeda(x) if pd.notnull(x) else "",  # Formata Preço em R$
    })
    return styled_df

# Interface com duas colunas
col1, col2 = st.columns([5, 16])

# Lista de períodos disponíveis
periodos_disponiveis = ["60 dias", "6 meses", "1 ano", "5 anos", "Base Total"]

# Função para filtrar dados por período
def filtrar_por_periodo(df, periodo):
    hoje = datetime.today()  # Corrigido para usar apenas datetime.today()
    if periodo == "60 dias":
        inicio = hoje - timedelta(days=60)
    elif periodo == "6 meses":
        inicio = hoje - timedelta(days=6 * 30)
    elif periodo == "1 ano":
        inicio = hoje - timedelta(days=365)
    elif periodo == "5 anos":
        inicio = hoje - timedelta(days=5 * 365)
    else:  # "max_periodo"
        return df  # Retorna todos os dados
    return df[df.index >= inicio]

# Coluna 1: Seleção de ativos
with col1:
    ativo_selecionado = st.selectbox("Selecione o ativo:", list(ativos.keys()))
    periodo_selecionado = st.selectbox("Selecione o período:", periodos_disponiveis)
    st.write("Visualizar Médias Móveis:")
    exibir_medias_moveis = st.checkbox("Exibir Médias Móveis (50, 100, 200 períodos)", value=False)

    st.divider()
# Coluna 2: Tabela de dados processados



# Obter dados do ativo selecionado
df_ativo = ativos_selected[ativo_selecionado]
df_ativo_filtrado = filtrar_por_periodo(df_ativo, periodo_selecionado)

variacao = df_ativo_filtrado['Preço'].pct_change().dropna()
desviop_7d = variacao.rolling(window=7).std()
desviop_14d = variacao.rolling(window=14).std()
desviop_21d = variacao.rolling(window=21).std()

media_desvios = pd.concat([desviop_7d, desviop_14d, desviop_21d], axis=1).mean(axis=1).tail(1).values[0]
media_desvios_perc = media_desvios * 100

media_desvios_series = pd.concat([desviop_7d, desviop_14d, desviop_21d], axis=1).mean(axis=1)

volatilidade_atual = media_desvios_series.iloc[-1] * 100
volatilidade_atual_decimal = volatilidade_atual / 100
var_95_diario = z_95 * volatilidade_atual_decimal
var_99_diario = z_99 * volatilidade_atual_decimal
var_99_percentual = var_99_diario * 100
var_95_percentual = var_95_diario * 100


dp = 8

# Calcular métricas
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

if not df_mes_atual.empty:
    preco_inicio_mes = df_mes_atual['Preço'].iloc[0]  # Primeiro preço do mês
    variacao_mensal = ((preco_atual - preco_inicio_mes) / preco_inicio_mes) * 100
else:
    preco_inicio_mes = preco_atual  # Caso não haja dados para o mês atual
    variacao_mensal = 0.0


# Identificar variações diárias
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



# Calcular as médias móveis apenas se o checkbox estiver ativo
if exibir_medias_moveis:
    df_ativo_filtrado['Media_50'] = df_ativo['Preço'].rolling(window=50).mean()
    df_ativo_filtrado['Media_100'] = df_ativo['Preço'].rolling(window=100).mean()
    df_ativo_filtrado['Media_200'] = df_ativo['Preço'].rolling(window=200).mean()



# Layout da aplicação
st.title("Monitoramento de Ativos - Cambirela")


with col1:
    
    col3, col4 = st.columns([5, 5])
    
    with col3:
        st.metric("Último Ajuste", f"{preco_atual:,.3f}")  
        st.metric("Volatilidade (%)", f"{volatilidade_atual:.3f}%")
        st.metric("Var % Base Filtrada", f"{variacao_base_filtrada:.2f}%")
        st.metric("Máx. Seq. Negativa", f"{max_seq_negativa} dias")  # Nova métrica
        st.metric("Mínima do Período", f"{minimo:,.3f}")

    with col4:   
        st.metric("Variação do Dia (%)", f"{variacao_dia:.2f}%")
        st.metric("Value at Risk (%)", f"{var_95_percentual:,.3f}%")
        st.metric("Var % Mensal", f"{variacao_mensal:.2f}%")
        st.metric("Máx. Seq. Positiva", f"{max_seq_positiva} dias")  # Nova métrica
        st.metric("Máxima do Período", f"{maximo:,.3f}")


with col2:

    # Gráfico do ativo
    # Obter os valores mínimo e máximo
    minimo = df_ativo_filtrado['Preço'].min()
    maximo = df_ativo_filtrado['Preço'].max()

    # Obter as datas correspondentes ao mínimo e máximo
    data_minimo = df_ativo_filtrado['Preço'].idxmin()
    data_maximo = df_ativo_filtrado['Preço'].idxmax()

    # Cor da linha personalizada
    verde_detalhes = brand_colors["VERDE_DETALHES"]
    branco = brand_colors["CREME"]


    # Criar o gráfico
    fig = go.Figure()


    show_legend = False  # Exibir a legenda apenas na primeira linha

    # Linha do preço
    fig.add_trace(go.Scatter(
        x=df_ativo_filtrado.index, 
        y=df_ativo_filtrado['Preço'], 
        mode='lines', 
        name=ativo_selecionado, 
        line=dict(color=verde_detalhes)
    ))

    # Marcar os dias da maior sequência positiva
    fig.add_trace(go.Scatter(
        x=dias_seq_positiva,
        y=df_ativo_filtrado.loc[dias_seq_positiva, 'Preço'],
        mode='markers',
        name='Maior Seq. Positiva',
        marker=dict(color=brand_colors["CREME"], size=8)
    ))

    # Marcar os dias da maior sequência negativa
    fig.add_trace(go.Scatter(
        x=dias_seq_negativa,
        y=df_ativo_filtrado.loc[dias_seq_negativa, 'Preço'],
        mode='markers',
        name='Maior Seq. Negativa',
        marker=dict(color=brand_colors["CINZA"], size=8)
    ))


    # Marcar os dias com variações positivas maiores que 2%
    fig.add_trace(go.Scatter(
        x=dias_var_positiva,
        y=df_ativo_filtrado.loc[dias_var_positiva, 'Preço'],
        mode='markers',
        name='Var. > +VaR%',
        marker=dict(color=brand_colors["CREME"], size=16)
    ))

    # Marcar os dias com variações negativas menores que -2%
    fig.add_trace(go.Scatter(
        x=dias_var_negativa,
        y=df_ativo_filtrado.loc[dias_var_negativa, 'Preço'],
        mode='markers',
        name='Var. < -VaR%',
        marker=dict(color=brand_colors["CINZA"], size=16)
    ))



    # Logo centralizado no gráfico
    fig.add_layout_image(
        dict(
            source=f'data:image/png;base64,{img_str}',
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            sizex=0.8, sizey=0.8,
            xanchor="center", yanchor="middle",
            opacity=0.1,
            layer="below"
        )
    )



    for i, (up, down, perc_up, perc_down) in enumerate(zip(desvios, desvios_neg, desvios_percentuais, desvios_neg_percentuais), start=1):
        # Linha de desvio positivo
        fig.add_trace(go.Scatter(
            x=df_ativo_filtrado.index[-15:], y=[up] * 10,
            mode='lines+text',
            line=dict(dash='dash', color=brand_colors['VERDE_TEXTO']),
            text=[''] * dp + [f'+{i} DP | {up:.2f} | <b style="color:#1F4741; font-size:16px;">(+{perc_up:.2f}%)</b>'] + [''] * 4,
            textposition="top center",
            textfont=dict(color="#FFFCF5", size=14),
            showlegend=(i == 1),  # Mostra legenda apenas na primeira iteração
            legendgroup="Desvios",  # Agrupa os traços como "Desvios"
            name="Exibir Desvios" if i == 1 else None,  # Define o nome da legenda apenas na primeira linha
            opacity=1
        ))

        # Linha de desvio negativo
        fig.add_trace(go.Scatter(
            x=df_ativo_filtrado.index[-15:], y=[down] * 10,
            mode='lines+text',
            line=dict(dash='dash', color=brand_colors['VERDE_TEXTO']),
            text=[''] * dp + [f'-{i} DP | {down:.2f} | <b style="color:#49E2B1; font-size:16px;">({perc_down:.2f}%)</b>'] + [''] * 5,
            textposition="top center",
            textfont=dict(color="#FFFCF5", size=14),
            showlegend=(i == 1),  # Mostra legenda apenas na primeira iteração
            legendgroup="Desvios",  # Agrupa os traços como "Desvios"
            name="Exibir Desvios" if i == 1 else None,  # Define o nome da legenda apenas na primeira linha
            opacity=1
        ))





    # Flecha no ponto mínimo
    fig.add_annotation(
        x=data_minimo,
        y=minimo,
        text="Mínima",
        showarrow=True,
        arrowhead=3,
        arrowsize=2,
        arrowwidth=3,
        arrowcolor=branco,
        font=dict(color=branco),
        ax=-150,  # Ajuste de posição horizontal
        ay=-70  # Ajuste de posição vertical
    )

    # Flecha no ponto máximo
    fig.add_annotation(
        x=data_maximo,
        y=maximo,
        text="Máxima",
        showarrow=True,
        arrowhead=3,
        arrowsize=2,
        arrowwidth=3,
        arrowcolor=branco,
        font=dict(color=branco),
        ax=-150,  # Ajuste de posição horizontal
        ay=70  # Ajuste de posição vertical
    )



    # Adicionar médias móveis ao gráfico
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



    # Configurações do layout
    fig.update_layout(
        title=f"Gráfico de Preços - {ativo_selecionado} ({periodo_selecionado})",
        yaxis_title="Preço",
        template="plotly_white",
        yaxis=dict(side='right', showgrid=False),
        width=1200,  # Largura do gráfico
        height=800   # Altura do gráfico
    )


    # Exibir o gráfico
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()

    if ativo_selecionado:
        ativo_df = ativos_selected[ativo_selecionado]
        ativo_df_processado = processar_dados_com_setas_coloridas(ativo_df)
        
        # Adiciona a coluna "Preço em R$" para Bitcoin
        if ativo_selecionado == 'Bitcoin' and 'UC1' in ativos_selected:
            uc1_df = ativos_selected['UC1']
            uc1_df = uc1_df.sort_index()  # Ordena UC1 por Data para consistência
            ativo_df_processado = adicionar_preco_em_real(ativo_df_processado, uc1_df)

        # Estiliza a tabela
        styled_df = estilizar_tabela(ativo_df_processado)
        st.write(f"Análise do Ativo: {ativo_selecionado}")
        st.dataframe(styled_df, use_container_width=True)
# Observações e insights (opcional)
st.write("### Insights")
st.text("Inclua aqui observações sobre o comportamento do ativo selecionado.")
