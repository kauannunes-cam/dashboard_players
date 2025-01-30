import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import datetime
from datetime import datetime, timedelta
import base64
from io import BytesIO
from PIL import Image
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

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


custom_colorscale = [
    [0, brand_colors["CINZA"]],
    [0.5, brand_colors["CREME"]],
    [1, brand_colors["VERDE_PRINCIPAL"]],
]

# Função para carregar a imagem e converter para base64
def load_image_as_base64(image_path):
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Caminho da logo (ajuste conforme necessário)
logo_path = "logo_transparente.png"


# Carregar logo como base64 para usar em marca d'água
img_str = load_image_as_base64(logo_path)

#############################  BANCO DE DADOS  ###################################
st.set_page_config(layout="wide")

@st.cache_data
def carregar_dados():
    xls = pd.ExcelFile("History Cot.xlsx")
    ativos = {}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=1)
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


ativos_combined = pd.concat(ativos_selected.values(), axis=1, keys=ativos_selected.keys())
ativos_combined = ativos_combined.tail(100)
ativos_combined.columns = ativos_combined.columns.get_level_values(0)

# Calcula a matriz de correlação
correlation_matrix = ativos_combined.corr()

grid_corrl = go.Figure(
    data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale=custom_colorscale,
        colorbar=dict(title="Correlação"),
        zmin=-1,
        zmax=1
    )
)

# Adicionar anotações para valores de correlação maiores que 0.65
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        correlation_value = correlation_matrix.values[i, j]
        if correlation_value > 0.65 or correlation_value < -0.65:
            grid_corrl.add_annotation(
                x=correlation_matrix.columns[i],
                y=correlation_matrix.columns[j],
                text=f"{correlation_value:.2f}",
                showarrow=False,
                font=dict(color=brand_colors["CREME"], size=12)
            )

# Configurações de layout para o heatmap
grid_corrl.update_layout(
    title="Matriz de Correlação dos Ativos",
    xaxis=dict(tickangle=-45),
    autosize=True,
    width=1000,
    height=600,
    margin=dict(l=50, r=50, t=100, b=100),
)


# Lista de ativos
ativos_predicao = ['BBDXY', 'Bitcoin', 'CL1', 'CO1', 'DM1', 'ES1', 'NQ1', 'ODF26', 'ODF27', 'ODN26', 
                   'T-02', 'T-10', 'UC1', 'UC2', 'USDMXN', 'VIX', 'XAU', 'XB1', 'BCOMGR', 'CDS Brazil']


# Função para treinar o modelo e fazer previsões
@st.cache_data
def treinar_e_prever(dados_acao):
    # Garantir que a coluna 'Preço' está presente no DataFrame
    if 'Preço' not in dados_acao.columns:
        raise KeyError("O DataFrame fornecido não contém uma coluna 'Preço'.")
        
    # Preparação dos dados
    dados_acao = dados_acao[['Preço']].tail(1000)
    cotacao = dados_acao['Preço'].to_numpy().reshape(-1, 1)
    tamanho_dados_treinamento = int(len(cotacao) * 0.8)

    escalador = MinMaxScaler(feature_range=(0, 1))
    dados_entre_0_e_1_treinamento = escalador.fit_transform(cotacao[:tamanho_dados_treinamento])
    dados_entre_0_e_1_teste = escalador.transform(cotacao[tamanho_dados_treinamento:])
    dados_entre_0_e_1 = np.concatenate([dados_entre_0_e_1_treinamento, dados_entre_0_e_1_teste]).reshape(-1, 1)

    # Preparação dos dados para treinamento
    treinamento_x, treinamento_y = [], []
    for i in range(60, tamanho_dados_treinamento):
        treinamento_x.append(dados_entre_0_e_1[i - 60:i, 0])
        treinamento_y.append(dados_entre_0_e_1[i, 0])
    
    treinamento_x, treinamento_y = np.array(treinamento_x), np.array(treinamento_y)
    treinamento_x = treinamento_x.reshape(treinamento_x.shape[0], treinamento_x.shape[1], 1)

    # Criação e treinamento do modelo LSTM
    modelo = Sequential([
        LSTM(50, return_sequences=True, input_shape=(treinamento_x.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    modelo.compile(optimizer="adam", loss="mean_squared_error")
    modelo.fit(treinamento_x, treinamento_y, batch_size=1, epochs=1)

    # Preparação dos dados para teste e predição
    dados_teste = dados_entre_0_e_1[tamanho_dados_treinamento - 60:]
    teste_x = np.array([dados_teste[i - 60:i, 0] for i in range(60, len(dados_teste))])
    teste_x = teste_x.reshape(teste_x.shape[0], teste_x.shape[1], 1)

    predicoes = modelo.predict(teste_x)
    predicoes = escalador.inverse_transform(predicoes)

    return predicoes, dados_acao['Preço'].iloc[tamanho_dados_treinamento:].values

# Função para obter a direção projetada
@st.cache_data
def get_direcao_projetada(ativos):
    direcoes = {}
    for ativo, dados_acao in ativos.items():
        if ativo in ativos_predicao:
            predicoes, valores_reais = treinar_e_prever(dados_acao)
            if len(predicoes) > 0:
                valor_projetado = predicoes[-1][0]
                ultimo_ajuste = dados_acao['Preço'].iloc[-1]
                var_perc_projetado = (valor_projetado - ultimo_ajuste) / ultimo_ajuste * 100
                direcao_mercado = "Alta" if var_perc_projetado > 0 else "Baixa"
            else:
                direcao_mercado = "Predição indisponível"
            direcoes[ativo] = direcao_mercado
    return direcoes

# Obter direções projetadas para cada ativo disponível
direcoes_mercado = get_direcao_projetada(ativos_selected)






st.title("Análise Quantitativa")

# Função para calcular a variação percentual e sequências com setas coloridas
def processar_dados_com_setas_coloridas(df):
    df = df.copy()
    df['Variação (%)'] = df['Preço'].pct_change(fill_method=None) * 100
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

variacao = df_ativo_filtrado['Preço'].pct_change(fill_method=None).dropna()
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
volatilidade = df_ativo_filtrado['Preço'].pct_change(fill_method=None).std() * np.sqrt(len(df_ativo_filtrado)) * 100


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
df_ativo_filtrado['Variação'] = df_ativo_filtrado['Preço'].pct_change(fill_method=None)

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

    # Calcular os spreads entre o preço e as médias móveis
    df_ativo_filtrado['Spread_MM100'] = df_ativo_filtrado['Preço'] - df_ativo_filtrado['Media_100']
    df_ativo_filtrado['Spread_MM200'] = df_ativo_filtrado['Preço'] - df_ativo_filtrado['Media_200']
    
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





with col1:
    col3, col4 = st.columns([5, 5])

    with col3:
        st.metric("Último Ajuste", f"{preco_atual:,.3f}")  
        st.metric("Volatilidade (%)", f"{volatilidade_atual:.3f}%")
        st.metric("Var % Base Filtrada", f"{variacao_base_filtrada:.2f}%")
        st.metric("Máx. Seq. Negativa", f"{max_seq_negativa} dias")
        st.metric("Mínima do Período", f"{minimo:,.3f}")
        if exibir_medias_moveis:
            st.metric("Spread MM100", f"{df_ativo_ultimo_spread_mm100:.3f}")
            st.metric("Máx. Spread MM100 (VaR 95%)", f"{limite_superior_100:.3f}")

    with col4:   
        st.metric("Variação do Dia (%)", f"{variacao_dia:.2f}%")
        st.metric("Value at Risk (%)", f"{var_95_percentual:,.3f}%")
        st.metric("Var % Mensal", f"{variacao_mensal:.2f}%")
        st.metric("Máx. Seq. Positiva", f"{max_seq_positiva} dias")
        st.metric("Máxima do Período", f"{maximo:,.3f}")
        if exibir_medias_moveis:
            st.metric("Spread MM200", f"{df_ativo_ultimo_spread_mm200:.3f}")
            st.metric("Máx. Spread MM200 (VaR 95%)", f"{limite_superior_200:.3f}")



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
        name='Variação Maior VaR%',
        marker=dict(color=brand_colors["CREME"], size=16)
    ))

    # Marcar os dias com variações negativas menores que -2%
    fig.add_trace(go.Scatter(
        x=dias_var_negativa,
        y=df_ativo_filtrado.loc[dias_var_negativa, 'Preço'],
        mode='markers',
        name='Variação Menor VaR%',
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

    # Exibe o título de predição apenas para ativos da lista `ativos_predicao`
    if ativo_selecionado in ativos_predicao and ativo_selecionado in direcoes_mercado:
        direcao_pregao = direcoes_mercado[ativo_selecionado]
        st.title(f"A predição para o ativo {ativo_selecionado} é: {direcao_pregao}")
        st.divider()
        
        
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


st.divider()

st.title(f'Analise por Sazonalidade do {ativo_selecionado}')



if ativo_selecionado:
    def normalize(df_ativo, base_value=100):
        return df_ativo / df_ativo.iloc[0] * base_value

    # Criar colunas auxiliares
    df_ativo['Year'] = df_ativo.index.year
    df_ativo['Quarter'] = df_ativo.index.quarter
    df_ativo['DayOfYear'] = df_ativo.index.dayofyear

    # Determinar trimestre atual
    current_quarter = pd.Timestamp.now().quarter  # Armazena uma vez o trimestre atual
    df_quarter = df_ativo[df_ativo['Quarter'] == current_quarter]

    # Verificar tendência do trimestre atual
    if not df_quarter.empty:
        first_price = df_quarter['Preço'].iloc[0]
        last_price = df_quarter['Preço'].iloc[-1]
        trend = "Alta" if last_price > first_price else "Baixa"
    else:
        trend = "Indefinida"

    # Convertendo o índice para coluna para plotly
    df_ativo.reset_index(inplace=True)

    fig_sazonal = go.Figure()

    for year in df_ativo['Year'].unique():
        yearly_data = df_ativo[df_ativo['Year'] == year]
        normalized_prices = normalize(yearly_data['Preço'])
        fig_sazonal.add_trace(go.Scatter(
            x=yearly_data['DayOfYear'],
            y=normalized_prices,
            mode='lines',
            name=str(year)
        ))

    # Atualizando o layout para remover as grades, ajustar a legenda e adicionar marca d'água
    fig_sazonal.update_layout(
        xaxis=dict(
            title='Meses',
            tickmode='array',
            tickvals=[31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365],
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            showgrid=False,
        ),
        height=700,
        yaxis=dict(
            title=f'Preço Normalizado -  Sazonalidade',
            side='right',
            showgrid=False
        ),
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        annotations=[
            dict(
                text=f'{ativo_selecionado}: Sazonalidade atual de {trend}',
                xref='paper',
                yref='paper',
                x=0.5,
                y=0.99,
                opacity=0.5,
                font=dict(size=60, color='#76807D'),
                showarrow=False
            )
        ]
    )

    st.plotly_chart(fig_sazonal, use_container_width=True)



st.divider()




scaler = MinMaxScaler()
ativos_normalizados = {}
for ativo, df in ativos.items():
    df_normalizado = pd.DataFrame(scaler.fit_transform(df[['Preço']]), columns=['Preço'], index=df.index)
    ativos_normalizados[ativo] = df_normalizado

# Seleção de ativos para o gráfico comparativo
selected_assets = list(ativos_normalizados.keys())


# Layout da aplicação
st.title("Correlação")
col1, col2 = st.columns([2, 2])

with col1:
    # Exibir a figura interativa no Streamlit
    st.plotly_chart(grid_corrl, use_container_width=True)
    

with col2:
    col1, col2 = st.columns([2, 2])

    with col1:
        # Exibir a figura interativa no Streamlit com pré-seleção de "UC1" para Ativo 1
        ativo_1 = st.selectbox("Selecione o Ativo 1", options=selected_assets, index=selected_assets.index("UC1"))

    with col2:
        # Seleção de Ativo 2 com pré-seleção de "XB1"
        ativo_2 = st.selectbox("Selecione o Ativo 2", options=selected_assets, index=selected_assets.index("XB1"))

    # Filtragem dos dados para os últimos 180 dias
    df_ativo_1 = ativos_normalizados[ativo_1].tail(180)
    df_ativo_2 = ativos_normalizados[ativo_2].tail(180)

    # Calcular o histórico de correlação para os últimos 60 dias
    df_corr = df_ativo_1['Preço'].rolling(window=5).corr(df_ativo_2['Preço'])

    # Configuração do gráfico comparativo
    fig_corrl = go.Figure()

    # Definir cores RGBA para preenchimento com transparência
    fillcolor_positive = "rgba(49, 226, 177, 0.1)"  # VERDE_DETALHES com 50% de opacidade
    fillcolor_negative = "rgba(31, 71, 65, 0.2)"    # VERDE_TEXTO com 50% de opacidade

    # Adicionar gráfico de montanha para a correlação com transparência configurada diretamente no fillcolor
    fig_corrl.add_trace(go.Scatter(
        x=df_corr.index,
        y=df_corr,
        fill='tozeroy',
        mode='none',
        name="Correlação",
        fillcolor=fillcolor_positive if df_corr.iloc[-1] > 0 else fillcolor_negative,
        yaxis="y2"
    ))
    
    # Linha para o ativo 1
    fig_corrl.add_trace(go.Scatter(
        x=df_ativo_1.index, y=df_ativo_1['Preço'],
        mode='lines', name=ativo_1,
        line=dict(color=brand_colors['VERDE_DETALHES'])
    ))

    # Linha para o ativo 2
    fig_corrl.add_trace(go.Scatter(
        x=df_ativo_2.index, y=df_ativo_2['Preço'],
        mode='lines', name=ativo_2,
        line=dict(color=brand_colors['VERDE_TEXTO'])
    ))


    # Configurações de layout do gráfico
    fig_corrl.update_layout(
        xaxis=dict(
            tickformat="%Y-%m-%d",
            nticks=28,
            tickangle=-45
        ),
        yaxis=dict(title="Preço Normalizado", showgrid=False, side="right"),
        yaxis2=dict(
            title="Correlação",
            overlaying="y",
            side="left",
            showgrid=False,
            range=[-1, 1]  # Limita o eixo da correlação entre -1 e 1
        ),
        plot_bgcolor="rgba(0,0,0,0)",  # Fundo transparente
        width=1000,
        height=500,
        margin=dict(l=20, r=20, t=50, b=50)
    )

    # Exibir o gráfico no Streamlit
    st.plotly_chart(fig_corrl, use_container_width=True)
st.divider()
    
# Rodapé
st.markdown("---")
st.markdown("**Desenvolvido por Kauan Nunes - Cambirela Educa**")
