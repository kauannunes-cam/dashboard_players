import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import datetime
import base64
from io import BytesIO
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import os
import numpy as np



hoje = datetime.datetime.today()

dp = 8

def formatar_moeda(valor):
    valor_str = f"R$ {valor:,.2f}"  # Formata com separador de milhar e duas casas decimais
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
logo_path = "./logo_transparente.png"


# Carregar logo como base64 para usar em marca d'água
img_str = load_image_as_base64(logo_path)

#############################  BANCO DE DADOS  ###################################
st.set_page_config(layout="wide")

@st.cache_data
def carregar_dados():
    xls = pd.ExcelFile("./History Cot.xlsx")
    ativos = {}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=2)
        df.columns.values[0] = 'Data'
        df.columns.values[1] = 'Preço'
        df.set_index('Data', inplace=True)
        df.sort_index(inplace=True)
        ativos[sheet_name] = df
    return ativos

ativos = carregar_dados()

# Lista de ativos selecionados
selected_assets = ['BBDXY', 'Bitcoin', 'CL1', 'CO1', 'DM1', 'ES1', 'NQ1', 'ODF26', 'ODF27', 'ODN26', 
                   'T-02', 'T-10', 'UC1', 'UC2', 'USDMXN', 'VIX', 'XAU', 'XB1']

# Filtra o dicionário para incluir apenas os ativos selecionados
ativos_selected = {key: ativos[key] for key in selected_assets if key in ativos}

# Combina os dataframes selecionados em um único dataframe para calcular a correlação
ativos_combined = pd.concat(ativos_selected.values(), axis=1, keys=ativos_selected.keys())
ativos_combined = ativos_combined.tail(100)
ativos_combined.columns = ativos_combined.columns.get_level_values(0)

# Calcula a matriz de correlação
correlation_matrix = ativos_combined.corr()

#############################  DASHBOARD  ###################################

# Configuração do layout do Streamlit
st.title("Análise Quantitativa")

# Definindo escala de cores personalizada com as cores da marca
custom_colorscale = [
    [0, brand_colors["CINZA"]],
    [0.5, brand_colors["CREME"]],
    [1, brand_colors["VERDE_PRINCIPAL"]],
]

# Criação do heatmap interativo com anotação para valores > 0.65
fig = go.Figure(
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
        if correlation_value > 0.65:
            fig.add_annotation(
                x=correlation_matrix.columns[i],
                y=correlation_matrix.columns[j],
                text=f"{correlation_value:.2f}",
                showarrow=False,
                font=dict(color=brand_colors["CREME"], size=12)
            )

# Configurações de layout para o heatmap
fig.update_layout(
    title="Matriz de Correlação dos Ativos",
    xaxis=dict(tickangle=-45),
    autosize=True,
    width=1000,
    height=600,
    margin=dict(l=50, r=50, t=100, b=100),
)


ativos_predicao = ['UC1', 'UC2', 'Bitcoin', 'XAU', 'ES1', 'NQ1', 'DM1', 'XB1', 'ODF27', 'ODF26', 'ODN26']


ativos_selected = {key: ativos[key] for key in selected_assets if key in ativos}

# Função para treinar o modelo e fazer previsões
@st.cache_data
def treinar_e_prever(dados_acao):
    # Garantir que a coluna 'Preço' está presente no DataFrame
    if 'Preço' not in dados_acao.columns:
        raise KeyError("O DataFrame fornecido não contém uma coluna 'Preço'.")
        
    # Preparação dos dados
    dados_acao = dados_acao[['Preço']].tail(300)
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

#st.title("Volatilidade com Predição")




################################################################

st.divider()
st.write("KPI e OKRs em construção...")
st.divider()


################################################################

########### Filtros ###########
col1, col2, col3 = st.columns([2, 3, 2])

# Seleção de ativo na coluna 1
with col1:
    ativo_selecionado = st.selectbox(
        "Selecione o Ativo para Visualizar Volatilidade", 
        options=list(ativos_selected.keys()),
        index=list(ativos_selected.keys()).index("UC1") if "UC1" in ativos_selected.keys() else 0  # Define UC1 como padrão, se existir
    )
df = ativos_selected[ativo_selecionado]

   
# Exibe o título de predição apenas para ativos da lista `ativos_predicao`
if ativo_selecionado in ativos_predicao and ativo_selecionado in direcoes_mercado:
    direcao_pregao = direcoes_mercado[ativo_selecionado]
    st.divider()
    st.title(f"A predição para o ativo {ativo_selecionado} é: {direcao_pregao}")

# Seleção de médias móveis e períodos na coluna 2 em duas linhas
with col2:
    col1, col2 = st.columns([2, 2])
    with col1:
        periodo_volatilidade = st.radio("Selecione o Período de Volatilidade:", ["3 Meses", "6 Meses", "12 Meses"], horizontal=True)
    with col2:
        st.write("Visualizar Médias Móveis:")
        exibir_medias_moveis = st.checkbox("Visualizar Médias Moveis (50,100 e 200 periodos)", value=False)

# Seleção de data final na coluna 3
df = ativos_selected[ativo_selecionado]
data_max = df.index.max().date()
with col3:
    data_final = st.date_input("Data Final", value=hoje.date(), max_value=hoje.date())


variacao_1 = df['Preço'].pct_change().dropna()
desviop_7d_1 = variacao_1.rolling(window=7).std()
desviop_14d_1 = variacao_1.rolling(window=14).std()
desviop_21d_1 = variacao_1.rolling(window=21).std()
media_desvios_1 = pd.concat([desviop_7d_1, desviop_14d_1, desviop_21d_1], axis=1).mean(axis=1).tail(1).values[0]
media_desvios_series_1 = pd.concat([desviop_7d_1, desviop_14d_1, desviop_21d_1], axis=1).mean(axis=1)
media_desvios_perc_1 = media_desvios_1 * 100


if periodo_volatilidade == "3 Meses":
    dias = 90
elif periodo_volatilidade == "6 Meses":
    dias = 180
else:
    dias = 252

# Filtragem e cálculos de volatilidade baseados na data final selecionada
df_filtrado = df.loc[:str(data_final)].tail(dias)
df_filtrado['Preço_Suave'] = df_filtrado['Preço']

# Calcular a volatilidade diária como média dos desvios de 7, 14 e 21 dias
ultimo_ajuste = df_filtrado['Preço'].iloc[-1]
variacao = df_filtrado['Preço'].pct_change().dropna()
desviop_7d = variacao.rolling(window=7).std()
desviop_14d = variacao.rolling(window=14).std()
desviop_21d = variacao.rolling(window=21).std()
media_desvios = pd.concat([desviop_7d, desviop_14d, desviop_21d], axis=1).mean(axis=1).tail(1).values[0]
media_desvios_perc = media_desvios * 100

# Calcular a série de volatilidades ao longo do período
media_desvios_series = pd.concat([desviop_7d, desviop_14d, desviop_21d], axis=1).mean(axis=1)

# Calcular a média móvel de volatilidade conforme o período selecionado
if periodo_volatilidade == "3 Meses":
    if len(media_desvios_series_1) >= 90:
        media_volatilidade = media_desvios_series_1.rolling(window=90).mean().iloc[-1] * 100  # Média móvel de 90 dias
    else:
        media_volatilidade = float('nan')  # Valor padrão caso não haja dados suficientes
elif periodo_volatilidade == "6 Meses":
    if len(media_desvios_series_1) >= 180:
        media_volatilidade = media_desvios_series_1.rolling(window=180).mean().iloc[-1] * 100  # Média móvel de 180 dias
    else:
        media_volatilidade = float('nan')  # Valor padrão caso não haja dados suficientes
        
elif periodo_volatilidade == "12 Meses":
    if len(media_desvios_series_1) >= 252:
        media_volatilidade = media_desvios_series_1.rolling(window=252).mean().iloc[-1] * 100  # Média móvel de 180 dias
    else:
        media_volatilidade = float('nan')  # Valor padrão caso não haja dados suficientes

# Obter a última volatilidade calculada
volatilidade_atual = media_desvios_series.iloc[-1] * 100 if not media_desvios_series.empty else float('nan')

# Cálculo das linhas dos desvios e suas variações percentuais
desvios = [ultimo_ajuste + i * media_desvios * ultimo_ajuste for i in range(1, dp)]
desvios_percentuais = [(desvio - ultimo_ajuste) / ultimo_ajuste * 100 for desvio in desvios]
desvios_neg = [ultimo_ajuste - i * media_desvios * ultimo_ajuste for i in range(1, dp)]
desvios_neg_percentuais = [(desvio - ultimo_ajuste) / ultimo_ajuste * 100 for desvio in desvios_neg]


########### Gráfico de Volatilidade ###########
fig_volatilidade = go.Figure()

# Linha de preço suavizada
fig_volatilidade.add_trace(go.Scatter(
    x=df_filtrado.index, y=df_filtrado['Preço_Suave'],
    mode='lines', name='Preço Suavizado',
    line=dict(color=brand_colors['CREME']),
    showlegend=True
))

# Adicionar linhas de desvios padrão com variação percentual e valores no gráfico
for i, (up, down, perc_up, perc_down) in enumerate(zip(desvios, desvios_neg, desvios_percentuais, desvios_neg_percentuais), start=1):
    # Linha de desvio positivo
    fig_volatilidade.add_trace(go.Scatter(
        x=df_filtrado.index[-10:], y=[up] * 10,
        mode='lines+text',
        line=dict(dash='dash', color=brand_colors['VERDE_TEXTO']),
        text=[''] * dp + [f'+{i} DP | {up:.2f} | <b style="color:#1F4741; font-size:18px;">(+{perc_up:.2f}%)</b>'] + [''] * 4,
        textposition="top center",  # Ajuste de posição para colocar o texto acima da linha
        textfont=dict(color="#FFFCF5", size=14),
        showlegend=False
    ))
    
    # Linha de desvio negativo
    fig_volatilidade.add_trace(go.Scatter(
        x=df_filtrado.index[-10:], y=[down] * 10,
        mode='lines+text',
        line=dict(dash='dash', color=brand_colors['VERDE_TEXTO']),
        text=[''] * dp + [f'-{i} DP | {down:.2f} | <b style="color:#49E2B1; font-size:18px;">({perc_down:.2f}%)</b>'] + [''] * 5,
        textposition="top center",  # Ajuste de posição para colocar o texto acima da linha
        textfont=dict(color="#FFFCF5", size=14),
        showlegend=False
    ))
    
# Linha de "Último Ajuste" no meio
fig_volatilidade.add_trace(go.Scatter(
    x=df_filtrado.index[-10:], y=[ultimo_ajuste] * 10,
    mode='lines+text',
    line=dict(color="white", width=2),
    text=[''] * 9 + [f'Ajust. Ant {ultimo_ajuste:.2f}'],

))


# Adiciona uma linha horizontal no preço atual
fig_volatilidade.add_trace(go.Scatter(
    x=df_filtrado.index[-10:], y=[ultimo_ajuste] * 10,
    mode='lines',
    line=dict(color="white", width=2),
    name='Último Ajuste',  # Nome definido para aparecer na legenda
    showlegend=True
))

# Adicionar volatilidade média e volatilidade do período no gráfico
fig_volatilidade.add_annotation(
    text=f"Volatilidade: {volatilidade_atual:.2f}%",
    xref="paper", yref="paper",
    x=0.5, y=1.10, showarrow=False,
    font=dict(size=44, color=brand_colors['CINZA']),
    opacity=0.8
)
fig_volatilidade.add_annotation(
    text=f"Volatilidade Média ({periodo_volatilidade}): {media_volatilidade:.2f}%",
    xref="paper", yref="paper",
    x=0.5, y=1, showarrow=False,
    font=dict(size=44, color=brand_colors['CINZA']),
    opacity=0.8
)

# Calcular as médias móveis sobre todo o dataframe
if exibir_medias_moveis:
    df['Media_50'] = df['Preço'].rolling(window=50).mean()
    df['Media_100'] = df['Preço'].rolling(window=100).mean()
    df['Media_200'] = df['Preço'].rolling(window=200).mean()

# Depois de calcular, filtrar os últimos 60 dias para visualização
df_visualizacao = df.loc[:str(data_final)].tail(dias)



# Adicionar médias móveis ao gráfico usando apenas o trecho filtrado
if exibir_medias_moveis:
    # Garantir que as médias móveis estão disponíveis em `df_visualizacao`
    fig_volatilidade.add_trace(go.Scatter(
        x=df_visualizacao.index, y=df_visualizacao['Media_50'],
        mode='lines', name='Média Móvel 50',
        line=dict(color=brand_colors['VERDE_PRINCIPAL'], dash="dot")
    ))
    fig_volatilidade.add_trace(go.Scatter(
        x=df_visualizacao.index, y=df_visualizacao['Media_100'],
        mode='lines', name='Média Móvel 100',
        line=dict(color=brand_colors['CINZA'], dash="dot")
    ))
    fig_volatilidade.add_trace(go.Scatter(
        x=df_visualizacao.index, y=df_visualizacao['Media_200'],
        mode='lines', name='Média Móvel 200',
        line=dict(color=brand_colors['CREME'], dash="dot")
    ))

# Cálculo das variações diárias percentuais
df_visualizacao['Variação (%)'] = df_visualizacao['Preço'].pct_change() * 100

# Encontrar a maior e menor variação percentual
max_variacao = df_visualizacao['Variação (%)'].max()
min_variacao = df_visualizacao['Variação (%)'].min()
max_date = df_visualizacao['Variação (%)'].idxmax()
min_date = df_visualizacao['Variação (%)'].idxmin()


# Adicionar gráfico de barras com as variações percentuais
fig_volatilidade.add_trace(go.Bar(
    x=df_visualizacao.index,
    y=df_visualizacao['Variação (%)'],
    name='Variação Diária (%)',
    marker_color=[
        brand_colors['VERDE_DETALHES'] if v >= 0 else brand_colors['VERDE_TEXTO'] 
        for v in df_visualizacao['Variação (%)']
    ],
    yaxis='y2',
    showlegend=True
))

# Configurar eixos
fig_volatilidade.update_layout(
    xaxis=dict(
        tickformat="%Y-%m-%d",
        nticks=20,
        showgrid=False  # Remove a grade do eixo x
    ),
    yaxis=dict(
        title="Preço",
        side="right",
        tickformat=",.2f",
        showgrid=False  # Remove a grade do eixo y
    ),
    yaxis2=dict(
        title="Variação Diária (%)",
        overlaying='y',
        side='left',
        tickformat=".2f",  # Formato com duas casas decimais
        showgrid=False,  # Remove a grade do eixo y secundário
        range=[min_variacao - 1, max_variacao + 1]  # Ajusta o range para focar nas variações
    ),
    width=1000,
    height=600,
    margin=dict(l=20, r=10, t=100, b=100),
    legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top", bgcolor=brand_colors['CINZA']),
)

# Adicionar texto dentro das barras para a maior variação positiva
fig_volatilidade.add_annotation(
    x=max_date,
    y=max_variacao * 0.6,  # Posiciona o texto dentro da barra, 80% da altura
    text=f"{max_variacao:.2f}%",
    showarrow=False,  # Sem seta
    font=dict(color=brand_colors['VERDE_TEXTO'], size=14),
    align="center",
    textangle=90,  # Rotaciona o texto para ficar na vertical
    yref="y2"  # Refere-se ao eixo y2 para posicionar corretamente na variação percentual
)

# Adicionar texto dentro das barras para a maior variação negativa
fig_volatilidade.add_annotation(
    x=min_date,
    y=min_variacao * 0.6,  # Posiciona o texto dentro da barra, 80% da altura
    text=f"{min_variacao:.2f}%",
    showarrow=False,  # Sem seta
    font=dict(color=brand_colors['VERDE_DETALHES'], size=14),
    align="center",
    textangle=90,  # Rotaciona o texto para ficar na vertical
    yref="y2"  # Refere-se ao eixo y2 para posicionar corretamente na variação percentual
)


# Adicionar flag para a maior variação positiva (seta da direita para o topo da barra)
fig_volatilidade.add_annotation(
    x=max_date,
    y=max_variacao,
    text="Max Var Positiva",
    showarrow=True,
    arrowhead=6,  # Estilo da seta
    arrowcolor=brand_colors['CINZA'],  # Cor da seta
    font=dict(color=brand_colors['VERDE_DETALHES']),
    xshift=0,  # Deslocamento horizontal para a direita
    yshift=5,  # Ajusta para aparecer acima da barra
    ax=0,  # Direção da seta da direita para a barra
    ay=-5,  # Direção da seta para cima, atingindo o topo da barra
    yref="y2"  # Posiciona no eixo y secundário
)

# Adicionar flag para a maior variação negativa (seta da esquerda para o topo da barra)
fig_volatilidade.add_annotation(
    x=min_date,
    y=min_variacao,
    text="Max Var Nagativa",
    showarrow=True,
    arrowhead=6,  # Estilo da seta
    arrowcolor=brand_colors['CINZA'],  # Cor da seta
    font=dict(color=brand_colors['VERDE_DETALHES']),
    xshift=0,  # Deslocamento horizontal para a esquerda
    yshift=-10,  # Ajusta para aparecer abaixo da barra
    ax=-0,  # Direção da seta da esquerda para a barra
    ay=5,  # Direção da seta para baixo, atingindo o topo da barra
    yref="y2"  # Posiciona no eixo y secundário
)

# Configuração de margens para evitar sobreposição
fig_volatilidade.update_layout(
    margin=dict(l=20, r=20, t=150, b=10),
    yaxis2=dict(tickformat=".2f")
)


fig_volatilidade.update_layout(
    xaxis=dict(
        tickformat="%Y-%m-%d",
        nticks=28,
        tickangle=-45
    )
    )

# Logo centralizado no gráfico
fig_volatilidade.add_layout_image(
    dict(
        source=f'data:image/png;base64,{img_str}',
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        sizex=0.8, sizey=0.8,
        xanchor="center", yanchor="middle",
        opacity=0.2,
        layer="below"
    )
)

# Exibir o gráfico no Streamlit
if ativo_selecionado in direcoes_mercado:
    direcao_pregao = direcoes_mercado[ativo_selecionado]
    
    # Exibir título com a direção para ativos permitidos
    st.divider()
    
    # Se o ativo selecionado for XB1 ou UC1, exibe a calculadora de risco ao lado
    if ativo_selecionado in ['XB1', 'UC1']:
        col1, col2, col3 = st.columns([3, 8, 3])
        
        
        with col1:
            df_visualizacao = df_filtrado[['Preço']].copy()
            df_visualizacao['Data'] = df_visualizacao.index.date  # Exibir apenas a data sem o horário
            df_visualizacao['Variação (%)'] = df_visualizacao['Preço'].pct_change() * 100
            
            # Calcular a sequência de pregões positivos ou negativos
            df_visualizacao['Sequência'] = (df_visualizacao['Variação (%)']
                                            .apply(lambda x: 1 if x > 0 else -1)
                                            .groupby((df_visualizacao['Variação (%)'] > 0) != (df_visualizacao['Variação (%)'] > 0).shift())
                                            .cumsum())
            df_visualizacao['Sequência'] = df_visualizacao['Sequência'].apply(lambda x: abs(x) if x > 0 else -abs(x))

            # Seleciona as colunas necessárias e ordena a DataFrame por data em ordem decrescente
            df_visualizacao = df_visualizacao[['Data', 'Preço', 'Variação (%)', 'Sequência']].sort_index(ascending=False)
            
            # Formatar a coluna Preço com duas casas decimais
            df_visualizacao['Preço'] = df_visualizacao['Preço'].map("{:.2f}".format)

            # Aplicar o gradiente de cores em toda a linha, baseado na coluna "Variação (%)"
            def color_row(row):
                norm_val = (row['Variação (%)'] - df_visualizacao['Variação (%)'].min()) / (df_visualizacao['Variação (%)'].max() - df_visualizacao['Variação (%)'].min())
                color = f'rgba(49, 226, 177, {norm_val})' if row['Variação (%)'] >= 0 else f'rgba(31, 71, 65, {norm_val})'
                return [f'background-color: {color}; color: white; text-align: right;' for _ in row]

            styled_df = df_visualizacao.style.apply(color_row, axis=1).background_gradient(subset=['Variação (%)'], cmap="Greens")

            # Exibir o DataFrame estilizado no Streamlit com scroll
            st.dataframe(styled_df, height=800, use_container_width=True)


        # Exibição do gráfico na coluna 1
        with col2:

            st.plotly_chart(fig_volatilidade, use_container_width=True)
 

        # Calculadora de risco na coluna 2
        with col3:
            st.title('Calculadora de Risco')
            ultimo_preco = df['Preço'].iloc[-1]
            
            # Calculadora específica para XB1
            if ativo_selecionado == 'XB1':
                contrato_tipo = st.radio("Tipo de Contrato", ["Cheio", "Mini"])
                # Ajuste no cálculo conforme a descrição
                col3, col4 = st.columns([2, 2])
                
                valor_por_contrato = 25 * (ultimo_preco * 0.2) if contrato_tipo == "Cheio" else ultimo_preco * 0.2
                valor_margem = 50000 if contrato_tipo == "Cheio" else 2000
                alavancagem_indice = valor_por_contrato/valor_margem
                
                with col3:
                    st.write(f"Valor por Contrato ({contrato_tipo}): {formatar_moeda(valor_por_contrato)}  |  Alav. {alavancagem_indice:.2}x")
                with col4:
                    st.write(f"Margem por Contrato ({contrato_tipo}): {formatar_moeda(valor_margem)}")

            # Calculadora específica para UC1
            if ativo_selecionado == 'UC1':
                contrato_tipo = st.radio("Tipo de Contrato", ["Cheio", "Mini"])
                # Definir o valor por contrato com base no tipo
                
                valor_por_contrato = ultimo_preco * 50 if contrato_tipo == "Cheio" else ultimo_preco * 10
                valor_margem = 250000 if contrato_tipo == "Cheio" else 10000
                alavancagem_dolar = valor_por_contrato/valor_margem                
                col3, col4 = st.columns([2, 2])
                
                with col3:
                    st.write(f"Valor por Contrato ({contrato_tipo}): {formatar_moeda(valor_por_contrato)}  |  Alav. {alavancagem_dolar:.2f}x")
                with col4:
                    st.write(f"Margem por Contrato ({contrato_tipo}): {formatar_moeda(valor_margem)}")
                
            # Entrada do número de contratos e cálculo do total da posição
            num_contratos = st.number_input("Quantidade de Contratos", min_value=1, value=1, step=1)
            total_posicao = num_contratos * valor_por_contrato
            st.write(f"Total da Posição: {formatar_moeda(total_posicao)}")

            # Tabela com valores positivos e negativos de desvio
            desvio_data = {
                "DP Positivo": [f"+{i} DP" for i in range(1, 6)],
                "Variação Líquida (+)": [formatar_moeda(total_posicao * perc_up / 100) for perc_up in desvios_percentuais[:5]],
                "DP Negativo": [f"-{i} DP" for i in range(1, 6)],
                "Variação Líquida (-)": [formatar_moeda(total_posicao * perc_down / 100) for perc_down in desvios_neg_percentuais[:5]]
            }
            df_desvio = pd.DataFrame(desvio_data)

            # Aplicar estilos na tabela
            def estilo_barras(valor):
                cor = "#49E2B1" if "+" in valor else "#1F4741"  # Cor verde para positivos e azul escuro para negativos
                largura = abs(float(valor.replace("R$", "").replace(".", "").replace(",", "."))) / total_posicao * 100
                return f"background: linear-gradient(90deg, {cor} {largura}%, transparent {largura}%); color: white;"

            # Exibir a tabela com estilo
            st.table(df_desvio.style.applymap(estilo_barras, subset=["Variação Líquida (+)", "Variação Líquida (-)"]))
    else:
        
        
        col1, col2 = st.columns([3, 8])
        
        
        with col1:
            # Criar DataFrame com Data, Preço e Variação (%)
            df_visualizacao = df_filtrado[['Preço']].copy()
            df_visualizacao['Variação (%)'] = df_visualizacao['Preço'].pct_change() * 100
            df_visualizacao = df_visualizacao[['Preço', 'Variação (%)']].tail(14)  # Limita a 14 linhas
            # Função para estilizar a coluna de Variação (%)
            def style_variacao(val):
                norm_val = (val - df_visualizacao['Variação (%)'].min()) / (df_visualizacao['Variação (%)'].max() - df_visualizacao['Variação (%)'].min())
                color = f'linear-gradient(90deg, {brand_colors["VERDE_TEXTO"]} {norm_val*100}%, {brand_colors["VERDE_DETALHES"]} {(1 - norm_val)*100}%);'
                return f'background: {color}; color: white; text-align: right;'

                styled_df = df_visualizacao.style.applymap(apply_gradient, subset=['Variação (%)'])

            # Aplicar o estilo ao DataFrame e exibir no Streamlit
            st.dataframe(df_visualizacao.style.applymap(style_variacao), height=400)        
        
        with col2:
            st.plotly_chart(fig_volatilidade, use_container_width=True) 
            
            
    st.divider()

else:
    # Exibe o gráfico mesmo sem predição disponível
    st.plotly_chart(fig_volatilidade, use_container_width=True)
    st.divider()
    

# Ajuste de estilo para transparência do fundo e espaçamento
st.markdown(f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: 1600px;
            padding: 1rem 2rem;
            background-color: rgba(0, 0, 0, 0);
        }}
    </style>
""", unsafe_allow_html=True)

#################

# Normalizar os dados dos ativos
scaler = MinMaxScaler()
ativos_normalizados = {}
for ativo, df in ativos.items():
    df_normalizado = pd.DataFrame(scaler.fit_transform(df[['Preço']]), columns=['Preço'], index=df.index)
    ativos_normalizados[ativo] = df_normalizado

# Seleção de ativos para o gráfico comparativo
selected_assets = list(ativos_normalizados.keys())

col1, col2 = st.columns([2, 2])

with col1:
    # Exibir a figura interativa no Streamlit
    st.plotly_chart(fig, use_container_width=True)
    

with col2:
    col1, col2 = st.columns([2, 2])

    with col1:
        # Exibir a figura interativa no Streamlit com pré-seleção de "UC1" para Ativo 1
        ativo_1 = st.selectbox("Selecione o Ativo 1", options=selected_assets, index=selected_assets.index("UC1"))

    with col2:
        # Seleção de Ativo 2 com pré-seleção de "XB1"
        ativo_2 = st.selectbox("Selecione o Ativo 2", options=selected_assets, index=selected_assets.index("XB1"))

    # Filtragem dos dados para os últimos 60 dias
    df_ativo_1 = ativos_normalizados[ativo_1].tail(60)
    df_ativo_2 = ativos_normalizados[ativo_2].tail(60)

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


###########  Volatilidade com Predição  #############
st.divider()
