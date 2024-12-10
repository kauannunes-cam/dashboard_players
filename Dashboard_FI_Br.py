import pandas as pd
import requests
import zipfile
import streamlit as st
from io import BytesIO
from PIL import Image
import base64
import plotly.express as px

pd.options.display.float_format = '{:.4f}'.format

# Função para carregar a imagem e converter para base64
def load_image_as_base64(image_path):
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Caminho da logo
logo_path = "./logo_transparente.png"
img_str = load_image_as_base64(logo_path)

# Configurações de cores
brand_colors = {
    "VERDE_TEXTO": "#1F4741",
    "VERDE_PRINCIPAL": "#2B6960",
    "VERDE_DETALHES": "#49E2B1",
    "CREME": "#FFFCF5",
    "CINZA": "#76807D",
    "PRETO": "#1B1B1B"
}

# Função para formatar valores em R$
def formatar_valor(valor):
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# Configuração inicial da página
st.set_page_config(layout="wide", page_title="Análise de Fundos de Investimento", page_icon="📊")

# Entrada de data e barra de pesquisa
col1, col2 = st.columns([1, 2])

with col1:
    data_selecionada = st.date_input("Selecione uma data:", value=pd.Timestamp.now())

with col2:
    fundo_pesquisado = st.text_input("🔎 Pesquisar Fundo de Investimento:", "")

# Carregar dados
@st.cache_data
def carregar_dados(data_selecionada):
    ano = data_selecionada.year
    mes = f"{data_selecionada.month:02d}"
    arquivo = f"inf_diario_fi_{ano}{mes}.zip"

    url = f'https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_{ano}{mes}.zip'
    download = requests.get(url)

    # Salvar o arquivo zip localmente
    with open(arquivo, "wb") as arquivo_cvm:
        arquivo_cvm.write(download.content)

    # Ler o conteúdo do arquivo zip
    arquivo_zip = zipfile.ZipFile(arquivo)
    dados_fundos = pd.read_csv(arquivo_zip.open(arquivo_zip.namelist()[0]), sep=";", encoding='ISO-8859-1')

    # Ler dados de cadastro dos fundos
    dados_cadastro = pd.read_csv('https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv', 
                                 sep=";", encoding='ISO-8859-1')
    dados_cadastro = dados_cadastro[['CNPJ_FUNDO', 'DENOM_SOCIAL']].drop_duplicates()

    # Filtrar para início e fim do mês
    data_inicio_mes = dados_fundos['DT_COMPTC'].min()
    data_fim_mes = dados_fundos['DT_COMPTC'].max()

    dados_fundos_filtrado = dados_fundos[(dados_fundos['DT_COMPTC'].isin([data_inicio_mes, data_fim_mes]))]

    # Merge com dados de cadastro
    base_final = pd.merge(dados_fundos_filtrado, dados_cadastro, how="left", 
                          left_on="CNPJ_FUNDO_CLASSE", right_on="CNPJ_FUNDO")
    
    base_final['Saldo Liq Cap/Res'] = base_final['CAPTC_DIA'] - base_final['RESG_DIA']

    base_final = base_final[['TP_FUNDO_CLASSE', 'CNPJ_FUNDO', 'DENOM_SOCIAL', 'DT_COMPTC', 'VL_QUOTA', 
                             'VL_PATRIM_LIQ', 'CAPTC_DIA', 'RESG_DIA', 'NR_COTST', 'Saldo Liq Cap/Res']]
    
    # Remover valores ausentes e duplicatas
    base_final = base_final.dropna(subset=['DENOM_SOCIAL', 'VL_PATRIM_LIQ'])
    base_final = base_final.drop_duplicates()

    return base_final

# Carregar a base de dados
base_final = carregar_dados(data_selecionada)

# Filtrar pelo fundo pesquisado
if fundo_pesquisado:
    base_final = base_final[base_final['DENOM_SOCIAL'].str.contains(fundo_pesquisado, case=False, na=False)]

# Resumo Total
total_captacao = base_final['CAPTC_DIA'].sum()
total_resgate = base_final['RESG_DIA'].sum()
saldo_liquido = total_captacao - total_resgate

col1, col2, col3 = st.columns(3)
col1.metric("Captação Total", formatar_valor(total_captacao))
col2.metric("Resgate Total", formatar_valor(total_resgate))
col3.metric("Saldo Líquido", formatar_valor(saldo_liquido))

# Layout de tabela e gráfico
col1, col2 = st.columns([7, 5])

# Tabela com CSS para ocupar todo o espaço
with col1:
    st.subheader("🔹 Tabela - Fundos de Investimento do Brasil")
    st.markdown("""
        <style>
            .dataframe {
                width: 100%;
                overflow-x: auto;
            }
            table {
                width: 100%;
                table-layout: fixed;
            }
        </style>
    """, unsafe_allow_html=True)

    st.dataframe(base_final[['TP_FUNDO_CLASSE', 'DT_COMPTC', 'DENOM_SOCIAL', 
                             'VL_PATRIM_LIQ', 'CAPTC_DIA', 'VL_QUOTA', 'RESG_DIA', 'Saldo Liq Cap/Res']])

# Gráfico: Maiores Fundos por Patrimônio Líquido
with col2:
    st.subheader("🔹 Top 5 Fundos por Patrimônio Líquido")
    top_fundos = base_final.sort_values(by="VL_PATRIM_LIQ", ascending=False).head(5)
    fig_barras = px.bar(
        top_fundos,
        x="VL_PATRIM_LIQ",
        y="DENOM_SOCIAL",
        orientation='h',
        text="TP_FUNDO_CLASSE",
        color="TP_FUNDO_CLASSE",
        title="Top 5 Fundos por Patrimônio Líquido"
    )
    fig_barras.update_traces(textposition="inside", insidetextanchor="middle")
    st.plotly_chart(fig_barras, use_container_width=True)

# Rodapé
st.markdown("---")
st.markdown("**Desenvolvido por Kauan Nunes - Cambirela Educa**")
