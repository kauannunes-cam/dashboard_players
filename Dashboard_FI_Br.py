import pandas as pd
import requests
import zipfile
import streamlit as st
from io import BytesIO
from PIL import Image
import base64
import plotly.graph_objects as go



brand_colors = {
    "VERDE_TEXTO": "#1F4741",
    "VERDE_PRINCIPAL": "#2B6960",
    "VERDE_DETALHES": "#49E2B1",
    "CREME": "#FFFCF5",
    "CINZA": "#76807D",
    "PRETO": "#1B1B1B"
}


# Configurações gerais
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

# Configuração inicial da página
st.set_page_config(layout="wide", page_title="Análise de Evolução de Cotas", page_icon="📊")

# Entrada de data e pesquisa por fundo
col1, col2 = st.columns([1, 2])
with col1:
    data_inicio = st.date_input("📅 Data Inicial:", pd.Timestamp(2024, 1, 1))
    data_fim = st.date_input("📅 Data Final:", pd.Timestamp.now())
with col2:
    fundo_pesquisado = st.text_input("🔎 Pesquise uma Asset/Gestora:", "")

# Carregar dados de múltiplos meses
@st.cache_data
def carregar_dados_multiplos(data_inicio, data_fim):
    base_final = pd.DataFrame()

    # Converter para Timestamp
    data_inicio = pd.Timestamp(data_inicio)
    data_fim = pd.Timestamp(data_fim)

    for ano in range(data_inicio.year, data_fim.year + 1):
        for mes in range(1, 13):
            if pd.Timestamp(ano, mes, 1) < data_inicio or pd.Timestamp(ano, mes, 1) > data_fim:
                continue
            try:
                url = f"https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_{ano}{mes:02d}.zip"
                download = requests.get(url)
                with zipfile.ZipFile(BytesIO(download.content)) as arquivo_zip:
                    dados_fundos = pd.read_csv(arquivo_zip.open(arquivo_zip.namelist()[0]), sep=";", encoding='ISO-8859-1')
                dados_cadastro = pd.read_csv("https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv", sep=";", encoding='ISO-8859-1')
                dados_cadastro = dados_cadastro[['CNPJ_FUNDO', 'DENOM_SOCIAL']].drop_duplicates()
                dados_fundos = pd.merge(dados_fundos, dados_cadastro, how="left", left_on="CNPJ_FUNDO_CLASSE", right_on="CNPJ_FUNDO")

                # Adicionar colunas faltantes com zeros
                for col in ['CAPTC_DIA', 'RESG_DIA', 'NR_COTST']:
                    if col not in dados_fundos.columns:
                        dados_fundos[col] = 0

                base_final = pd.concat([base_final, dados_fundos[['DT_COMPTC', 'CNPJ_FUNDO_CLASSE', 'DENOM_SOCIAL',
                                                                  'VL_QUOTA', 'VL_PATRIM_LIQ', 'CAPTC_DIA', 'RESG_DIA', 'NR_COTST']]])
            except:
                continue

    base_final['DT_COMPTC'] = pd.to_datetime(base_final['DT_COMPTC'], format='%Y-%m-%d')
    return base_final.dropna().drop_duplicates()

# Carregar e filtrar dados
if fundo_pesquisado:
    base_final = carregar_dados_multiplos(data_inicio, data_fim)
    base_final = base_final[base_final['DENOM_SOCIAL'].str.contains(fundo_pesquisado, case=False)]

    if not base_final.empty:
        fundos_disponiveis = base_final['DENOM_SOCIAL'].unique()
        fundo_selecionado = st.selectbox("Selecione um Fundo de Investimento:", fundos_disponiveis)
        dados_fundo = base_final[base_final['DENOM_SOCIAL'] == fundo_selecionado].sort_values(by="DT_COMPTC")

        # Calcular métricas
        dados_fundo['DIFF_VL_QUOTA'] = dados_fundo['VL_QUOTA'].pct_change() * 100
        dados_fundo['DIFF_VL_PATRIM'] = dados_fundo['VL_PATRIM_LIQ'].pct_change() * 100
        dados_fundo['SALD_DIA'] = dados_fundo['CAPTC_DIA'] - dados_fundo['RESG_DIA']
        cap_total = dados_fundo['CAPTC_DIA'].sum()
        resg_total = dados_fundo['RESG_DIA'].sum()
        saldo_acumulado = cap_total - resg_total
        perf_acumulada = ((dados_fundo['VL_QUOTA'].iloc[-1] / dados_fundo['VL_QUOTA'].iloc[0]) - 1) * 100


        # Exibir métricas
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Variação % Acumulada", f"{perf_acumulada:.2f}%")
        col2.metric("Patrimônio Líquido Atual", f"R$ {dados_fundo['VL_PATRIM_LIQ'].iloc[-1]:,.2f}")
        col3.metric("Qtd de Cotistas Atual", f"{int(dados_fundo['NR_COTST'].iloc[-1]):,}")
        col4.metric("Captação Acumulada", f"R$ {cap_total:,.2f}")
        col5.metric("Resgate Acumulado", f"R$ {resg_total:,.2f}")
        col6.metric("Saldo Líquido Acumulado", f"R$ {saldo_acumulado:,.2f}")


        # Gráfico
        st.subheader("📈 Evolução das QUOTAS e Patrimônio Líquido")
        fig = go.Figure()

        # Linha VL_QUOTA
        fig.add_trace(go.Scatter(
            x=dados_fundo['DT_COMPTC'], 
            y=dados_fundo['VL_QUOTA'],
            name="VL_QUOTA",
            line=dict(color=brand_colors["VERDE_PRINCIPAL"])
        ))

        # Linha VL_PATRIM_LIQ
        fig.add_trace(go.Scatter(
            x=dados_fundo['DT_COMPTC'], 
            y=dados_fundo['VL_PATRIM_LIQ'],
            name="VL_PATRIM_LIQ",
            line=dict(color=brand_colors["CINZA"]),
            yaxis="y2"
        ))

        # Barras Captação Líquida
        cores_barras = [
            brand_colors["VERDE_DETALHES"] if valor >= 0 else brand_colors["VERDE_TEXTO"]
            for valor in dados_fundo['SALD_DIA']
        ]

        # Adicionar barras com cores dinâmicas
        fig.add_trace(go.Bar(
            x=dados_fundo['DT_COMPTC'], 
            y=dados_fundo['SALD_DIA'],
            name="Captação Líquida",
            marker=dict(color=cores_barras)
        ))

        # Configurações do layout
        fig.update_layout(
            yaxis=dict(
                title="Valor da QUOTA",
                showgrid=False,  # Remover grid do eixo y
                zeroline=False   # Remover linha zero
            ),
            yaxis2=dict(
                title="VL Patrimônio Líquido",
                overlaying="y",
                side="right",
                showgrid=False,  # Remover grid do eixo y2
                zeroline=False
            ),
            xaxis=dict(
                title="",
                showgrid=False,  # Remover grid do eixo x
                zeroline=False
            ),
            plot_bgcolor="rgba(0,0,0,0)",  # Fundo transparente
            legend=dict(
                x=0.5,
                y=1.2,
                orientation="h",
                xanchor="center"
            ),
            height=500
        )

        # Adicionar imagem no fundo
        fig.add_layout_image(
            dict(
                source=f'data:image/png;base64,{img_str}',
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                sizex=0.8, sizey=0.8,
                xanchor="center", yanchor="middle",
                opacity=0.6,
                layer="below"
            )
        )
        st.plotly_chart(fig, use_container_width=True)


    # Filtrar pelo fundo pesquisado
    base_final = base_final[base_final['DENOM_SOCIAL'].str.contains(fundo_pesquisado, case=False, na=False)]
else:
    st.info("Digite o nome de um Fundo de Investimento para visualizar a lista de fundos, gráfico e tabela.")

# Rodapé
st.markdown("---")
st.markdown("**Desenvolvido por Kauan Nunes - Cambirela Educa**")
