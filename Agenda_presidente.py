import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta

# Cores da marca Cambirela
brand_colors = {
    "VERDE_TEXTO": "#d2b589",
    "VERDE_PRINCIPAL": "#2d63b2",
    "VERDE_DETALHES": "#7cb9f2",
    "CREME": "#d5d5d5",
    "CINZA": "#5c6972",
    "PRETO": "#1B1B1B"
}

# Função para buscar a agenda via web scraping
def buscar_agenda(data):
    url = f"https://www.gov.br/planalto/pt-br/acompanhe-o-planalto/agenda-do-presidente-da-republica-lula/agenda-do-presidente-da-republica/{data}"
    response = requests.get(url)
    
    if response.status_code != 200:
        return pd.DataFrame(), "Erro ao acessar a página"

    soup = BeautifulSoup(response.content, "html.parser")
    compromissos = []

    # Coleta dos compromissos
    for item in soup.find_all("li", class_="item-compromisso-wrapper"):
        horario = item.find("time", class_="compromisso-inicio").get_text(strip=True) if item.find("time") else "N/A"
        titulo = item.find("h2", class_="compromisso-titulo").get_text(strip=True) if item.find("h2") else "N/A"
        local = item.find("div", class_="compromisso-local").get_text(strip=True) if item.find("div", class_="compromisso-local") else "N/A"

        compromissos.append({
            "Horário": horario,
            "Título": titulo,
            "Local": local
        })

    return pd.DataFrame(compromissos), None

# Função para encontrar a agenda mais recente
def encontrar_agenda_mais_recente(data_inicial):
    for dias_passados in range(30):  # Tenta retroceder até 30 dias
        data_teste = (data_inicial - timedelta(days=dias_passados)).strftime("%Y-%m-%d")
        agenda_df, erro = buscar_agenda(data_teste)
        if not agenda_df.empty:
            return agenda_df, data_teste
    return pd.DataFrame(), None

# Função para exibir os dados no dashboard
def exibir_dashboard():
    st.set_page_config(page_title="Agenda Presidencial", layout="wide")

    # Estilização CSS
    st.markdown(
        f"""
        <style>
            body {{
                background-color: {brand_colors['CREME']};
                color: {brand_colors['VERDE_TEXTO']};
            }}
            .main-header {{
                text-align: center;
                margin: 20px 0;
            }}
            .card {{
                background-color: white;
                border-radius: 8px;
                box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
                margin: 10px 0;
                padding: 15px;
            }}
            .card h2 {{
                color: {brand_colors['PRETO']};
                font-size: 1.1rem;
                margin-bottom: 5px;
            }}
            .card p {{
                color: {brand_colors['CINZA']};
                font-size: 0.9rem;
                margin: 0;
            }}
            .time {{
                font-size: 1.3rem;
                color: {brand_colors['VERDE_PRINCIPAL']};
                font-weight: bold;
                margin-bottom: 5px;
            }}

            /* Texto dentro da st.success */
            div.stAlert[data-testid="stAlert-success"] p {
                color: #ffffff !important;
                font-weight: bold;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Cabeçalho
    st.markdown('<div class="main-header"><h1>📅 Agenda Presidencial</h1></div>', unsafe_allow_html=True)

    # Entrada de Data
    data_input = st.date_input("Escolha a data", value=datetime.today())
    data_inicial = datetime.combine(data_input, datetime.min.time())

    # Busca da agenda mais recente
    st.info("Buscando a agenda mais recente...")
    agenda_df, data_encontrada = encontrar_agenda_mais_recente(data_inicial)

    if agenda_df.empty or data_encontrada is None:
        st.error("Nenhuma agenda encontrada nos últimos 30 dias.")
        return

    st.success(f"Agenda encontrada para a data: **{data_encontrada}**")

    # Exibição dos compromissos
    for _, row in agenda_df.iterrows():
        st.markdown(
            f"""
            <div class="card">
                <div class="time">{row['Horário']}</div>
                <h2>{row['Título']}</h2>
                <p>📍 {row['Local']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        

# Main
if __name__ == "__main__":
    exibir_dashboard()


# Rodapé
st.markdown("---")
st.markdown("**Desenvolvido por Kauan Nunes - Trader QUANT**")
