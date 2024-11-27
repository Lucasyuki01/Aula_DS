import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Função para carregar o modelo treinado
@st.cache_data
def load_model():
    with open('C:\\Users\\Yuki\\Desktop\\github\\Aula_DS\\brasileirao_predicition\\modelo_treinado.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Carregar o modelo
model = load_model()

# Função para carregar os dados
@st.cache_data
def load_data():
    data = pd.read_csv('C:\\Users\\Yuki\\Desktop\\github\\Aula_DS\\brasileirao_predicition\\data.csv')
    return data

data = load_data()

# Definindo as features usadas para o modelo
features = ['ano_campeonato', 'rodada', 'time_mandante', 'time_visitante', 'publico',
            'gols_marcados_mandante', 'gols_sofridos_mandante', 'gols_marcados_visitante',
            'gols_sofridos_visitante', 'diferenca_gols_mandante', 'diferenca_gols_visitante']

# Função para simular resultados de uma rodada
def simulate_round(round_number, data):
    # Filtrar os jogos da rodada
    round_games = data[data['rodada'] == round_number]
    predictions = []
    for _, row in round_games.iterrows():
        # Simular cada jogo
        input_features = np.array([row[features]]).astype(np.float32)
        prediction = model.predict(input_features)
        predictions.append(prediction[0])
    round_games['predicted_result'] = predictions
    return round_games

# Função para simular um confronto direto
def simulate_match(team1, team2, data):
    match_data = data[(data['time_mandante'] == team1) & (data['time_visitante'] == team2)]
    if not match_data.empty:
        input_features = np.array([match_data.iloc[0][features]]).astype(np.float32)
        result = model.predict(input_features)
        return result[0]
    return "No match data available"

# Função para mostrar análises de um time
def show_team_analysis(team, data):
    team_data = data[(data['time_mandante'] == team) | (data['time_visitante'] == team)]
    plt.figure()
    plt.plot(team_data['rodada'], team_data['gols_mandante'], label='Goals Scored')
    plt.plot(team_data['rodada'], team_data['gols_visitante'], label='Goals Conceded')
    plt.title(f'Performance of {team} over the season')
    plt.xlabel('Round')
    plt.ylabel('Goals')
    plt.legend()
    plt.show()
    return plt

# Título da aplicação
st.title('Simulador do Campeonato Brasileiro')

# Abas principais
tab1, tab2, tab3 = st.tabs(["Simulação de Rodadas", "Confronto Direto", "Análise de Time"])

with tab1:
    st.header("Simulação de Rodadas do Campeonato")
    rodada = st.number_input('Escolha a Rodada', min_value=1, max_value=38, value=1, step=1)
    if st.button('Simular Rodada'):
        # Verificar se todas as colunas necessárias estão presentes antes de simular a rodada
        if {'gols_marcados_mandante', 'gols_sofridos_mandante', 'gols_marcados_visitante', 'gols_sofridos_visitante', 'diferenca_gols_mandante', 'diferenca_gols_visitante'}.issubset(data.columns):
            results = simulate_round(rodada, data)
            st.write(results)
        else:
            st.error("Algumas colunas necessárias estão faltando no DataFrame.")
with tab2:
    st.header("Simulação de Confronto Direto")
    team1 = st.selectbox('Selecione o Time da Casa', options=data['time_mandante'].unique())
    team2 = st.selectbox('Selecione o Time Visitante', options=data['time_visitante'].unique())
    if st.button('Simular Confronto'):
        result = simulate_match(team1, team2, data)
        st.write(f"Resultado previsto: {result}")

with tab3:
    st.header("Análise de Time")
    team = st.selectbox('Selecione o Time', options=data['time_mandante'].unique())
    if st.button('Mostrar Análises'):
        plt = show_team_analysis(team, data)
        st.pyplot(plt)
