import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# Determinar o diretório base a partir do diretório atual do script
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Caminhos para os modelos e dados
model_path = os.path.join(base_path, 'models', 'best_random_forest_model.pkl')
encoder_path = os.path.join(base_path, 'models', 'label_encoder.pkl')
symptoms_path = os.path.join(base_path, 'data', 'Testing.csv')

# Carregar o modelo e o encoder
model = joblib.load(model_path)
encoder = joblib.load(encoder_path)

# Carregar a lista de sintomas
symptoms_list = list(pd.read_csv(symptoms_path).columns[:-1])

# Traduções de sintomas
sintomas_traduzidos = {
    "itching": "Coceira",
    "skin_rash": "Erupção cutânea",
    "nodal_skin_eruptions": "Erupções cutâneas nodulares",
    "continuous_sneezing": "Espirros contínuos",
    "shivering": "Tremores",
    "chills": "Calafrios",
    "joint_pain": "Dor nas articulações",
    "stomach_pain": "Dor de estômago",
    "acidity": "Acidez",
    "ulcers_on_tongue": "Úlceras na língua",
    "muscle_wasting": "Perda de massa muscular",
    "vomiting": "Vômito",
    "burning_micturition": "Micção dolorosa",
    "spotting_urination": "Urina com sangue",
    "fatigue": "Fadiga",
    "weight_gain": "Ganho de peso",
    "anxiety": "Ansiedade",
    "cold_hands_and_feet": "Mãos e pés frios",
    "mood_swings": "Mudanças de humor",
    "weight_loss": "Perda de peso",
    "restlessness": "Inquietação",
    "lethargy": "Letargia",
    "patches_in_throat": "Manchas na garganta",
    "irregular_sugar_level": "Nível irregular de açúcar",
    "cough": "Tosse",
    "high_fever": "Febre alta",
    "sunken_eyes": "Olhos fundos",
    "breathlessness": "Falta de ar",
    "sweating": "Sudorese",
    "dehydration": "Desidratação",
    "indigestion": "Indigestão",
    "headache": "Dor de cabeça",
    "yellowish_skin": "Pele amarelada",
    "dark_urine": "Urina escura",
    "nausea": "Náusea",
    "loss_of_appetite": "Perda de apetite",
    "pain_behind_the_eyes": "Dor atrás dos olhos",
    "back_pain": "Dor nas costas",
    "constipation": "Constipação",
    "abdominal_pain": "Dor abdominal",
    "diarrhoea": "Diarreia",
    "mild_fever": "Febre leve",
    "yellow_urine": "Urina amarela",
    "yellowing_of_eyes": "Amarelamento dos olhos",
    "acute_liver_failure": "Falência hepática aguda",
    "fluid_overload": "Sobrecarga de fluidos",
    "swelling_of_stomach": "Inchaço do estômago",
    "swelled_lymph_nodes": "Linfonodos inchados",
    "malaise": "Mal-estar",
    "blurred_and_distorted_vision": "Visão embaçada e distorcida",
    "phlegm": "Fleuma",
    "throat_irritation": "Irritação na garganta",
    "redness_of_eyes": "Vermelhidão dos olhos",
    "sinus_pressure": "Pressão sinusal",
    "runny_nose": "Coriza",
    "congestion": "Congestão",
    "chest_pain": "Dor no peito",
    "weakness_in_limbs": "Fraqueza nos membros",
    "fast_heart_rate": "Frequência cardíaca rápida",
    "pain_during_bowel_movements": "Dor durante movimentos intestinais",
    "pain_in_anal_region": "Dor na região anal",
    "bloody_stool": "Fezes com sangue",
    "irritation_in_anus": "Irritação no ânus",
    "neck_pain": "Dor no pescoço",
    "dizziness": "Tontura",
    "cramps": "Cãibras",
    "bruising": "Hematomas",
    "obesity": "Obesidade",
    "swollen_legs": "Pernas inchadas",
    "swollen_blood_vessels": "Vasos sanguíneos inchados",
    "puffy_face_and_eyes": "Rosto e olhos inchados",
    "enlarged_thyroid": "Tireoide aumentada",
    "brittle_nails": "Unhas frágeis",
    "swollen_extremeties": "Extremidades inchadas",
    "excessive_hunger": "Fome excessiva",
    "extra_marital_contacts": "Contatos extraconjugais",
    "drying_and_tingling_lips": "Secura e formigamento nos lábios",
    "slurred_speech": "Fala arrastada",
    "knee_pain": "Dor no joelho",
    "hip_joint_pain": "Dor na articulação do quadril",
    "muscle_weakness": "Fraqueza muscular",
    "stiff_neck": "Pescoço rígido",
    "swelling_joints": "Articulações inchadas",
    "movement_stiffness": "Rigidez de movimento",
    "spinning_movements": "Movimentos giratórios",
    "loss_of_balance": "Perda de equilíbrio",
    "unsteadiness": "Instabilidade",
    "weakness_of_one_body_side": "Fraqueza em um lado do corpo",
    "loss_of_smell": "Perda de olfato",
    "bladder_discomfort": "Desconforto na bexiga",
    "foul_smell_of_urine": "Cheiro ruim de urina",
    "continuous_feel_of_urine": "Sensação contínua de urinar",
    "passage_of_gases": "Passagem de gases",
    "internal_itching": "Coceira interna",
    "toxic_look_(typhos)": "Aparência tóxica (tifo)",
    "depression": "Depressão",
    "irritability": "Irritabilidade",
    "muscle_pain": "Dor muscular",
    "altered_sensorium": "Sensorium alterado",
    "red_spots_over_body": "Manchas vermelhas pelo corpo",
    "belly_pain": "Dor abdominal",
    "abnormal_menstruation": "Menstruação anormal",
    "dischromic_patches": "Manchas discrômicas",
    "watering_from_eyes": "Lacrimejamento dos olhos",
    "increased_appetite": "Aumento do apetite",
    "polyuria": "Poliúria",
    "family_history": "Histórico familiar",
    "mucoid_sputum": "Escarro mucoso",
    "rusty_sputum": "Escarro enferrujado",
    "lack_of_concentration": "Falta de concentração",
    "visual_disturbances": "Distúrbios visuais",
    "receiving_blood_transfusion": "Recebimento de transfusão de sangue",
    "receiving_unsterile_injections": "Recebimento de injeções não esterilizadas",
    "coma": "Coma",
    "stomach_bleeding": "Sangramento estomacal",
    "distention_of_abdomen": "Distensão abdominal",
    "history_of_alcohol_consumption": "Histórico de consumo de álcool",
    "blood_in_sputum": "Sangue no escarro",
    "prominent_veins_on_calf": "Veias proeminentes na panturrilha",
    "palpitations": "Palpitações",
    "painful_walking": "Caminhada dolorosa",
    "pus_filled_pimples": "Pústulas",
    "blackheads": "Cravos",
    "scurring": "Escoriações",
    "skin_peeling": "Descamação da pele",
    "silver_like_dusting": "Poeira prateada",
    "small_dents_in_nails": "Pequenas depressões nas unhas",
    "inflammatory_nails": "Unhas inflamadas",
    "blister": "Bolha",
    "red_sore_around_nose": "Ferida vermelha ao redor do nariz",
    "yellow_crust_ooze": "Exsudação de crosta amarela"
}

doencas_traduzidas = {
    "(vertigo) Paroymsal  Positional Vertigo": "Vertigem Posicional Paroxística",
    "AIDS": "AIDS",
    "Acne": "Acne",
    "Alcoholic hepatitis": "Hepatite Alcoólica",
    "Allergy": "Alergia",
    "Arthritis": "Artrite",
    "Bronchial Asthma": "Asma Brônquica",
    "Cervical spondylosis": "Espondilose Cervical",
    "Chicken pox": "Catapora",
    "Chronic cholestasis": "Colestase Crônica",
    "Common Cold": "Resfriado Comum",
    "Dengue": "Dengue",
    "Diabetes ": "Diabetes",
    "Dimorphic hemmorhoids(piles)": "Hemorroidas Dimórficas",
    "Drug Reaction": "Reação a Medicamentos",
    "Fungal infection": "Infecção Fúngica",
    "GERD": "DRGE (Doença do Refluxo Gastroesofágico)",
    "Gastroenteritis": "Gastroenterite",
    "Heart attack": "Infarto",
    "Hepatitis B": "Hepatite B",
    "Hepatitis C": "Hepatite C",
    "Hepatitis D": "Hepatite D",
    "Hepatitis E": "Hepatite E",
    "Hypertension ": "Hipertensão",
    "Hyperthyroidism": "Hipertireoidismo",
    "Hypoglycemia": "Hipoglicemia",
    "Hypothyroidism": "Hipotireoidismo",
    "Impetigo": "Impetigo",
    "Jaundice": "Icterícia",
    "Malaria": "Malária",
    "Migraine": "Enxaqueca",
    "Osteoarthristis": "Osteoartrite",
    "Paralysis (brain hemorrhage)": "Paralisia (hemorragia cerebral)",
    "Peptic ulcer diseae": "Doença de Úlcera Péptica",
    "Pneumonia": "Pneumonia",
    "Psoriasis": "Psoríase",
    "Tuberculosis": "Tuberculose",
    "Typhoid": "Tifo",
    "Urinary tract infection": "Infecção do Trato Urinário",
    "Varicose veins": "Varizes",
    "hepatitis A": "Hepatite A"
}

tratamentos = {
    "Vertigem Posicional Paroxística": "Manobras de reposicionamento e exercícios de reabilitação vestibular. Medicamentos podem ser usados temporariamente para aliviar os sintomas.",
    "AIDS": "Tratamento antirretroviral para controlar o vírus e prevenir progressão da doença. Monitoramento contínuo da carga viral e função imunológica.",
    "Acne": "Uso de cremes tópicos com retinoides ou antibióticos. Limpeza de pele e, em casos mais severos, terapias com luz ou medicamentos orais como isotretinoína.",
    "Hepatite Alcoólica": "Cessação do consumo de álcool, tratamento nutricional e, em casos graves, terapia com corticosteroides ou pentoxifilina.",
    "Alergia": "Identificação e evitação de alérgenos. Uso de antialérgicos e, em casos severos, imunoterapia.",
    "Artrite": "Exercícios físicos, medicamentos anti-inflamatórios e modificadores da doença. Terapias físicas como fisioterapia também são recomendadas.",
    "Asma Brônquica": "Uso de inaladores de longa duração e controladores, além de corticosteroides para controle das crises.",
    "Espondilose Cervical": "Exercícios de fisioterapia, medicamentos para dor e, em casos específicos, procedimentos cirúrgicos.",
    "Catapora": "Isolamento para evitar a disseminação, loções calmantes para a pele e antitérmicos para controle da febre.",
    "Colestase Crônica": "Medicamentos para controlar a coceira e proteção hepática. Em alguns casos, procedimentos para desobstruir os canais biliares.",
    "Resfriado Comum": "Repouso, hidratação e uso de analgésicos e antitérmicos. Remédios para alívio dos sintomas como congestão nasal podem ser utilizados.",
    "Dengue": "Hidratação intensiva, uso de antitérmicos (evitar aspirina devido ao risco de sangramento) e monitoramento clínico para sinais de alerta de dengue hemorrágica.",
    "Diabetes": "Monitoramento e controle da glicose, dieta balanceada e exercícios físicos. Uso de medicamentos orais ou insulina conforme necessário.",
    "Hemorroidas Dimórficas": "Banho de assento, uso de pomadas anti-hemorroidárias e, em casos graves, intervenção cirúrgica.",
    "Reação a Medicamentos": "Cessar o uso do medicamento causador e tratamento dos sintomas. Uso de antialérgicos e corticosteroides se necessário.",
    "Infecção Fúngica": "Antifúngicos tópicos ou orais dependendo da severidade e localização da infecção.",
    "DRGE": "Mudanças dietéticas, evitar alimentos que desencadeiam os sintomas e uso de antiácidos e inibidores da bomba de prótons.",
    "Gastroenterite": "Reposição de fluidos e eletrólitos para prevenir desidratação. Dieta leve e, em alguns casos, medicamentos para náusea.",
    "Infarto": "Atendimento de emergência com reperfusão miocárdica e uso de medicamentos para dissolver coágulos. Terapia de reabilitação cardíaca pós-infarto.",
    "Hepatite B": "Monitoramento da função hepática e, em casos crônicos, antivirais para reduzir a progressão da doença hepática.",
    "Hepatite C": "Terapia antiviral para eliminar o vírus, com monitoramento contínuo da função hepática e detecção de fibrose ou cirrose.",
    "Hepatite D": "Tratamento com antivirais em conjunto com terapia para Hepatite B, visto que a infecção depende da presença do vírus da Hepatite B.",
    "Hepatite E": "Geralmente auto-limitada, requer apenas tratamento de suporte. Em casos crônicos, antivirais podem ser necessários.",
    "Hipertensão": "Mudanças de estilo de vida como dieta saudável e exercícios, além de medicamentos anti-hipertensivos para controlar a pressão arterial.",
    "Hipertireoidismo": "Medicamentos antitireoidianos, tratamento com iodo radioativo ou cirurgia dependendo da causa e severidade.",
    "Hipoglicemia": "Consumo imediato de uma fonte rápida de glicose. Ajustes no plano de tratamento de diabetes podem ser necessários.",
    "Hipotireoidismo": "Terapia de reposição hormonal com levotiroxina para normalizar os níveis hormonais.",
    "Impetigo": "Antibióticos tópicos e, em infecções mais extensas, antibióticos orais.",
    "Icterícia": "Tratamento da condição subjacente que está causando a icterícia, como obstrução biliar ou doenças hepáticas.",
    "Malária": "Antimaláricos específicos dependendo do tipo de parasita e da área geográfica da infecção.",
    "Enxaqueca": "Medicamentos para alívio da dor e medicamentos preventivos para reduzir a frequência dos ataques.",
    "Osteoartrite": "Manejo da dor com anti-inflamatórios, fisioterapia e, em casos avançados, cirurgia de substituição articular.",
    "Paralisia (hemorragia cerebral)": "Reabilitação intensiva com fisioterapia, terapia ocupacional e, em alguns casos, cirurgia para aliviar a pressão no cérebro.",
    "Doença de Úlcera Péptica": "Inibidores da bomba de prótons para reduzir a produção de ácido e permitir a cura das úlceras.",
    "Pneumonia": "Antibióticos, repouso, hidratação e, em casos graves, hospitalização para tratamento intensivo.",
    "Psoríase": "Cremes tópicos, fototerapia e medicamentos sistêmicos para controlar os sintomas e reduzir a inflamação.",
    "Tuberculose": "Tratamento prolongado com uma combinação de antibióticos para eliminar completamente a bactéria.",
    "Tifo": "Antibióticos e cuidados de suporte para reduzir os sintomas e prevenir complicações.",
    "Infecção do Trato Urinário": "Antibióticos para eliminar a infecção, muita água para ajudar a limpar a bexiga.",
    "Varizes": "Meias de compressão, mudanças de estilo de vida e, em casos mais severos, procedimentos para fechar ou remover as veias.",
    "Hepatite A": "Tratamento de suporte, já que a condição geralmente resolve por si só. Vacinação é recomendada para prevenção."
}

prevalencia_doencas = {
    "Vertigem Posicional Paroxística": 2.4,
    "AIDS": 0.8,
    "Acne": 85.0,
    "Hepatite Alcoólica": 4.5,
    "Alergia": 30.0,
    "Artrite": 25.0,
    "Asma Brônquica": 10.0,
    "Espondilose Cervical": 15.0,
    "Catapora": 90.0,  # Em crianças sem vacinação
    "Colestase Crônica": 0.5,
    "Resfriado Comum": 75.0,  # Incidência anual em porcentagem da população
    "Dengue": 0.2,
    "Diabetes": 8.5,
    "Hemorroidas Dimórficas": 5.0,
    "Reação a Medicamentos": 10.0,
    "Infecção Fúngica": 20.0,
    "DRGE (Doença do Refluxo Gastroesofágico)": 20.0,
    "Gastroenterite": 30.0,
    "Infarto": 3.0,
    "Hepatite B": 1.5,
    "Hepatite C": 1.0,
    "Hepatite D": 0.4,
    "Hepatite E": 1.2,
    "Hipertensão": 25.0,
    "Hipertireoidismo": 1.3,
    "Hipoglicemia": 0.5,
    "Hipotireoidismo": 5.0,
    "Impetigo": 2.0,
    "Icterícia": 2.5,
    "Malária": 1.8,
    "Enxaqueca": 15.0,
    "Osteoartrite": 18.0,
    "Paralisia (hemorragia cerebral)": 2.1,
    "Doença de Úlcera Péptica": 10.0,
    "Pneumonia": 5.0,
    "Psoríase": 3.0,
    "Tuberculose": 0.3,
    "Tifo": 0.1,
    "Infecção do Trato Urinário": 50.0,
    "Varizes": 15.0,
    "Hepatite A": 1.4
}

def plot_prevalence(disease):
    prevalence = prevalencia_doencas.get(disease, 0)
    fig, ax = plt.subplots()
    ax.pie([prevalence, 100-prevalence], labels=[disease, 'Outros'], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

# Função para prever a doença e registrar o diagnóstico
def predict_disease(symptoms):
    input_data = np.array([symptoms])
    prediction_proba = model.predict_proba(input_data)[0]
    prediction = model.predict(input_data)[0]
    predicted_disease = encoder.inverse_transform([prediction])[0]
    predicted_disease_translated = doencas_traduzidas.get(predicted_disease, "Doença não traduzida")
    tratamento_recomendado = tratamentos.get(predicted_disease_translated, "Tratamento não disponível.")
    
    return predicted_disease_translated, prediction_proba, tratamento_recomendado

# Título do app
st.title('Diagnóstico de Doença com Base em Sintomas')

# Seleção de sintomas via multiselect
translated_symptoms = [sintomas_traduzidos.get(symptom, "Sem tradução") for symptom in symptoms_list]
selected_symptoms_translated = st.multiselect("Selecione os sintomas:", options=translated_symptoms)

# Converter sintomas selecionados de volta para o formato original
selected_symptoms = [symptom for symptom, translation in sintomas_traduzidos.items() if translation in selected_symptoms_translated]

# Preparar a entrada para o modelo
symptoms_input = [1 if symptom in selected_symptoms else 0 for symptom in symptoms_list]

# Botão para diagnosticar
if st.button('Diagnosticar'):
    if any(symptoms_input):  # Verifica se algum sintoma foi selecionado
        diagnosis, probabilities, tratamento = predict_disease(symptoms_input)
        st.write(f"A doença prevista é: {diagnosis}")
        st.write(f"Tratamento recomendado: {tratamento}")
        
        # Chamar a função para desenhar o gráfico de prevalência
        plot_prevalence(diagnosis)

        # Ordenar probabilidades e pegar os índices das maiores probabilidades
        sorted_indices = np.argsort(probabilities)[::-1]
        top_indices = sorted_indices[:5]  # Suponha que você quer mostrar os top 5
        probabilities_sorted = probabilities[top_indices]

        # Traduzir doenças para os rótulos
        diseases = [doencas_traduzidas.get(encoder.inverse_transform([index])[0], "Doença não traduzida") for index in top_indices]

        # Criar gráfico de setores das probabilidades
        fig, ax = plt.subplots()
        ax.pie(probabilities_sorted, labels=diseases, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Garante que a torta seja desenhada como um círculo.
        st.pyplot(fig)
        
    else:
        st.error("Por favor, selecione pelo menos um sintoma.")
