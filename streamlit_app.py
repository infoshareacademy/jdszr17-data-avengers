# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Wczytaj dane treningowe
data = pd.read_csv('overstimulation_dataset.csv')

# Przygotowanie danych
X = data.drop('Overstimulated', axis=1)
y = data['Overstimulated']

# Skalowanie danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Trening modeli
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_scaled, y)

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X, y)

model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_scaled, y)

# Streamlit UI
st.title("Overstimulation Detection App")
st.write("Wprowadź dane dotyczące stylu życia, aby przewidzieć, czy osoba jest przeciążona sensorycznie.")

# Funkcja do wprowadzania danych
def user_input_features():
    Age = st.slider('Age', 18, 60, 30)
    Sleep_Hours = st.slider('Sleep Hours', 3.0, 10.0, 7.0)
    Screen_Time = st.slider('Screen Time (hours)', 1.0, 12.0, 5.0)
    Stress_Level = st.slider('Stress Level (1-10)', 1, 10, 5)
    Noise_Exposure = st.slider('Noise Exposure (0-5)', 0, 5, 2)
    Social_Interaction = st.slider('Social Interaction (per day)', 0, 10, 5)
    Work_Hours = st.slider('Work Hours', 4, 15, 8)
    Exercise_Hours = st.slider('Exercise Hours', 0.0, 3.0, 1.0)
    Caffeine_Intake = st.slider('Caffeine Intake (cups)', 0, 5, 2)
    Multitasking_Habit = st.selectbox('Multitasking Habit', [0, 1])
    Anxiety_Score = st.slider('Anxiety Score (1-10)', 1, 10, 5)
    Depression_Score = st.slider('Depression Score (1-10)', 1, 10, 5)
    Sensory_Sensitivity = st.slider('Sensory Sensitivity (0-4)', 0, 4, 2)
    Meditation_Habit = st.selectbox('Meditation Habit', [0, 1])
    Overthinking_Score = st.slider('Overthinking Score (1-10)', 1, 10, 5)
    Irritability_Score = st.slider('Irritability Score (1-10)', 1, 10, 5)
    Headache_Frequency = st.slider('Headache Frequency (per week)', 0, 7, 2)
    Sleep_Quality = st.slider('Sleep Quality (1-4)', 1, 4, 3)
    Tech_Usage_Hours = st.slider('Tech Usage Hours', 1.0, 10.0, 5.0)
    
    data_input = {
        'Age': Age,
        'Sleep_Hours': Sleep_Hours,
        'Screen_Time': Screen_Time,
        'Stress_Level': Stress_Level,
        'Noise_Exposure': Noise_Exposure,
        'Social_Interaction': Social_Interaction,
        'Work_Hours': Work_Hours,
        'Exercise_Hours': Exercise_Hours,
        'Caffeine_Intake': Caffeine_Intake,
        'Multitasking_Habit': Multitasking_Habit,
        'Anxiety_Score': Anxiety_Score,
        'Depression_Score': Depression_Score,
        'Sensory_Sensitivity': Sensory_Sensitivity,
        'Meditation_Habit': Meditation_Habit,
        'Overthinking_Score': Overthinking_Score,
        'Irritability_Score': Irritability_Score,
        'Headache_Frequency': Headache_Frequency,
        'Sleep_Quality': Sleep_Quality,
        'Tech_Usage_Hours': Tech_Usage_Hours
    }
    return pd.DataFrame(data_input, index=[0])

input_df = user_input_features()

# Wybór modelu
model_choice = st.selectbox('Wybierz model:', ['KNN', 'Random Forest', 'Logistic Regression'])

# Predykcja
if model_choice == 'KNN':
    input_scaled = scaler.transform(input_df)
    prediction = model_knn.predict(input_scaled)
elif model_choice == 'Random Forest':
    prediction = model_rf.predict(input_df)
else:
    input_scaled = scaler.transform(input_df)
    prediction = model_lr.predict(input_scaled)

# Wyświetlanie wyniku
st.subheader('Wynik predykcji:')
if prediction[0] == 1:
    st.error('Osoba jest PRZECIĄŻONA sensorycznie (Overstimulated).')
else:
    st.success('Osoba NIE jest przeciążona sensorycznie (Not Overstimulated).')
