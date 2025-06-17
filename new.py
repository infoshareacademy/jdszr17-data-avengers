import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Overstimulation App", layout="wide")

# Sidebar navigation
st.sidebar.title("Nawigacja")
page = st.sidebar.radio(
    "Przejdź do:",
    ["Eksploracja danych", "Predykcja przeciążenia"],
    index=0
)

# --- Load data ---
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('overstimulation_dataset.csv')
    except FileNotFoundError:
        st.error("❌ Nie znaleziono pliku 'overstimulation_dataset.csv'. Upewnij się, że plik znajduje się w tym samym folderze co ten skrypt.")
        st.stop()
    return data

data = load_data()

# --- Predykcja: wczytaj modele tylko jeśli użytkownik wybierze ---
@st.cache_resource
def train_models(data):
    X = data.drop('Overstimulated', axis=1)
    y = data['Overstimulated']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_knn = KNeighborsClassifier(n_neighbors=5)
    model_knn.fit(X_scaled, y)
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X, y)
    model_lr = LogisticRegression(max_iter=1000, random_state=42)
    model_lr.fit(X_scaled, y)
    return scaler, model_knn, model_rf, model_lr

# --- Eksploracja danych ---
def show_exploration(data):
    st.title("📊 Eksploracja danych")
    with st.expander("1. Proporcje osób przeciążonych sensorycznie"):
        overstim_counts = data["Overstimulated"].value_counts()
        labels = ['Overstimulated (1)', 'Not Overstimulated (0)']
        fig1, ax1 = plt.subplots()
        ax1.pie(overstim_counts, labels=labels, autopct='%1.1f%%', colors=sns.color_palette("Set2"), startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

    with st.expander("2. Rozkład zmiennych binarnych"):
        binary_cols = ['Meditation_Habit', 'Multitasking_Habit']
        for col in binary_cols:
            count = data[col].value_counts()
            fig, ax = plt.subplots()
            ax.pie(count, labels=[str(x) for x in count.index], autopct='%1.1f%%', colors=sns.color_palette('viridis', len(count)).as_hex())
            ax.set_title(f'{col} distribution')
            st.pyplot(fig)

    with st.expander("3. Rozkład zmiennych kategorycznych"):
        categorical_cols = ['Sensory_Sensitivity', 'Sleep_Quality', 'Noise_Exposure', 'Headache_Frequency']
        for col in categorical_cols:
            counts = data[col].value_counts().sort_index()
            fig, ax = plt.subplots()
            sns.barplot(x=counts.index, y=counts.values, ax=ax, color='lightblue')
            ax.set_title(f'{col} distribution')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            st.pyplot(fig)

    with st.expander("4. Histogramy zmiennych numerycznych"):
        fig = data.hist(figsize=(12, 10), color='lightblue', bins=20, edgecolor='gray')
        plt.suptitle("Histogram of numerical columns in dataset")
        st.pyplot(plt.gcf())

    with st.expander("5. Zależność: Wiek a sen"):
        fig, ax = plt.subplots()
        sns.lineplot(x='Age', y='Sleep_Hours', data=data, marker='o', ax=ax)
        ax.set_title('Sleep hours vs Age')
        st.pyplot(fig)

    with st.expander("6. Zależność: Stres a sen"):
        fig, ax = plt.subplots()
        sns.lineplot(x='Stress_Level', y='Sleep_Hours', data=data, marker='o', ax=ax)
        ax.set_title('Sleep hours vs Stress level')
        st.pyplot(fig)

    with st.expander("7. Zależność: Czas przed ekranem a stres"):
        fig, ax = plt.subplots()
        sns.lineplot(x='Stress_Level', y='Screen_Time', data=data, marker='o', ax=ax)
        ax.set_title('Screen time vs Stress level')
        st.pyplot(fig)

    with st.expander("8. Zależność: Przeciążenie vs czas przed ekranem i stres"):
        fig, ax = plt.subplots()
        sns.scatterplot(x='Screen_Time', y='Stress_Level', hue='Overstimulated', data=data, ax=ax)
        ax.set_title('Screen time vs overstimulation')
        st.pyplot(fig)

    with st.expander("9. Pairplot wybranych cech"):
        pairplot_fig = sns.pairplot(data[['Age', 'Sleep_Hours', 'Screen_Time', 'Stress_Level', 'Overstimulated']],
                                    hue='Overstimulated', palette='colorblind')
        st.pyplot(pairplot_fig.fig)

    with st.expander("10. Heatmapa korelacji"):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
        ax.set_title('Correlation heatmap')
        st.pyplot(fig)

# --- Predykcja ---
def user_input_features():
    st.header("Podaj dane do predykcji przeciążenia")
    Age = st.slider('Wiek', 18, 60, 30, step=1)
    Sleep_Hours = st.slider('Godziny snu', 3, 12, 7, step=1)
    Screen_Time = st.slider('Czas przed ekranem (h)', 1, 16, 5, step=1)
    Stress_Level = st.slider('Poziom stresu (1-10)', 1, 10, 5, step=1)
    Noise_Exposure = st.slider('Ekspozycja na hałas (0-5)', 0, 5, 2, step=1)
    Social_Interaction = st.slider('Kontakty społeczne (na dzień)', 0, 20, 5, step=1)
    Work_Hours = st.slider('Godziny pracy', 1, 16, 8, step=1)
    Exercise_Hours = st.slider('Ćwiczenia (h)', 0, 4, 1, step=1)
    Caffeine_Intake = st.slider('Ilość kofeiny (filiżanki)', 0, 10, 2, step=1)
    Multitasking_Habit = st.selectbox('Nawyk multitaskingu', [0, 1])
    Anxiety_Score = st.slider('Poziom lęku (1-10)', 1, 10, 5, step=1)
    Depression_Score = st.slider('Poziom depresji (1-10)', 1, 10, 5, step=1)
    Sensory_Sensitivity = st.slider('Wrażliwość sensoryczna (0-4)', 0, 4, 2, step=1)
    Meditation_Habit = st.selectbox('Nawyk medytacji', [0, 1])
    Overthinking_Score = st.slider('Nadmierne myślenie (1-10)', 1, 10, 5, step=1)
    Irritability_Score = st.slider('Drażliwość (1-10)', 1, 10, 5, step=1)
    Headache_Frequency = st.slider('Częstość bólów głowy (na tydzień)', 0, 7, 2, step=1)
    Sleep_Quality = st.slider('Jakość snu (1-4)', 1, 4, 3, step=1)
    Tech_Usage_Hours = st.slider('Technologie (h)', 1, 16, 5, step=1)

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

def show_prediction(data):
    scaler, model_knn, model_rf, model_lr = train_models(data)
    st.title("🔮 Predykcja przeciążenia sensorycznego")
    input_df = user_input_features()
    model_choice = st.selectbox('Wybierz model:', ['KNN', 'Random Forest', 'Logistic Regression'])

    if model_choice == 'KNN':
        input_scaled = scaler.transform(input_df)
        prediction = model_knn.predict(input_scaled)
    elif model_choice == 'Random Forest':
        prediction = model_rf.predict(input_df)
    else:
        input_scaled = scaler.transform(input_df)
        prediction = model_lr.predict(input_scaled)

    st.subheader('Wynik predykcji:')
    if prediction[0] == 1:
        st.error('Osoba jest PRZECIĄŻONA sensorycznie (Overstimulated).')
    else:
        st.success('Osoba NIE jest przeciążona sensorycznie (Not Overstimulated).')

# -------- Routing na podstrony --------
if page == "Eksploracja danych":
    show_exploration(data)
elif page == "Predykcja przeciążenia":
    show_prediction(data)
