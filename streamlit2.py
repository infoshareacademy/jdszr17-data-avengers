import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Overstimulation App", layout="wide")

st.title("Overstimulation Detection App")
st.write("Wprowadź dane dotyczące stylu życia, aby przewidzieć, czy osoba jest przeciążona sensorycznie.")

# Próba wczytania danych
try:
    data = pd.read_csv('overstimulation_dataset.csv')
except FileNotFoundError:
    st.error("❌ Nie znaleziono pliku 'overstimulation_dataset.csv'. Upewnij się, że plik znajduje się w tym samym folderze co ten skrypt.")
    st.stop()

# Sekcja: Eksploracja danych
with st.expander("📊 Eksploracja danych (wizualizacje)"):
    st.subheader("1. Proporcje osób przeciążonych sensorycznie")
    overstim_counts = data["Overstimulated"].value_counts()
    labels = ['Overstimulated (1)', 'Not Overstimulated (0)']
    fig1, ax1 = plt.subplots()
    ax1.pie(overstim_counts, labels=labels, autopct='%1.1f%%', colors=sns.color_palette("Set2"), startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    st.subheader("2. Rozkład zmiennych binarnych")
    binary_cols = ['Meditation_Habit', 'Multitasking_Habit']
    for col in binary_cols:
        count = data[col].value_counts()
        fig, ax = plt.subplots()
        ax.pie(count, labels=count.index, autopct='%1.1f%%', colors=sns.color_palette('viridis', len(count)).as_hex())
        ax.set_title(f'{col} distribution')
        st.pyplot(fig)

    st.subheader("3. Rozkład zmiennych kategorycznych")
    categorical_cols = ['Sensory_Sensitivity', 'Sleep_Quality', 'Noise_Exposure', 'Headache_Frequency']
    for col in categorical_cols:
        counts = data[col].value_counts().sort_index()
        fig, ax = plt.subplots()
        sns.barplot(x=counts.index, y=counts.values, ax=ax, color='lightblue')
        ax.set_title(f'{col} distribution')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        st.pyplot(fig)

    st.subheader("4. Histogramy zmiennych numerycznych")
    fig = data.hist(figsize=(12, 10), color='lightblue', bins=20, edgecolor='gray')
    plt.suptitle("Histogram of numerical columns in dataset")
    st.pyplot(plt.gcf())

    st.subheader("5. Zależność: Wiek a sen")
    fig, ax = plt.subplots()
    sns.lineplot(x='Age', y='Sleep_Hours', data=data, marker='o', ax=ax)
    ax.set_title('Sleep hours vs Age')
    st.pyplot(fig)

    st.subheader("6. Zależność: Stres a sen")
    fig, ax = plt.subplots()
    sns.lineplot(x='Stress_Level', y='Sleep_Hours', data=data, marker='o', ax=ax)
    ax.set_title('Sleep hours vs Stress level')
    st.pyplot(fig)

    st.subheader("7. Zależność: Czas przed ekranem a stres")
    fig, ax = plt.subplots()
    sns.lineplot(x='Stress_Level', y='Screen_Time', data=data, marker='o', ax=ax)
    ax.set_title('Screen time vs Stress level')
    st.pyplot(fig)

    st.subheader("8. Zależność: Przeciążenie vs czas przed ekranem i stres")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Screen_Time', y='Stress_Level', hue='Overstimulated', data=data, ax=ax)
    ax.set_title('Screen time vs overstimulation')
    st.pyplot(fig)

    st.subheader("9. Pairplot wybranych cech")
    pairplot_fig = sns.pairplot(data[['Age', 'Sleep_Hours', 'Screen_Time', 'Stress_Level', 'Overstimulated']],
                                hue='Overstimulated', palette='colorblind')
    st.pyplot(pairplot_fig.fig)

    st.subheader("10. Heatmapa korelacji")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Correlation heatmap')
    st.pyplot(fig)