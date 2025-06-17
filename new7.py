import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Overstimulation App", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Glossary", "Data Exploration", "Overstimulation Prediction"],
    index=1  # domyślnie Data Exploration
)

@st.cache_data
def load_data():
    try:
        data = pd.read_csv('overstimulation_dataset.csv')
    except FileNotFoundError:
        st.error("❌ File 'overstimulation_dataset.csv' not found. Please make sure the file is in the same folder as this script.")
        st.stop()
    return data

data = load_data()

@st.cache_resource
def train_model(data):
    X = data.drop('Overstimulated', axis=1)
    y = data['Overstimulated']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model_lr = LogisticRegression(max_iter=1000, random_state=42)
    model_lr.fit(X_scaled, y)
    return scaler, model_lr

def show_glossary():
    st.title("📖 Glossary")
    glossary = {
        "Age": "Age of the individual in years.",
        "Sleep_Hours": "Average number of sleep hours per night.",
        "Screen_Time": "Average daily hours spent looking at screens.",
        "Stress_Level": "Self-reported stress level (1 - low, 10 - high).",
        "Noise_Exposure": "Exposure level to noise on a scale (0 - none, 5 - high).",
        "Social_Interaction": "Number of social interactions per day.",
        "Work_Hours": "Daily hours spent working.",
        "Exercise_Hours": "Daily hours spent exercising.",
        "Caffeine_Intake": "Daily number of caffeine cups consumed.",
        "Multitasking_Habit": "Whether multitasking is a common habit (0 - no, 1 - yes).",
        "Anxiety_Score": "Self-reported anxiety level (1 - low, 10 - high).",
        "Depression_Score": "Self-reported depression level (1 - low, 10 - high).",
        "Sensory_Sensitivity": "Sensitivity to sensory stimuli (0 - low, 4 - high).",
        "Meditation_Habit": "Whether meditation is regularly practiced (0 - no, 1 - yes).",
        "Overthinking_Score": "Frequency of overthinking (1 - low, 10 - high).",
        "Irritability_Score": "Frequency of irritability (1 - low, 10 - high).",
        "Headache_Frequency": "Number of headaches experienced per week.",
        "Sleep_Quality": "Quality of sleep rated (1 - poor, 4 - excellent).",
        "Tech_Usage_Hours": "Hours of daily technology use.",
        "Overstimulated": "Indicates if the person is overstimulated (0 - no, 1 - yes)."
    }
    for term, description in glossary.items():
        st.markdown(f"**{term}**: {description}")

def show_exploration(data):
    st.title("📊 Data Exploration")
    with st.expander("1. Proportion of overstimulated people"):
        overstim_counts = data["Overstimulated"].value_counts()
        labels = ['Overstimulated (1)', 'Not Overstimulated (0)']
        fig1, ax1 = plt.subplots(figsize=(3.75, 3.75))
        ax1.pie(overstim_counts, labels=labels, autopct='%1.1f%%', colors=sns.color_palette("Set2"), startangle=90)
        ax1.axis('equal')
        ax1.set_title('Proportion of overstimulated people', fontsize=18)
        st.pyplot(fig1, use_container_width=False)

    with st.expander("2. Distribution of binary variables"):
        binary_cols = ['Meditation_Habit', 'Multitasking_Habit']
        for col in binary_cols:
            count = data[col].value_counts()
            fig, ax = plt.subplots(figsize=(3.75, 3.75))
            ax.pie(count, labels=[str(x) for x in count.index], autopct='%1.1f%%', colors=sns.color_palette('viridis', len(count)).as_hex())
            ax.set_title(f'{col} distribution', fontsize=18)
            st.pyplot(fig, use_container_width=False)

    with st.expander("3. Distribution of categorical variables"):
        categorical_cols = ['Sensory_Sensitivity', 'Sleep_Quality', 'Noise_Exposure', 'Headache_Frequency']
        for col in categorical_cols:
            counts = data[col].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(5, 3.75))
            sns.barplot(x=counts.index, y=counts.values, ax=ax, color='lightblue')
            ax.set_title(f'{col} distribution', fontsize=18)
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            st.pyplot(fig, use_container_width=False)

    # 4. Histogram powiększony o 20% (względem poprzednich wartości)
    with st.expander("4. Histograms of numerical variables"):
        fig = data.hist(figsize=(17.11, 12.40), color='lightblue', bins=20, edgecolor='gray')
        plt.suptitle("Histogram of numerical columns in dataset", fontsize=22)
        st.pyplot(plt.gcf(), use_container_width=False)

    with st.expander("5. Sleep hours vs Age"):
        fig, ax = plt.subplots(figsize=(5, 3.75))
        sns.lineplot(x='Age', y='Sleep_Hours', data=data, marker='o', ax=ax)
        ax.set_title('Sleep hours vs Age', fontsize=18)
        st.pyplot(fig, use_container_width=False)

    with st.expander("6. Sleep hours vs Stress level"):
        fig, ax = plt.subplots(figsize=(5, 3.75))
        sns.lineplot(x='Stress_Level', y='Sleep_Hours', data=data, marker='o', ax=ax)
        ax.set_title('Sleep hours vs Stress level', fontsize=18)
        st.pyplot(fig, use_container_width=False)

    with st.expander("7. Screen time vs Stress level"):
        fig, ax = plt.subplots(figsize=(5, 3.75))
        sns.lineplot(x='Stress_Level', y='Screen_Time', data=data, marker='o', ax=ax)
        ax.set_title('Screen time vs Stress level', fontsize=18)
        st.pyplot(fig, use_container_width=False)

    # 8. Powiększony wykres scatterplot (o 25%)
    with st.expander("8. Overstimulation vs Screen time and Stress"):
        fig, ax = plt.subplots(figsize=(6.25, 4.6875))
        sns.scatterplot(x='Screen_Time', y='Stress_Level', hue='Overstimulated', data=data, ax=ax)
        ax.set_title('Screen time vs Overstimulation', fontsize=18)
        st.pyplot(fig, use_container_width=False)

    with st.expander("9. Pairplot of selected features"):
        pairplot_fig = sns.pairplot(
            data[['Age', 'Sleep_Hours', 'Screen_Time', 'Stress_Level', 'Overstimulated']],
            hue='Overstimulated', palette='colorblind', plot_kws={'s': 30}
        )
        pairplot_fig.fig.set_size_inches(7.5, 7.5)
        pairplot_fig.fig.suptitle("Pairplot of selected features", fontsize=22, y=1.02)
        st.pyplot(pairplot_fig.fig, use_container_width=False)

    # 10. Heatmapa powiększona o 20%
    with st.expander("10. Correlation heatmap"):
        fig, ax = plt.subplots(figsize=(13.50, 6.75))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
        ax.set_title('Correlation heatmap', fontsize=22)
        st.pyplot(fig, use_container_width=False)

def user_input_features():
    st.header("Enter data for overstimulation prediction")
    Age = st.slider('Age', 18, 60, 30, step=1)
    Sleep_Hours = st.slider('Sleep hours', 3, 12, 7, step=1)
    Screen_Time = st.slider('Screen time (h)', 1, 16, 5, step=1)
    Stress_Level = st.slider('Stress level (1-10)', 1, 10, 5, step=1)
    Noise_Exposure = st.slider('Noise exposure (0-5)', 0, 5, 2, step=1)
    Social_Interaction = st.slider('Social interactions (per day)', 0, 20, 5, step=1)
    Work_Hours = st.slider('Work hours', 1, 16, 8, step=1)
    Exercise_Hours = st.slider('Exercise (h)', 0, 4, 1, step=1)
    Caffeine_Intake = st.slider('Caffeine intake (cups)', 0, 10, 2, step=1)
    Multitasking_Habit = st.selectbox('Multitasking habit', [0, 1])
    Anxiety_Score = st.slider('Anxiety score (1-10)', 1, 10, 5, step=1)
    Depression_Score = st.slider('Depression score (1-10)', 1, 10, 5, step=1)
    Sensory_Sensitivity = st.slider('Sensory sensitivity (0-4)', 0, 4, 2, step=1)
    Meditation_Habit = st.selectbox('Meditation habit', [0, 1])
    Overthinking_Score = st.slider('Overthinking score (1-10)', 1, 10, 5, step=1)
    Irritability_Score = st.slider('Irritability score (1-10)', 1, 10, 5, step=1)
    Headache_Frequency = st.slider('Headache frequency (per week)', 0, 7, 2, step=1)
    Sleep_Quality = st.slider('Sleep quality (1-4)', 1, 4, 3, step=1)
    Tech_Usage_Hours = st.slider('Tech usage (h)', 1, 16, 5, step=1)

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
    scaler, model_lr = train_model(data)
    st.title("🔮 Overstimulation Prediction")
    input_df = user_input_features()
    input_scaled = scaler.transform(input_df)
    prediction = model_lr.predict(input_scaled)
    st.markdown('<h2 style="font-size: 2em;">Prediction result:</h2>', unsafe_allow_html=True)
    if prediction[0] == 1:
        st.error('The person is OVERSTIMULATED.', icon="⚠️")
    else:
        st.success('The person is NOT overstimulated.', icon="✅")

if page == "Glossary":
    show_glossary()
elif page == "Data Exploration":
    show_exploration(data)
elif page == "Overstimulation Prediction":
    show_prediction(data)
