import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from pathlib import Path  

st.set_page_config(
    page_title="Toxic Comment Detection API",
    page_icon="🤖",
    layout="wide"
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SAMPLED_DATA_PATH = ROOT_DIR / "data" / "train_sampled_50k.csv"

API_URL = "http://localhost:8000/predict"

st.title("Toxic Comment Detection API 🤖")
st.markdown("A Streamlit dashboard to test the deployed MLOps pipeline.")

st.header("Real-Time Prediction API")
with st.form("comment_form"):
    comment_text = st.text_area(
        "Enter a comment to classify:",
        "you are a wonderful, amazing person!"
    )
    submit_button = st.form_submit_button("Classify Comment")

if submit_button and comment_text:
    with st.spinner("Calling the API..."):
        try:
            response = requests.post(API_URL, json={"text": comment_text})
            
            if response.status_code == 200:
                prediction = response.json()
                label = prediction['label']
                is_toxic = prediction['is_toxic']
                
                if is_toxic:
                    st.error(f"**Classification: {label.upper()}**")
                else:
                    st.success(f"**Classification: {label.upper()}**")
            else:
                st.error(f"API Error (Code {response.status_code}): {response.text}")
        except requests.exceptions.ConnectionError:
            st.error(
                "Connection Error: Could not connect to the API."
                "Is the Docker container running?"
                "Run: docker run -p 8000:8000 toxic-api"
            )
        except Exception as e:
            st.error(f"An unknown error occurred: {e}")

st.header("Training Data Overview")
st.markdown(f"Visualizing the `{SAMPLED_DATA_PATH.name}` file.")

@st.cache_data
def load_data():
    if not SAMPLED_DATA_PATH.exists():
        return None
    df = pd.read_csv(SAMPLED_DATA_PATH)
    target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for col in target_columns:
        if col not in df.columns:
            df[col] = 0
            
    df['is_toxic'] = (df[target_columns].sum(axis=1) > 0).astype(int)
    return df

df_sampled = load_data()

if df_sampled is not None:
    st.dataframe(df_sampled.head())

    st.subheader("Toxicity Distribution")
    if 'is_toxic' in df_sampled.columns:
        toxic_counts = df_sampled['is_toxic'].value_counts().reset_index()
        toxic_counts.columns = ['label', 'count']
        toxic_counts['label'] = toxic_counts['label'].map({0: 'Non-Toxic', 1: 'Toxic'})
        
        fig = px.pie(
            toxic_counts,
            names='label',
            values='count',
            title='Proportion of Toxic vs. Non-Toxic Comments',
            color='label',
            color_discrete_map={'Non-Toxic': 'green', 'Toxic': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not find 'is_toxic' column to plot distribution.")

else:
    st.error(
        f"Could not load data from `{SAMPLED_DATA_PATH}`. "
        f"Please run the `src/train.py` script first to generate the sampled data file."
    )

