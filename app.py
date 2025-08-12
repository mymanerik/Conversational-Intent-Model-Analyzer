# app.py
# Conversational Intent Model Analyzer
# This version benchmarks leading AI models (OpenAI, Google, Anthropic) for the
# core task of customer intent classification and uses a live Firestore database.

import streamlit as st
import pandas as pd
import plotly.express as px
import openai
import google.generativeai as genai
import anthropic
from google.cloud import firestore
from google.oauth2 import service_account
from datetime import datetime
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Conversational Intent Model Analyzer",
    page_icon="🔬",
    layout="wide"
)

# --- FIRESTORE DATABASE CONNECTION ---
# This uses Streamlit's secrets management for secure authentication in TOML format.
try:
    # Get credentials from Streamlit secrets (which parses TOML into a dict)
    creds_dict = st.secrets["firestore_credentials"]
    creds = service_account.Credentials.from_service_account_info(creds_dict)
    # Explicitly pass the project ID and the correct database ID
    db = firestore.Client(credentials=creds, project=creds_dict['project_id'], database="cima")

    # Safely get a unique identifier for this app instance
    app_id = os.environ.get('__app_id', 'default-app-id')
    submissions_ref = db.collection(f"artifacts/{app_id}/public/data/submissions")
except Exception as e:
    st.error(f"Could not connect to Firestore. Please ensure your secrets are in the correct TOML format. Error: {e}")
    st.stop()


# --- STATIC DATA ---
# Separated lists for active and future models.
INTEGRATED_MODELS = [
    "OpenAI: gpt-4o",
    "Google: Gemini 2.5 Pro",
    "Anthropic: Claude 3.5 Sonnet",
]

FUTURE_MODELS = [
    "AI21 Labs: Jurassic-1 Grande", "AI21 Labs: Jurassic-1 Jumbo", "AI21 Labs: Jurassic-1 Large",
    "AI21 Labs: Jurassic-2 Light", "AI21 Labs: Jurassic-2 Mid", "AI21 Labs: Jurassic-2 Ultra",
    "Alibaba Cloud (Qwen): qwen2.5-coder-32b-instruct", "Alibaba Cloud (Qwen): qwen2.5-72b-instruct",
    "Alibaba Cloud (Qwen): qwen2-vl-72b-instruct", "Alibaba Cloud (Qwen): Qwen3 235B A22B Instruct",
    "Alibaba Cloud (Qwen): Qwen3-Coder 480B A35B Instruct", "Alibaba Cloud (Qwen): qwen3-32b",
    "Anthropic: Claude 2.0", "Anthropic: Claude 2.1", "Anthropic: Claude 3 Haiku",
    "Anthropic: Claude 3 Opus", "Anthropic: Claude 3 Sonnet", "Anthropic: Claude Instant 1.2",
    "Cohere: Command", "Cohere: Command Light", "Cohere: Command Nightly", "Cohere: Command R",
    "Cohere: Command R+", "Cohere: Generate", "DeepSeek: DeepSeek-Coder-33B",
    "DeepSeek: DeepSeek-LLM-67B", "DeepSeek: DeepSeek-V2", "Google: Gemini 1.0 Pro",
    "Google: Gemini 1.5 Flash", "Google: Gemini 1.5 Pro", "Google: Gemini 2.0 Flash",
    "Google: Gemini 2.0 Flash-Lite", "Google: Gemini 2.0 Pro Experimental", "Google: Gemini 2.5 Flash",
    "Google: Gemini 2.5 Flash-Lite", "Hugging Face (Specialized): SamLowe/roberta-base-go_emotions",
    "Hugging Face (Specialized): yorickvp/llava-13b", "IBM: granite-3.3-8b-instruct",
    "Inflection AI: Inflection-3 Pi", "Inflection AI: Inflection-3 Productivity",
    "LG AI Research: EXAONE 3.5 32B Instruct", "Mistral AI: Mistral 7B", "Mistral AI: Mistral Large 2",
    "Mistral AI: Mistral Small 2", "Mistral AI: Mixtral 8x22B", "Mistral AI: Mixtral 8x7B",
    "Moonshot AI: Kimi K2 Instruct", "OpenAI (Open-Weight): gpt-oss-120b", "OpenAI (Open-Weight): gpt-oss-20b",
    "OpenAI: gpt-5", "OpenAI: gpt-5-mini", "OpenAI: gpt-5-nano", "OpenAI: gpt-4", "OpenAI: gpt-4-turbo",
    "OpenAI: gpt-4.1", "OpenAI: gpt-4.1-mini", "OpenAI: gpt-4.1-nano",
    "Perplexity AI: sonar", "Perplexity AI: sonar deep research", "Perplexity AI: sonar pro",
    "Perplexity AI: sonar reasoning", "Perplexity AI: sonar reasoning pro", "xAI: Grok 4 Heavy",
    "xAI: Grok 4", "xAI: Grok 1.5", "xAI: Grok 1.5 Vision", "xAI: Grok 1", "Yi (01.AI): yi-large",
    "Zhipu AI: GLM-4.5-Air"
]

INTENT_CATEGORIES = ["Billing Inquiry", "Cancellation Request", "Technical Support", "Product Information", "Positive Feedback", "Negative Feedback", "Unclassified"]
SYSTEM_PROMPT_INTENT = f"""
You are an expert intent classification system for a customer service bot. Your task is to analyze the user's message and classify it into ONE of the following predefined categories: {', '.join(INTENT_CATEGORIES)}.
Respond with ONLY the category name and nothing else.
"""

# --- API CALL FUNCTIONS ---
def analyze_with_openai(api_key, user_message):
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "system", "content": SYSTEM_PROMPT_INTENT}, {"role": "user", "content": user_message}],
            temperature=0.0, max_tokens=15
        )
        intent = response.choices[0].message.content.strip()
        return intent if intent in INTENT_CATEGORIES else "Unclassified"
    except Exception as e:
        st.error("That key is NOT for the model you chose; or the key has expired")
        return None

def analyze_with_google(api_key, user_message):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
        full_prompt = f"{SYSTEM_PROMPT_INTENT}\n\nUser Message: \"{user_message}\""
        response = model.generate_content(full_prompt)
        intent = response.text.strip()
        return intent if intent in INTENT_CATEGORIES else "Unclassified"
    except Exception as e:
        st.error("That key is NOT for the model you chose; or the key has expired")
        return None

def analyze_with_anthropic(api_key, user_message):
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620", max_tokens=15, temperature=0.0,
            system=SYSTEM_PROMPT_INTENT, messages=[{"role": "user", "content": user_message}]
        )
        intent = message.content[0].text.strip()
        return intent if intent in INTENT_CATEGORIES else "Unclassified"
    except Exception as e:
        st.error("That key is NOT for the model you chose; or the key has expired")
        return None

# --- SIDEBAR FOR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Model Selection")
    st.write("Select an AI model and provide the required API key.")
    model_choice = st.selectbox("Choose AI Model:", INTEGRATED_MODELS, index=0)
    api_key = st.text_input("Enter API Key for Selected Model", type="password")
    st.markdown("""
    * [Get an OpenAI API Key](https://platform.openai.com/api-keys)
    * [Get a Google Gemini API Key](https://aistudio.google.com/app/apikey)
    * [Get an Anthropic API Key](https://console.anthropic.com)
    """)
    st.info("Your API key is not stored and is only used for the current session.")

    st.header("Future Integrations")
    st.info("The following models will be integrated into this demo at a future date.")
    st.selectbox(
        "Future Model List:",
        FUTURE_MODELS
    )

# --- MAIN APP HEADER ---
st.title("🔬 Conversational Intent Model Analyzer")
st.markdown("""
This project is a professional-grade conversion of an NDA project completed recently into a live working GitHub / Streamlit - ready demo. It is an interactive web application that directly addresses evaluating and benchmarking leading AI models for the task of customer intent classification.
""")
st.markdown("---")

# --- MAIN APPLICATION LAYOUT ---
col1, col2 = st.columns((1, 1.2))

# --- COLUMN 1: LIVE INTENT CLASSIFICATION ---
with col1:
    st.header("Live Intent Classification")
    st.write("Enter a customer message to see how different AI models interpret the intent.")
    user_input = st.text_area("Customer Message:", "I was charged twice this month for my subscription, can you please look into it?", height=100)

    if st.button("Analyze Intent", type="primary"):
        if not api_key:
            st.warning("Please enter your API key in the sidebar.")
        else:
            analyzed_intent = None
            provider = model_choice.split(':')[0]
            with st.spinner(f"Asking {provider} to classify intent..."):
                if provider == "OpenAI":
                    analyzed_intent = analyze_with_openai(api_key, user_input)
                elif provider == "Google":
                    analyzed_intent = analyze_with_google(api_key, user_input)
                elif provider == "Anthropic":
                    analyzed_intent = analyze_with_anthropic(api_key, user_input)

            if analyzed_intent:
                # Save the successful result to Firestore
                try:
                    submission_data = {
                        "timestamp": datetime.now(),
                        "model": model_choice,
                        "message": user_input,
                        "intent": analyzed_intent
                    }
                    submissions_ref.add(submission_data)
                    st.success(f"**Predicted Intent ({model_choice}):** `{analyzed_intent}`")
                    st.toast("Analysis saved to database!")
                except Exception as e:
                    st.error(f"Failed to save result to database: {e}")

                st.write("""
                **How this helps an AI Data Analyst:** An analyst performs these head-to-head comparisons to select the best model for a specific use case. If a model consistently misclassifies an intent (e.g., confusing a `Billing Inquiry` with a `Cancellation Request`), it signals a need to refine the training data or adjust the model's decision pathways. This data-driven selection process is key to optimizing conversational AI.
                """)

# --- COLUMN 2: LIVE DATA ANALYSIS DASHBOARD ---
with col2:
    # Load data from Firestore
    try:
        docs = submissions_ref.stream()
        submissions = [doc.to_dict() for doc in docs]
        n_submissions = len(submissions)
    except Exception as e:
        st.error(f"Could not load data from Firestore: {e}")
        submissions = []
        n_submissions = 0

    st.header(f"Customer Interaction Data Analysis based on {n_submissions} Submissions")
    
    if n_submissions > 0:
        df = pd.DataFrame(submissions)
        st.subheader("Distribution of Customer Intents")
        intent_counts = df['intent'].value_counts().reset_index()
        intent_counts.columns = ['Intent', 'Count']
        
        fig = px.bar(
            intent_counts, x='Intent', y='Count', color='Intent',
            title='Frequency of Each Customer Intent',
            labels={'Count': 'Number of Interactions', 'Intent': 'Classified Intent'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        st.write("""
        **How this helps an AI Data Analyst:** This analysis helps prioritize work. An analyst would focus on improving the models for the most frequent intents (`Billing Inquiry`, `Technical Support`) to achieve the largest impact on overall system performance and customer satisfaction. This directly relates to **identifying trends to enhance system performance.**
        """)
    else:
        st.info("No submissions yet. Analyze a message to see the data populate here!")


st.markdown("---")
# Updated footer with markdown links - re-typed to ensure no hidden characters
st.markdown("[Conversational Intent Model Analyzer](https://github.com/mymanerik/Conversational-Intent-Model-Analyzer/tree/master) | [🌐Erik Malson](https://Erik.ml) / [@MyManErik](https://instagram.com/mymanerik/) | [@AIinTheAM](https://YouTube.com/@AIinTheAm)")
