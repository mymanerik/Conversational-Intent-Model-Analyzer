# app.py
# Conversational Intent Model Analyzer
# This version benchmarks leading AI models (OpenAI, Google, Anthropic) for the
# core task of customer intent classification.

import streamlit as st
import pandas as pd
import plotly.express as px
import openai
import google.generativeai as genai
import anthropic

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Conversational Intent Model Analyzer",
    page_icon="🔬",
    layout="wide"
)

# --- STATIC DATA ---
# Separated lists for active and future models.
INTEGRATED_MODELS = [
    "OpenAI: gpt-5",
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
    "OpenAI: gpt-5-mini", "OpenAI: gpt-5-nano", "OpenAI: gpt-4", "OpenAI: gpt-4-turbo",
    "OpenAI: gpt-4.1", "OpenAI: gpt-4.1-mini", "OpenAI: gpt-4.1-nano", "OpenAI: gpt-4o",
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
            model="gpt-5", messages=[{"role": "system", "content": SYSTEM_PROMPT_INTENT}, {"role": "user", "content": user_message}],
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
    # Initialize session state for accordion behavior
    if 'active_expander' not in st.session_state:
        st.session_state.active_expander = 'Model Selection'

    # Create the expanders and check their state
    model_expander = st.expander("⚙️ Model Selection", expanded=(st.session_state.active_expander == 'Model Selection'))
    future_expander = st.expander("Future Integrations", expanded=(st.session_state.active_expander == 'Future Integrations'))

    # Logic to manage the accordion state
    if model_expander and st.session_state.active_expander != 'Model Selection':
        st.session_state.active_expander = 'Model Selection'
        st.experimental_rerun()

    if future_expander and st.session_state.active_expander != 'Future Integrations':
        st.session_state.active_expander = 'Future Integrations'
        st.experimental_rerun()

    # Place content inside the expander containers
    with model_expander:
        st.write("Select an AI model and provide the required API key.")
        model_choice = st.selectbox("Choose AI Model:", INTEGRATED_MODELS, index=0)
        api_key = st.text_input("Enter API Key for Selected Model", type="password")
        st.markdown("""
        * [Get an OpenAI API Key](https://platform.openai.com/api-keys)
        * [Get a Google Gemini API Key](https://aistudio.google.com/app/apikey)
        * [Get an Anthropic API Key](https://console.anthropic.com)
        """)
        st.info("Your API key is not stored and is only used for the current session.")

    with future_expander:
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
                st.success(f"**Predicted Intent ({model_choice}):** `{analyzed_intent}`")
                st.write("""
                **How this helps an AI Data Analyst:** An analyst performs these head-to-head comparisons to select the best model for a specific use case. If a model consistently misclassifies an intent (e.g., confusing a `Billing Inquiry` with a `Cancellation Request`), it signals a need to refine the training data or adjust the model's decision pathways. This data-driven selection process is key to optimizing conversational AI.
                """)

# --- COLUMN 2: DATA ANALYSIS DASHBOARD ---
with col2:
    st.header("Customer Interaction Data Analysis")
    st.write("This dashboard visualizes trends from a sample dataset of customer interactions, a typical starting point for analysis.")

    try:
        df = pd.read_csv('customer_interactions.csv')
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

    except FileNotFoundError:
        st.error("Error: `customer_interactions.csv` not found.")
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")

st.markdown("---")
# Updated footer with markdown links
st.markdown("[Conversational Intent Model Analyzer](https://github.com/mymanerik/Conversational-Intent-Model-Analyzer/tree/master) | [🌐Erik Malson](https://Erik.ml) / [@MyManErik](https://instagram.com/mymanerik/) | [@AIinTheAM](https://YouTube.com/@AIinTheAm)")
