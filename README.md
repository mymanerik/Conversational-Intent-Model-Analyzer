Conversational Intent Model Analyzer
This project is a professional-grade conversion of an NDA project completed recently into a live working GitHub ready demo. It is an interactive web application that directly addresses evaluating and benchmarking leading AI models for the task of customer intent classification.

This application showcases a hands-on ability to analyze AI-powered customer interactions, interpret consumer intent, and leverage insights... to refine AI intent models.

🚀 Live Demo

<<<nextup>>>

🎯 Project Purpose
The goal of this project is to demonstrate a sophisticated, hands-on understanding of the AI Data Analyst role by:

Evaluating AI Models for Intent Recognition: The application makes real-time API calls to OpenAI (GPT-5), Google (Gemini 2.5 Pro), and Anthropic (Claude 3.5 Sonnet). This simulates the process of "assessing model launch success" and selecting the best tool for accurately interpreting consumer intent.

Analyzing Interaction Data: A dashboard visualizes a sample dataset of customer messages, demonstrating the ability to "identify trends to enhance system performance."

Simulating Model Optimization: By comparing how different models handle the same input, the application provides the foundational data needed to "refine AI intent models, response selection, and decision pathways."

Building a Professional Tool: The application is built with a secure, scalable approach, requiring users to input their own API keys—a standard practice in real-world AI development.

✨ Key Features
Live Multi-API Integration: Perform intent classification using real-time calls to the latest models from OpenAI, Google, and Anthropic.

Model Benchmarking Interface: Directly compare how different state-of-the-art models classify the same customer message.

Searchable Model Catalog: An organized, searchable dropdown shows awareness of the broader AI ecosystem, with a focus on the three fully integrated providers.

Secure, Multi-Key Input: A sidebar allows users to securely input their API keys for each service.

Interaction Data Dashboard: A bar chart visualizes the frequency of different intents from a sample dataset, highlighting where optimization efforts should be focused.

🛠️ How to Run
This application requires API keys from OpenAI, Google, and Anthropic to be fully functional.

Prerequisites:

Python 3.8+

An OpenAI API Key: Get one from platform.openai.com

A Google API Key: Get one from aistudio.google.com

An Anthropic API Key: Get one from console.anthropic.com

Setup Instructions:

Clone the repository:

git clone [Your GitHub Repo URL]
cd [Your Repo Name]

Install the required packages:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

Once the app is running, enter your API keys in the sidebar to activate the intent classification feature for the integrated models.

💡 How This Project Aligns with an AI Data Analyst Role
This project directly demonstrates the qualifications required for an AI Data Analyst role:

AI Data Analysis & Intent Modeling: The core function of the app is to "analyze AI-powered customer interactions... ensuring AI models accurately interpret consumer intent."

Model Optimization & Continuous Improvement: The app is a tool for "model evaluation." By highlighting misclassifications, it provides the exact data needed to "implement updates to conversational AI."

Data Curation & Annotation: While not creating guidelines, it shows an understanding of the result of good data. Consistent misclassifications from a model would signal to an analyst that the "training data needs to be refined."

Data Analysis Expertise: It uses pandas and Plotly to structure and visualize a dataset, a key skill for the role.

By building a tool that allows for the direct comparison and analysis of different AI models, this project demonstrates a proactive, data-driven approach to enhancing the quality of AI-powered conversations.
