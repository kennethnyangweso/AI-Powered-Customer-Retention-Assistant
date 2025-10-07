# AI-Powered-Customer-Retention-Assistant

## 🚀 Project Overview

This project is an AI-powered chatbot built to answer customer churn-related questions using the Syriatel Churn Dataset. It helps businesses and analysts interact with churn data in a conversational way, similar to how one would ask a human data analyst.

The chatbot was developed step by step, starting from data preprocessing, moving into machine learning modeling, and finally deployed with a Flask web application styled like a messaging app.

## ❓ Problem Statement

Telecom companies lack an interactive, explainable, and accessible tool that allows non-technical stakeholders (marketing, customer success, product teams) to:

- Understand why customers churn.
- Explore the dataset through natural language queries.
- Use model insights to improve retention campaigns

## 🎯 Objectives

1. Develop a RAG-powered chatbot that allows users to:

- Ask questions about churn patterns and trends.

- Get natural language explanations for customer churn risk.

- Summarize dataset insights for decision making.

2. Deploy the chatbot as an interactive web application (Flask).

## 🛠️ Development Procedure

### Chatbot Experimentation

- Initially tried PandasAI with OpenAI integration → faced dependency & API issues.

- Explored Hugging Face Transformers (DistilBERT Q&A) to allow natural language queries.

- Realized general Q&A struggled with structured dataset questions → shifted to direct statistical calculations with Pandas for churn-related queries.

### Deployment with Flask

- Built a Flask web application to host the chatbot.

- Created routes:

  - / → loads homepage (chat UI).

  - /ask → processes user queries and returns responses.

- Implemented logic to handle dataset-driven questions directly (e.g., churn rate, churn by state, effect of international plan).

- Styled the chatbot interface in dark mode with a WhatsApp-style UI for an engaging user experience.

### Key Functionalities

The chatbot can currently answer:

- Overall churn statistics → “What is the churn rate?”

- Counts → “How many customers have churned?”

- Feature impact → “How does international plan affect churn?”

- Location trends → “Which state has the highest churn rate?”

- Spending behavior → “What is the average total charges for churned customers?”

### Future Improvements

- Enhance the NLP pipeline to interpret a wider variety of queries.

- Integrate feature importance insights from ML models.

- Expand deployment to Streamlit / Docker / Cloud hosting for production use.

### ⚙️ Tech Stack

- Python (Flask, Pandas, Scikit-learn, Transformers)

- Frontend: HTML, CSS, JavaScript (WhatsApp-style UI)

- ML Models: Logistic Regression, Random Forest, XGBoost

- Deployment: Flask web server


## 📸 Outcome 

  <img width="1092" height="621" alt="Screenshot (37)" src="https://github.com/user-attachments/assets/ca9b9297-29db-4f2c-8087-edef82326972" />

  

