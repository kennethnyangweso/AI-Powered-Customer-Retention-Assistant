# AI-Powered-Customer-Retention-Assistant

## 🏢  **Business Understanding**

Customer churn (when customers stop using a service) is one of the biggest challenges in the telecom industry. Companies lose revenue not only from lost customers but also from the high cost of acquiring new ones.

Traditional churn prediction models (Random Forest, XGBoost, etc.) can flag customers at risk, but business teams often struggle to interpret these predictions and take action.

This project integrates Machine Learning + Large Language Models (LLMs) + Retrieval-Augmented Generation (RAG) to create a chatbot assistant that helps business teams: 

Understand churn drivers in plain English.

Ask questions about the dataset and churn patterns.

Generate insights that can guide retention strategies.

## ❓ Problem Statement

Telecom companies lack an interactive, explainable, and accessible tool that allows non-technical stakeholders (marketing, customer success, product teams) to:

- Understand why customers churn.
- Explore the dataset through natural language queries.
- Use model insights to improve retention campaigns

🎯 Objectives

1. Build a churn prediction model (baseline ML models: Random Forest, XGBoost).

2. Explain churn predictions using explainability techniques (SHAP).

3. Develop a RAG-powered chatbot that allows users to:

- Ask questions about churn patterns and trends.

- Get natural language explanations for customer churn risk.

- Summarize dataset insights for decision making.

4. Deploy the chatbot as an interactive web application (Streamlit, Gradio, or Flask).

## 📊 Metrics of Success

1. **Explainability Metrics:**

- Quality of explanations measured by SHAP feature contributions.

2. **LLM Chatbot Success Metrics:**

- Response Relevance: % of queries where responses align with dataset facts.

- User Satisfaction: Feedback from test users (e.g., 1–5 rating on clarity).

- Query Coverage: Ability of chatbot to answer at least 80% of user queries about churn.
