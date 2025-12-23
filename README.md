# 🤖 AI-Powered Customer Retention Assistant

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Library-orange?logo=scikitlearn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![DistilBERT](https://img.shields.io/badge/DistilBERT-Transformer%20Model-9cf?logo=transformer&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-🤗-ffcc00?logo=huggingface&logoColor=black)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-000000?logo=flask&logoColor=white)
![HTML](https://img.shields.io/badge/HTML5-Frontend-E34F26?logo=html5&logoColor=white)
![CSS](https://img.shields.io/badge/CSS3-Styling-1572B6?logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-Frontend-F7DF1E?logo=javascript&logoColor=black)
![License](https://img.shields.io/badge/License-BSD%203--Clause-blue)


---

## 📌 Project Overview

**AI-Powered Customer Retention Assistant** is an interactive chatbot that combines:

- Machine Learning (ML) models
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)

to allow users to **ask natural language questions about customer churn drivers**, explore insights from the dataset, and receive explainable responses — *all through a human-like conversational interface*. 

This tool helps data analysts and business users interpret churn patterns without writing code.

---

## 🧠 Problem Statement

Telecom and subscription-based companies face challenges such as:

- Understanding **why customers churn**
- Exploring churn metrics without deep technical skills
- Communicating churn insights to non-technical stakeholders

This project aims to address these by providing a **conversational assistant** that interprets churn data using natural language understanding and explainable analytics. 

---

## 🎯 Objectives

1. Build an AI assistant that lets users ask questions like:
   - “What’s the overall churn rate?”
   - “Which demographic has the highest churn?”
   - “How does a service feature affect churn?”
2. Support natural language explanations for churn insights.
3. Deploy the assistant as an interactive web application using Flask. :contentReference[oaicite:3]{index=3}

---

## 📁 Repository Structure

  
    ├── Customer Retention Notebook.ipynb   # Data exploration + initial prototypes
    ├── build_index.py                      # Builds RAG retrieval index
    ├── chat_app.py                         # Main chatbot backend
    ├── explanation.py                      # Explanation utilities
    ├── gradio.py                           # (Optional) Gradio UI code
    ├── index.html | script.js | style.css  # Frontend UI assets
    ├── requirements.txt                    # Dependencies
    ├── LICENSE                             # BSD-3-Clause License
    └── README.md                           # This file

## 🛠️ Development Procedure
 
### Chatbot Experimentation

- Initially tried PandasAI with OpenAI integration → faced dependency & API issues.

- Explored Hugging Face Transformers (DistilBERT Q&A) to allow natural language queries.

- Realized general Q&A struggled with structured dataset questions → shifted to direct statistical calculations with Pandas for churn-related queries.

## 🚀 Deployment with Flask

- Built a Flask web application to host the chatbot.

- Created routes:

  - / → loads homepage (chat UI).

  - /ask → processes user queries and returns responses.

- Implemented logic to handle dataset-driven questions directly (e.g., churn rate, churn by state, effect of international plan).

- Styled the chatbot interface in dark mode with a WhatsApp-style UI for an engaging user experience.

### 💬 Key Functionalities

The chatbot can currently answer:

- Overall churn statistics → “What is the churn rate?”

- Counts → “How many customers have churned?”

- Feature impact → “How does international plan affect churn?”

- Location trends → “Which state has the highest churn rate?”

- Spending behavior → “What is the average total charges for churned customers?”

### 🔍 Future Improvements

- Enhance the NLP pipeline to interpret a wider variety of queries.

- Integrate feature importance insights from ML models.

- Expand deployment to Streamlit / Docker / Cloud hosting for production use.

### ⚙️ Tech Stack

- Python (Flask, Pandas, Scikit-learn, Transformers)

- Frontend: HTML, CSS, JavaScript (WhatsApp-style UI)

- ML Models: Logistic Regression, Random Forest, XGBoost

- Deployment: Flask web server


## 🚀 How to Run Locally

1. Clone the Repository

       git clone https://github.com/kennethnyangweso/AI-Powered-Customer-Retention-Assistant.git
       cd AI-Powered-Customer-Retention-Assistant

2. Create a Virtual Environment

       python -m venv venv
       source venv/bin/activate    # macOS / Linux
       venv\Scripts\activate       # Windows

3. Install Dependencies

       pip install -r requirements.txt

4. Build the RAG Index

       python build_index.py

5. Run the Chat App

       python chat_app.py

Open a browser and navigate to http://localhost:5000


## 📸 Outcome 

  <img width="1092" height="621" alt="Screenshot (37)" src="https://github.com/user-attachments/assets/ca9b9297-29db-4f2c-8087-edef82326972" />

## **👤 Author**

**Kenneth Nyangweso**

**Data Scientist | Electrical & Telecommunications Engineer**
