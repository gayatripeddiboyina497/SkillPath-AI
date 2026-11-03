# SkillPath AI 🎯
### AI-Powered Career Companion

**Discover the skills your future career needs.**

SkillPath AI is a data-driven career guidance application that analyzes job market trends, identifies in-demand skills, predicts relevant skills using Machine Learning, and helps users build a career learning roadmap.

## 🚀 Features

- **Job Market Overview:** Explore job roles and job market patterns.
- **In-Demand Skills Analysis:** Identify skills frequently required by employers.
- **AI-Based Skill Prediction:** Get relevant skill recommendations using a trained Machine Learning model.
- **Career Learning Roadmap:** Discover skills to focus on for career development.
- **Data Export:** Download results in CSV and PDF formats.
- **Model Evaluation:** View model comparison and evaluation results.

## 🛠️ Technologies Used

- Python
- Pandas
- Streamlit
- Scikit-learn
- Machine Learning
- Joblib
- Matplotlib / Streamlit Charts

## 📂 Project Structure

```text
SkillPath-AI/
│
├── app.py
├── data_science_job_posts_2025.csv
├── model_comparison_results.csv
├── requirements.txt
├── README.md
│
└── models/
    ├── skill_model.joblib
    ├── tfidf_vectorizer.joblib
    └── skill_labels.joblib
```

## 📊 Dataset

The project uses a job-postings dataset to explore job roles and the skills associated with them.

## 🧠 Machine Learning

A trained Machine Learning model is used to generate skill recommendations based on the application input.

Model evaluation and comparison results are displayed in the application when the evaluation results file is available.

## ▶️ Run Locally

1. Clone the repository:

```bash
git clone https://github.com/gayatripeddiboyina497/SkillPath-AI.git
```

2. Open the project folder:

```bash
cd SkillPath-AI
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
streamlit run app.py
```

## 🌐 Live Application

[Open SkillPath AI](https://skillpath-ai-gayatri.streamlit.app/)

## 👩‍💻 Developer

**Gayatri Peddiboyina**

Data Science Student | Aspiring Data Science Professional

[GitHub Profile](https://github.com/gayatripeddiboyina497)
