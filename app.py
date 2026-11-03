
import ast
import joblib
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

# Page settings

st.set_page_config(
    page_title="SkillPath AI",
    page_icon="🎯",
    layout="wide"
)

# Professional SkillPath AI styling
st.markdown("""
<style>
    /* Main application background */
    .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #111827 55%, #17152d 100%);
        color: #f3f4f6;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Typography */
    h1 {
        font-size: 2.7rem !important;
        font-weight: 800 !important;
        letter-spacing: -1.2px;
        color: #f8fafc !important;
    }

    h2, h3 {
        color: #e5e7eb !important;
        letter-spacing: -0.4px;
    }

    p, label, .stCaption {
        color: #cbd5e1;
    }

    /* Form controls */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
    }

    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {
        border-color: #818cf8;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid #6366f1;
        padding: 0.65rem 1.1rem;
        font-weight: 650;
        transition: all 0.2s ease;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        min-height: 3rem;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #a5b4fc;
        transform: translateY(-1px);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.72);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.1rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #a5b4fc;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    /* Data tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #334155;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Dividers */
    hr {
        border-color: #334155;
    }

    /* Sidebar, if used later */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div style="padding: 28px 30px; border-radius: 20px; background: linear-gradient(120deg, #1e1b4b, #172554, #1e293b); border: 1px solid #3730a3; margin-bottom: 24px;">
    <div style="color: #a5b4fc; font-size: 13px; font-weight: 700; letter-spacing: 2px; margin-bottom: 12px;">AI-POWERED CAREER COMPANION</div>
    <div style="color: #ffffff; font-size: 38px; font-weight: 800; line-height: 1.2;">SkillPath AI 🎯</div>
    <div style="color: #c7d2fe; font-size: 19px; margin-top: 10px;">Discover the skills your future career needs.</div>
    <div style="color: #cbd5e1; font-size: 14px; margin-top: 12px; line-height: 1.7;">Explore job-skill patterns, identify skills to develop, and build your career roadmap using data-driven recommendations.</div>
</div>
""", unsafe_allow_html=True)

st.caption(
    "A data-driven portfolio project | Built with Python, Pandas and Streamlit"
)



# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data_science_job_posts_2025.csv"
MODEL_PATH = BASE_DIR / "models" / "skill_model.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.joblib"
LABELS_PATH = BASE_DIR / "models" / "skill_labels.joblib"



# Load and clean dataset
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["job_title"]).copy()
    df["job_title"] = df["job_title"].str.strip().str.lower()

    def parse_skills(value):
        if pd.isna(value):
            return []

        try:
            parsed = ast.literal_eval(value)
            if not isinstance(parsed, list):
                return []
        except (ValueError, SyntaxError):
            return []

        cleaned = []
        for skill in parsed:
            skill = str(skill).strip().lower()

            if skill == "sklearn":
                skill = "scikit-learn"

            if skill and skill not in cleaned:
                cleaned.append(skill)

        return cleaned

    df["skills_clean"] = df["skills"].apply(parse_skills)
    return df


@st.cache_resource
def load_ml_assets():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    labels = joblib.load(LABELS_PATH)
    return model, vectorizer, labels




try:
    df = load_data()
    ml_model, ml_vectorizer, ml_labels = load_ml_assets()

    # Prepare role-based skill counts
    role_skill_counts = defaultdict(Counter)
    role_posting_counts = Counter()

    for _, row in df.iterrows():
        role = row["job_title"]
        skills = set(row["skills_clean"]) - {"amazon"}

        role_posting_counts[role] += 1

        for skill in skills:
            role_skill_counts[role][skill] += 1

  
    
    # Prepare skill co-occurrence counts
    skill_pairs = defaultdict(Counter)

    for skills in df["skills_clean"]:
        unique_skills = set(skills) - {"amazon"}

        
        for skill in unique_skills:
            for other_skill in unique_skills:
                if skill != other_skill:
                    skill_pairs[skill][other_skill] += 1
    # Job Market Overview
    st.subheader("Job Market Overview")

    top_roles = pd.DataFrame(
        role_posting_counts.most_common(10),
        columns=["Job Role", "Number of Postings"]
    )

    st.bar_chart(
        top_roles.set_index("Job Role")
    )
    
    
    # Top In-Demand Skills
    st.subheader("Top In-Demand Skills")

    overall_skill_counts = Counter()

    for skills in df["skills_clean"]:
        overall_skill_counts.update(set(skills) - {"amazon"})

    top_skills = pd.DataFrame(
        overall_skill_counts.most_common(10),
        columns=["Skill", "Job Postings"]
    )

    st.bar_chart(
        top_skills.set_index("Skill")
    )


    
    
    # User inputs
    st.subheader("Find Your Recommended Skills")
    st.caption(
        "Choose your target job role to discover relevant skills "
        "and build your career roadmap."
    )

    available_roles = sorted(role_posting_counts.keys())

    if not available_roles:
        st.error("No job roles available for recommendations.")
        st.stop()

    selected_role = st.selectbox(
        "Select your target job role",
        available_roles,
        index=available_roles.index("data scientist")
        if "data scientist" in available_roles else 0
    )

    selected_seniority = st.selectbox(
        "Select your seniority level",
        ["junior", "midlevel", "senior", "lead"],
        index=0
    )

    available_industries = sorted(
        df["industry"].dropna().astype(str).unique()
    )

    selected_industry = st.selectbox(
        "Select your industry",
        available_industries
    )

    all_skills = sorted(
        skill for skill in skill_pairs.keys()
        if skill != "amazon"
    )

    known_skills = st.multiselect(
        "Select the skills you already know",
        all_skills,
        help="Choose any skills you already have. Recommendations will exclude them."
    )

    top_n = st.slider(
        "Number of skills to recommend",
        min_value=3,
        max_value=10,
        value=5
    )

    if st.button("Recommend Skills", type="primary"):
        role_counts = role_skill_counts[selected_role]
        posting_count = role_posting_counts[selected_role]

        if not known_skills:
            st.info(
                "You haven't selected any existing skills. "
                "We'll use the most frequently listed skills for this role."
            )

        # Role-based fallback ranking
        role_ranked = sorted(
            role_counts.items(),
            key=lambda item: item[1],
            reverse=True
        )

        # Co-occurrence recommendations
        scores = Counter()

        for skill in known_skills:
            for candidate, count in skill_pairs.get(skill, {}).items():
                if candidate not in known_skills:
                    scores[candidate] += count

        # Keep recommendations relevant to the selected role
        relevant_skills = set(role_counts.keys())

        cooccurrence_recs = [
            (skill, score)
            for skill, score in scores.items()
            if skill in relevant_skills
        ]

        cooccurrence_recs.sort(
            key=lambda item: item[1],
            reverse=True
        )


        
        # ML-based skill recommendations
        if role_posting_counts[selected_role] >= 20:
            model_input = (
                f"{selected_role} {selected_seniority} {selected_industry}"
            )
            model_input_vector = ml_vectorizer.transform([model_input])
            skill_probabilities = ml_model.predict_proba(
                model_input_vector
            )[0]

            ml_recommendations = sorted(
                [
                    (str(skill), float(probability))
                    for skill, probability in zip(
                        ml_labels, skill_probabilities
                    )
                    if probability >= 0.30
                    and skill not in known_skills
                    and skill != "amazon"
                ],
                key=lambda item: item[1],
                reverse=True
            )[:top_n]

        else:
            ml_recommendations = []
            st.warning(
                f"Only {role_posting_counts[selected_role]} job posting(s) "
                "are available for this role. Showing fallback "
                "recommendations instead of ML predictions."
            )

        if ml_recommendations:
            recommendations = ml_recommendations
            method = "ML-based skill prediction"
        elif known_skills and cooccurrence_recs:
            recommendations = cooccurrence_recs[:top_n]
            method = "Skill co-occurrence"
        else:
            recommendations = [
                (skill, count)
                for skill, count in role_ranked
                if skill not in known_skills and skill != "amazon"
            ][:top_n]
            method = "Role-based skill frequency"
        if not recommendations:
            st.warning(
                "No new recommendations were found. "
                "Try selecting fewer existing skills."
            )
        else:
            st.success(f"Recommendations generated using {method}.")

            st.caption(
                f"Based on {posting_count} job postings for "
                f"**{selected_role.title()}** in this dataset. "
                "These are suggestions, not guaranteed job requirements."
            )

            
            # Build results table with clear score meaning.
            if method == "ML-based skill prediction":
                          
                results = pd.DataFrame(
                    [
                        {
                            "Recommended Skill": str(skill),
                            "Score": f"{float(score):.2%}",
                        }
                        for skill, score in recommendations
                    ]
                )
            else:
                results = pd.DataFrame(
                    [
                        {
                            "Recommended Skill": str(skill),
                            "Score": round(float(score), 4),
                        }
                        for skill, score in recommendations
                    ]
                )

            st.dataframe(results, use_container_width=True, hide_index=True)

            # CSV download
            csv_data = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📊 Download Recommendations (CSV)",
                data=csv_data,
                file_name="skillpath_ai_recommendations.csv",
                mime="text/csv",
                key="download_recommendations_csv",
            ) 
            # Skill Learning Roadmap
            st.subheader("Your Skill Learning Roadmap")
            st.caption(
                "Learn these recommended skills one by one "
                "to build your knowledge for the selected role."
            )

            for step, (skill, score) in enumerate(
                recommendations, start=1
            ):
                st.markdown(
                    f"**Step {step}:** Learn {str(skill).title()}"
                )


            # Create a fresh PDF canvas for this run and save it exactly once.
            pdf_buffer = BytesIO()
            pdf_canvas = canvas.Canvas(pdf_buffer, pagesize=letter)
            page_width, page_height = letter
            pdf_canvas.setTitle("SkillPath AI Recommendations")
            pdf_canvas.setFont("Helvetica-Bold", 16)
            pdf_canvas.drawString(50, page_height - 50, "SkillPath AI")
            pdf_canvas.setFont("Helvetica", 11)
            pdf_canvas.drawString(
                50, page_height - 75,
                f"Role: {selected_role.title()} | Seniority: {selected_seniority.title()}"
            )
            pdf_canvas.drawString(
                50, page_height - 93,
                f"Industry: {selected_industry} | Method: {method}"
            )
            pdf_canvas.drawString(
                50, page_height - 111,
                f"Based on {posting_count} job postings in the dataset."
            )

            y_position = page_height - 150
            pdf_canvas.setFont("Helvetica-Bold", 12)
            pdf_canvas.drawString(50, y_position, "Recommended Skill")
            pdf_canvas.drawString(360, y_position, "Score")
            y_position -= 24
            pdf_canvas.setFont("Helvetica", 11)

            for index, row in results.iterrows():
                if y_position < 55:
                    pdf_canvas.showPage()
                    pdf_canvas.setFont("Helvetica-Bold", 12)
                    pdf_canvas.drawString(50, page_height - 50, "SkillPath AI Recommendations")
                    y_position = page_height - 80
                    pdf_canvas.setFont("Helvetica", 11)
            for index, row in results.iterrows():
                if y_position < 55:
                    pdf_canvas.showPage()
                    pdf_canvas.setFont("Helvetica-Bold", 12)
                    pdf_canvas.drawString(50, page_height - 50, "SkillPath AI Recommendations")
                    y_position = page_height - 80
                    pdf_canvas.setFont("Helvetica", 11)

                skill_text = str(row["Recommended Skill"])
                score_text = str(row["Score"])
                pdf_canvas.drawString(50, y_position, skill_text[:45])
                pdf_canvas.drawString(360, y_position, score_text[:20])
                y_position -= 20

            pdf_canvas.save()
            pdf_buffer.seek(0)
            st.download_button(
                label="📄 Download Recommendations (PDF)",
                data=pdf_buffer.getvalue(),
                file_name="skillpath_ai_recommendations.pdf",
                mime="application/pdf",
                key="download_recommendations_pdf",
            )
                
    st.info(
        "ML-based scores represent the model's estimated probabilities "
        "for each skill. Other recommendation methods use dataset counts, "
        "so their scores are not directly comparable."
    )

    # Dataset information
    st.divider()
    
    st.subheader("📊 Dataset Overview")
    st.write("A quick summary of the job-posting data used by SkillPath AI.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="💼 Job Postings",
            value=f"{len(df):,}"
        )

    with col2:
        st.metric(
            label="🎯 Job Roles",
            value=f"{df['job_title'].nunique():,}"
        )

    with col3:
        unique_skill_count = len({
            skill
            for skills in df["skills_clean"]
            for skill in skills
            if skill != "amazon"
        })

        st.metric(
            label="🧠 Unique Skills",
            value=f"{unique_skill_count:,}"
        )

    st.caption(
        "Statistics are calculated from the dataset loaded by the application."
    )
    st.caption(
        "Recommendations reflect the supplied dataset and may not represent "
        "the entire job market."
    )

except FileNotFoundError:
    st.error(
        "Dataset not found. Make sure the CSV file is inside the "
        "'data' folder and is named 'data_science_job_posts_2025.csv'."
    )

except Exception as error:
    st.error(f"Something went wrong: {error}")
    st.caption(
        "Check the dataset file and column names, then try again."
    )

# Model Evaluation and Comparison
st.divider()
st.header("📊 Model Evaluation & Comparison")

comparison_path = BASE_DIR / "model_comparison_results.csv"

st.subheader("📊 Model Evaluation & Comparison")

if comparison_path.exists():
    comparison_df = pd.read_csv(comparison_path)

    st.subheader("📈 Performance Comparison")
    st.dataframe(comparison_df, use_container_width=True)

    if "Model" in comparison_df.columns and "Hit Rate@5 (%)" in comparison_df.columns:
        chart_df = comparison_df.set_index("Model")[["Hit Rate@5 (%)"]]
        st.bar_chart(chart_df)

else:
    st.info(
        "Model comparison results are not available yet. "
        "Model evaluation can be added after generating the evaluation results."
    )

    st.caption(
        "Hit Rate@5 measures how often a hidden skill appeared in the "
        "top five recommendations during our evaluation. It does not "
        "measure job placement or guarantee recommendation quality."
    )
else:
    st.warning(
        "Model comparison file not found. Check that "
        "'model_comparison_results.csv' is in the project folder."
    )

evaluation_path = BASE_DIR / "evaluation_results.csv"

if evaluation_path.exists():
    with st.expander("View Detailed Evaluation Results"):
        evaluation_df = pd.read_csv(evaluation_path)
        st.dataframe(evaluation_df, use_container_width=True)
