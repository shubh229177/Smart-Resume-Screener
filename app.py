import streamlit as st
import pandas as pd
import joblib
import re
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Screener", layout="wide")

#  LOAD DATA 
@st.cache_data
def load_data():
    data = pd.read_csv("resume_dataset_2.csv")
    data['combined'] = data['Resume_Text'] + " " + data['Skills'].fillna("")
    data['Cleaned'] = data['combined'].apply(clean_text)
    return data

@st.cache_resource
def load_model():
    return joblib.load("models/tfidf.pkl")

#  CLEAN TEXT 
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    return text

data = load_data()
tfidf = load_model()

#  PDF PARSER 
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except:
        text = ""
    return text

#  SIDEBAR 
with st.sidebar:
    st.header("⚙️ Filters")
    top_n = st.slider("Top Candidates", 1, 10, 5)
    min_exp = st.slider("Minimum Experience", 0, 10, 0)

#  HEADER 
st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>
💼 AI Resume Screening System
</h1>
<p style='text-align: center;'>
Upload resumes & find best candidates instantly 🚀
</p>
""", unsafe_allow_html=True)

#  JD INPUT 
jd = st.text_area(
    "📄 Enter Job Description",
    height=180,
    placeholder="Example: Looking for a Machine Learning Engineer with Python, ML, Docker, NLP..."
)

#  FILE UPLOAD 
st.subheader("📂 Upload Resumes")

uploaded_files = st.file_uploader(
    "Drag & drop or browse PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

#  PROCESS UPLOAD 
extra_resumes = []

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} resumes uploaded")

    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        cleaned = clean_text(text)

        extra_resumes.append({
            "Name": file.name,
            "Job_Role": "Uploaded Resume",
            "Skills": "Extracted from PDF",
            "Years_Experience": 0,
            "Cleaned": cleaned
        })

extra_df = pd.DataFrame(extra_resumes)

# Combine
full_data = pd.concat([data, extra_df], ignore_index=True)

#  MATCH FUNCTION 
def get_top_candidates(jd):
    jd = clean_text(jd)

    jd_vec = tfidf.transform([jd])
    resume_vecs = tfidf.transform(full_data['Cleaned'])

    scores = cosine_similarity(jd_vec, resume_vecs)[0]

    df = full_data.copy()
    df['Match_Score'] = scores

    df = df.sort_values(by="Match_Score", ascending=False)
    df = df.groupby("Name").first().reset_index()

    return df

#  BUTTON 
if st.button("🚀 Find Best Candidates"):
    if jd.strip() == "":
        st.warning("⚠️ Please enter a Job Description")
    else:
        with st.spinner("Analyzing resumes..."):
            results = get_top_candidates(jd)

            results = results[results['Years_Experience'] >= min_exp]
            results = results.head(top_n)

        st.subheader("🏆 Top Matches")

        if len(results) == 0:
            st.error("No matching candidates found 😢")
        else:
            for _, row in results.iterrows():
                st.markdown(f"""
                <div style="
                    border-radius: 12px;
                    padding: 15px;
                    margin-bottom: 12px;
                    background-color: #1e1e1e;
                    color: white;
                ">
                    <h4>👤 {row['Name']}</h4>
                    <p><b>Role:</b> {row['Job_Role']}</p>
                    <p><b>Experience:</b> {row['Years_Experience']} years</p>
                    <p><b>Skills:</b> {row['Skills']}</p>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns([4,1])

                with col2:
                    st.metric("Score", f"{row['Match_Score']:.2f}")
                    st.progress(float(row['Match_Score']))

                # Tag uploaded resume
                if ".pdf" in row['Name']:
                    st.caption("📄 Uploaded Resume")

                st.divider()

        #  DOWNLOAD 
        csv = results.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name="top_candidates.csv",
            mime="text/csv",
        )