import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import re

# ---------------- LOAD DATA ----------------
data = pd.read_csv("resume_dataset_2.csv")
tfidf = joblib.load("models/tfidf.pkl")

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

data['combined'] = data['University'] + " " + data['Graduation_Year'].astype(str)
data['Cleaned'] = data['combined'].apply(clean_text)

# ---------------- FUNCTION ----------------
def get_top_candidates(jd, top_n=5):
    jd = clean_text(jd)

    jd_vec = tfidf.transform([jd])
    resume_vecs = tfidf.transform(data['Cleaned'])

    scores = cosine_similarity(jd_vec, resume_vecs)[0]
    top_idx = scores.argsort()[-top_n:][::-1]

    results = data.iloc[top_idx][['Name', 'Job_Role', 'Skills', 'Years_Experience']].copy()
    results['Match_Score'] = scores[top_idx]

    return results

# ---------------- STREAMLIT UI ----------------
st.title("📄 Smart Resume Screener")

st.write("Enter Job Description below 👇")

jd_input = st.text_area("Job Description")

top_n = st.slider("Number of candidates", 1, 10, 5)

# button
if st.button("Find Candidates"):
    if jd_input.strip() == "":
        st.warning("Please enter job description!")
    else:
        results = get_top_candidates(jd_input, top_n)

        st.subheader("Top Matching Candidates 🔥")
        st.dataframe(results)