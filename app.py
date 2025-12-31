import streamlit as st
import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# STREAMLIT CONFIG

st.set_page_config(
    page_title="Job & CV Hybrid Compatibility Analyzer",
    layout="wide"
)

st.title("🔍 Job Description & CV Compatibility Analyzer")
st.write("TF-IDF + Transformer tabanlı hibrit CV uyumluluk analizi.")

st.divider()


# LOAD TRANSFORMER MODEL (CACHE)

@st.cache_resource
def load_semantic_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

semantic_model = load_semantic_model()


# CV PDF READER

def extract_cv_text(pdf_file) -> str:
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    return text.lower().strip()


def clean_text(text: str) -> str:
    blacklist = [
        "linkedin",
        "privacy policy",
        "terms",
        "cookie",
        "sign in",
        "log in",
        "çerez",
        "gizlilik",
        "kullanıcı anlaşması"
    ] # ekleme yapılabilir

    cleaned = text.lower()
    for bad in blacklist:
        cleaned = cleaned.replace(bad, "")

    return cleaned.strip()


# TF-IDF SCORE

def tfidf_score(job_text: str, cv_text: str) -> float:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=6000
    )

    vectors = vectorizer.fit_transform([job_text, cv_text])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    return score * 100

# SEMANTIC SCORE (TRANSFORMER)

def semantic_score(job_text: str, cv_text: str) -> float:
    embeddings = semantic_model.encode(
        [job_text, cv_text],
        normalize_embeddings=True
    )

    score = np.dot(embeddings[0], embeddings[1])
    return score * 100


# FINAL HYBRID SCORE

def hybrid_score(job_text: str, cv_text: str):
    tfidf = tfidf_score(job_text, cv_text)
    semantic = semantic_score(job_text, cv_text)

    final = (0.4 * tfidf) + (0.6 * semantic)

    return round(final, 2), round(tfidf, 2), round(semantic, 2)


# STREAMLIT UI

st.subheader("📌 İş İlanı Açıklaması")

job_description = st.text_area(
    "İş ilanı açıklamasını buraya yapıştırın",
    height=300,
    placeholder="Responsibilities, requirements, qualifications..."
)

st.subheader("📄 CV Yükle")
cv_file = st.file_uploader(
    "CV (PDF)",
    type=["pdf"]
)

analyze_btn = st.button("🚀 Analizi Başlat")

if analyze_btn:
    if not job_description or not cv_file:
        st.error("Lütfen iş ilanı açıklamasını girin ve CV yükleyin.")
        st.stop()

    with st.spinner("Metinler hazırlanıyor..."):
        job_text = clean_text(job_description)
        cv_text = extract_cv_text(cv_file)

    if len(job_text.split()) < 80:
        st.error("İş ilanı açıklaması çok kısa veya geçersiz.")
        st.stop()

    if len(cv_text.split()) < 80:
        st.error("CV metni okunamadı veya çok kısa.")
        st.stop()

    with st.spinner("Hibrit uyumluluk hesaplanıyor..."):
        final, tfidf, semantic = hybrid_score(job_text, cv_text)

    st.divider()
    st.subheader("📊 Sonuçlar")

    st.progress(int(final))
    st.metric("🎯 Final Uyumluluk Skoru", f"%{final}")

    col1, col2 = st.columns(2)
    col1.metric("📘 TF-IDF (Kelime Eşleşmesi)", f"%{tfidf}")
    col2.metric("🧠 Semantic (Anlamsal Benzerlik)", f"%{semantic}")

    if final >= 75:
        st.success("✅ CV bu ilan için oldukça uygun.")
    elif final >= 50:
        st.warning("⚠️ CV kısmen uygun, geliştirilebilir.")
    else:
        st.error("❌ CV bu ilan için zayıf.")

    with st.expander("📄 İş İlanı Açıklaması (ilk 3000 karakter)"):
        st.write(job_text[:3000])

    with st.expander("📄 CV Metni (ilk 2000 karakter)"):
        st.write(cv_text[:2000])
