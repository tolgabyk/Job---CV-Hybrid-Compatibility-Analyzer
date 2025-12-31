# Job---CV-Hybrid-Compatibility-Analyzer

Bu proje, bir iş ilanı açıklaması ile CV (PDF) arasındaki uyumluluğu anlamsal (semantic) olarak analiz eden bir Streamlit tabanlı web uygulamasıdır.

Kullanıcı:
- İş ilanı açıklamasını manuel olarak yapıştırır
- CV’sini PDF olarak yükler
- Sistem, metinleri analiz ederek % uyumluluk skoru üretir

Özellikler
- PDF CV okuma
- Manuel job description girişi
- Semantic similarity analizi
- Hybrid model yaklaşımı
- TF-IDF (kelime bazlı örtüşme)
- Sentence Transformers (anlamsal benzerlik)
- Yüzdelik uyumluluk skoru
- Streamlit arayüzü

Neden Metin Sınırı Var?

Metin	Karakter Limiti
CV	            = 2000
Job Description =	3000

Sebep:

- Uzun metinlerde semantic dilution oluşması
- Gürültülü verinin embedding kalitesini düşürmesi
- Performans ve doğruluk dengesi


📦 Kurulum
pip install streamlit PyPDF2 scikit-learn sentence-transformers torch

▶️ Çalıştırma
streamlit run app.py

# Geliştirilebilir Alanlar

CV section bazlı embedding

Skill extraction (NER)

Role-based weighting

Çoklu ilan karşılaştırması

ATS uyum skoru

LLM destekli açıklama üretimi

