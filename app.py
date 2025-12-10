import streamlit as st
import google.auth
from google.oauth2 import service_account
from google.api_core.client_options import ClientOptions
from google.generativeai import GenerativeModel
import google.generativeai as genai

import PyPDF2
import docx
import nltk
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------------------------------
# 1️⃣ LOAD SERVICE ACCOUNT CREDENTIALS
# -----------------------------------------------------
@st.cache_resource
def load_credentials():
    return service_account.Credentials.from_service_account_file(
        "service-account.json",
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

credentials = load_credentials()

# Required for Gemini client to work with service accounts
client_options = ClientOptions(api_endpoint="https://generativelanguage.googleapis.com")

# -----------------------------------------------------
# 2️⃣ SELECT GEMINI MODEL
# -----------------------------------------------------
@st.cache_resource
def get_model():
    return genai.GenerativeModel(
        model_name="models/gemini-1.5-flash",
        client_options=client_options,
        credentials=credentials
    )

model = get_model()

# -----------------------------------------------------
# 3️⃣ TEXT EXTRACTION
# -----------------------------------------------------
def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text

    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join(p.text for p in doc.paragraphs)
        return text

    else:
        return file.read().decode("utf-8")


# -----------------------------------------------------
# 4️⃣ AI ANALYSIS PIPELINE
# -----------------------------------------------------
def analyze_text_with_gemini(text, mode):
    prompt = f"""
You are an AI Document Analyzer. Perform the selected operation:

Mode: {mode}

Document:
{text}

Provide a clean, structured output.
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini error: {e}"


# -----------------------------------------------------
# 5️⃣ STREAMLIT UI
# -----------------------------------------------------
st.set_page_config(page_title="AI Document Analyzer", layout="wide")
st.title("📄 AI Document Analyzer (Gemini 1.5 Flash)")

uploaded_file = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"])

analysis_mode = st.selectbox(
    "Choose analysis type:",
    [
        "Summarization",
        "Keywords",
        "Sentiment Analysis",
        "Q&A",
        "Rewrite",
        "Grammar Check",
        "Full Analysis"
    ]
)

if uploaded_file:
    st.success("File uploaded successfully!")
    text = extract_text(uploaded_file)
    st.subheader("Extracted Text")
    st.text_area("Document Content", text, height=250)

    if st.button("Run AI Analysis"):
        with st.spinner("Analyzing with Gemini 1.5 Flash..."):
            output = analyze_text_with_gemini(text, analysis_mode)

        st.subheader("🔍 AI Result")
        st.write(output)

