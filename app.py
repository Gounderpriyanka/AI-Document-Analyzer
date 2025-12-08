import os
import re
from io import BytesIO
from collections import Counter

import streamlit as st
import PyPDF2
import docx
from textblob import TextBlob

import nltk
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
from wordcloud import WordCloud

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER

import google.generativeai as genai


# =========================
# 1) STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="AI Document Analyzer",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 AI Document Analyzer (Gemini Only)")
st.markdown(
    "Upload a **PDF / DOCX / TXT** and get:\n"
    "- ✅ Summaries\n"
    "- ✅ Teacher-style explanations\n"
    "- ✅ Quiz questions (MCQ)\n"
    "- ✅ Flashcards (Q/A)\n"
    "- 📊 Sentiment + keywords + word cloud\n"
    "- 📄 Downloadable PDF report"
)


# =========================
# 2) NLTK SETUP (CLOUD-SAFE)
# =========================
@st.cache_resource
def ensure_nltk():
    """Download required NLTK data once (cached)."""
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
    except Exception:
        # If download fails (rare in some locked environments), we still continue using a fallback stoplist.
        return False
    return True


ensure_nltk()

def get_stopwords_set():
    try:
        return set(stopwords.words("english"))
    except Exception:
        # Fallback (keeps the app running even if NLTK data isn't available)
        return {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
            "with", "by", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "can", "shall", "as", "it", "this", "that"
        }

STOPWORDS = get_stopwords_set()


# =========================
# 3) GEMINI INIT (STREAMLIT CLOUD SAFE)
# =========================
@st.cache_resource
def init_gemini():
    """
    Initialize Gemini with an API key from:
    - st.secrets (recommended for Streamlit Cloud)
    - environment variable: GEMINI_API_KEY
    Returns a configured GenerativeModel or None.
    """
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    # Configure once
    genai.configure(api_key=api_key)

    # Use a valid, widely available model name (2025-safe)
    return genai.GenerativeModel("gemini-1.5-flash")


def gemini_generate(prompt: str) -> str:
    model = init_gemini()
    if model is None:
        return (
            "⚠️ Gemini API key not found.\n\n"
            "Add in Streamlit Cloud → Settings → Secrets:\n"
            "GEMINI_API_KEY = \"YOUR_KEY_HERE\""
        )

    try:
        resp = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 2048}
        )
        return (resp.text or "").strip()
    except Exception as e:
        # Keep the app alive and show a readable error (instead of crashing)
        return f"⚠️ Gemini request failed:\n{str(e)}"


# =========================
# 4) FILE EXTRACTION
# =========================
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(uploaded_file) -> str:
    text_parts = []
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        p = page.extract_text()
        if p:
            text_parts.append(p)
    return "\n".join(text_parts)


@st.cache_data(show_spinner=False)
def extract_text_from_docx(uploaded_file) -> str:
    doc = docx.Document(uploaded_file)
    return "\n".join([p.text for p in doc.paragraphs if p.text])


@st.cache_data(show_spinner=False)
def extract_text_from_txt(uploaded_file) -> str:
    return uploaded_file.read().decode("utf-8", errors="ignore")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# =========================
# 5) TOPIC SEGMENTATION (FOR PREVIEW)
# =========================
def segment_topics(text: str, chunk_chars: int = 520) -> list[str]:
    """Split content into readable topic chunks (for user preview)."""
    sentences = nltk.sent_tokenize(text)
    topics = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) + 1 <= chunk_chars:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                topics.append(buf)
            buf = s
    if buf:
        topics.append(buf)
    return topics


# =========================
# 6) LOCAL NLP ANALYSIS (NO HEAVY MODELS)
# =========================
def analyze_text(text: str):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # -1..1

    words = [w.lower() for w in re.findall(r"\b[\w']+\b", text) if w.lower() not in STOPWORDS]
    keywords = Counter(words).most_common(12)

    # Simple entity proxy: capitalized tokens (safe, dependency-free)
    entities = sorted(set(re.findall(r"\b[A-Z][a-zA-Z0-9_]+\b", text)))

    return sentiment, keywords, entities, words


# =========================
# 7) GEMINI WORKFLOW (CHUNKING + STITCHING)
# =========================
def split_into_chunks(text: str, max_words: int = 900) -> list[str]:
    tokens = text.split()
    return [" ".join(tokens[i:i + max_words]) for i in range(0, len(tokens), max_words)]


def build_prompt(mode: str, chunk: str) -> str:
    if mode == "Summarize":
        return f"Summarize the following document clearly and concisely. Use headings if helpful:\n\n{chunk}"
    if mode == "Explain Like a Teacher":
        return (
            "Explain the following content like a teacher for beginners. "
            "Use simple language, include 2–3 examples, and end with a short takeaway:\n\n"
            f"{chunk}"
        )
    if mode == "Generate Quiz Questions":
        return (
            "Create exactly 5 quiz questions from this content. For each question:\n"
            "- Provide 4 options (A–D)\n"
            "- Mark the correct answer clearly (e.g., Correct: B)\n\n"
            f"{chunk}"
        )
    if mode == "Create Flashcards":
        return (
            "Create 10 flashcards in this exact format (one per line pair):\n"
            "Q: ...\nA: ...\n\n"
            f"{chunk}"
        )
    return f"Process this text:\n\n{chunk}"


def run_ai_pipeline(text: str, mode: str) -> str:
    chunks = split_into_chunks(text)
    partials = []

    for c in chunks:
        p = build_prompt(mode, c)
        out = gemini_generate(p)
        if out and not out.startswith("⚠️ Gemini API key not found"):
            partials.append(out)

    if not partials:
        # If key missing or every chunk failed, return the last returned message (already user-friendly)
        return gemini_generate("Test")  # will return either missing-key message or an error message

    if len(partials) == 1:
        return partials[0]

    stitched = "\n\n".join(partials)
    final_prompt = (
        "You are a helpful assistant. Condense the following multiple partial outputs into one "
        "coherent final response (remove repetition, keep key points):\n\n"
        f"{stitched}"
    )
    return gemini_generate(final_prompt)


# =========================
# 8) WORD CLOUD + PDF REPORT
# =========================
def generate_wordcloud_image(words: list[str]) -> BytesIO:
    wc = WordCloud(width=900, height=450, background_color="white").generate(" ".join(words))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def create_pdf_report(summary: str, sentiment: float, keywords, entities, wc_buf: BytesIO) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        fontSize=22,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#004AAD"),
    )

    story = [Paragraph("🧠 AI Document Analyzer Report", title_style), Spacer(1, 18)]

    story.append(Paragraph("📑 AI Output", styles["Heading2"]))
    story.append(Paragraph(summary.replace("\n", "<br/>"), styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("😊 Sentiment", styles["Heading2"]))
    story.append(Paragraph(f"Polarity score: {sentiment:.3f} (−1 negative → +1 positive)", styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("🔑 Keywords", styles["Heading2"]))
    story.append(ListFlowable([ListItem(Paragraph(f"{k} — {c}", styles["BodyText"])) for k, c in keywords], bulletType="bullet"))
    story.append(Spacer(1, 10))

    story.append(Paragraph("🏷️ Entities (simple)", styles["Heading2"]))
    if entities:
        story.append(ListFlowable([ListItem(Paragraph(e, styles["BodyText"])) for e in entities[:40]], bulletType="bullet"))
    else:
        story.append(Paragraph("No entities found.", styles["BodyText"]))

    story.append(Spacer(1, 12))

    if wc_buf:
        story.append(Paragraph("☁️ Word Cloud", styles["Heading2"]))
        story.append(Image(wc_buf, width=5.8 * inch, height=2.9 * inch))

    doc.build(story)
    buffer.seek(0)
    return buffer


# =========================
# 9) APP UI (UPLOAD → ANALYZE → DOWNLOAD)
# =========================
uploaded_file = st.file_uploader("📁 Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"])

if uploaded_file:
    ext = uploaded_file.name.lower().split(".")[-1]

    with st.spinner("📥 Extracting text..."):
        if ext == "pdf":
            raw_text = extract_text_from_pdf(uploaded_file)
        elif ext == "docx":
            raw_text = extract_text_from_docx(uploaded_file)
        else:
            raw_text = extract_text_from_txt(uploaded_file)

    text = clean_text(raw_text)

    if len(text) < 80:
        st.error("Not enough readable content found in this file.")
        st.stop()

    st.subheader("📄 Extracted Text (Preview)")
    st.text_area("Preview (first ~1500 chars)", text[:1500] + ("..." if len(text) > 1500 else ""), height=240)

    st.subheader("🧩 Detected Topics (Preview)")
    topics = segment_topics(text, chunk_chars=520)
    for i, t in enumerate(topics[:5], start=1):
        st.markdown(f"**Topic {i}:** {t[:260]}{'...' if len(t) > 260 else ''}")

    st.subheader("🎛️ Choose AI Task")
    mode = st.selectbox("What should the AI do?", ["Summarize", "Explain Like a Teacher", "Generate Quiz Questions", "Create Flashcards"])

    # If key missing, show clear instruction (and still allow local analysis)
    model_ready = init_gemini() is not None
    if not model_ready:
        st.warning(
            "Gemini is not configured. To enable AI output, add this secret in Streamlit Cloud:\n\n"
            "GEMINI_API_KEY = \"YOUR_KEY_HERE\""
        )

    with st.spinner("🤖 Generating AI output..."):
        ai_output = run_ai_pipeline(text, mode)

    st.subheader(f"🎯 AI Output — {mode}")
    st.write(ai_output)

    st.subheader("📊 Additional Analysis (Local)")
    sentiment, keywords, entities, words = analyze_text(text)

    st.write(f"😊 Sentiment polarity: **{sentiment:.3f}**")
    st.write("🔑 Keywords:", ", ".join([f"{k}({c})" for k, c in keywords[:10]]))
    st.write("🏷️ Entities (simple):", ", ".join(entities[:20]) if entities else "None detected")

    st.subheader("☁️ Word Cloud")
    wc_buf = generate_wordcloud_image(words)
    st.image(wc_buf, caption="Word cloud (from extracted text)")

    st.subheader("📄 Download Report (PDF)")
    pdf_buf = create_pdf_report(ai_output, sentiment, keywords, entities, wc_buf)
    st.download_button("📥 Download PDF Report", pdf_buf, "AI_Document_Report.pdf", mime="application/pdf")
