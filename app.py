import streamlit as st
import re
from io import BytesIO
from collections import Counter

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
    layout="wide"
)

st.title("🧠 AI Document Analyzer (Streamlit Cloud Ready)")
st.markdown(
    "Upload a **PDF / DOCX / TXT** and get: *summary, teacher-style explanation, quiz, flashcards*, "
    "plus sentiment, keywords, word cloud, and a downloadable PDF report."
)


# =========================
# 2) NLTK SETUP (safe on cloud)
# =========================
@st.cache_resource
def ensure_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

ensure_nltk()

STOPWORDS = set(stopwords.words("english"))


# =========================
# 3) GEMINI (no transformers/tokenizers)
# =========================
@st.cache_resource
def init_gemini():
    # You must add this in Streamlit Cloud:
    # Settings -> Secrets -> GEMINI_API_KEY = "YOUR_KEY"
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return True


def gemini_generate(prompt: str, model: str = "gemini-1.5-flash") -> str:
    if init_gemini() is None:
        raise RuntimeError(
            "Missing Gemini API key in Streamlit Secrets. "
            "Add GEMINI_API_KEY in Settings → Secrets."
        )
    model_obj = genai.GenerativeModel(model)
    resp = model_obj.generate_content(prompt)
    return (resp.text or "").strip()


# =========================
# 4) FILE EXTRACTION
# =========================
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(uploaded_file) -> str:
    text = []
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        p = page.extract_text()
        if p:
            text.append(p)
    return "\n".join(text)


@st.cache_data(show_spinner=False)
def extract_text_from_docx(uploaded_file) -> str:
    doc = docx.Document(uploaded_file)
    return "\n".join([p.text for p in doc.paragraphs if p.text])


@st.cache_data(show_spinner=False)
def extract_text_from_txt(uploaded_file) -> str:
    return uploaded_file.read().decode("utf-8", errors="ignore")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# =========================
# 5) LIGHTWEIGHT ANALYSIS (no heavy deps)
# =========================
def analyze_text(text: str):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # -1..1

    words = [w.lower() for w in re.findall(r"\b[\w']+\b", text) if w.lower() not in STOPWORDS]
    keywords = Counter(words).most_common(12)

    # Simple "entities" via capitalization (fast and dependency-free)
    entities = sorted(set(re.findall(r"\b[A-Z][a-zA-Z0-9_]+\b", text)))
    return sentiment, keywords, entities, words


# =========================
# 6) CHUNKING (to keep Gemini calls stable)
# =========================
def chunks_by_words(text: str, max_words: int = 900) -> list[str]:
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), max_words):
        chunks.append(" ".join(tokens[i:i + max_words]))
    return chunks


def build_mode_prompt(mode: str, chunk: str) -> str:
    if mode == "Summarize":
        return f"Summarize the following document clearly and concisely (use headings when helpful):\n\n{chunk}"
    if mode == "Explain Like a Teacher":
        return (
            "Explain the following content like a teacher for beginners. "
            "Use simple language, include 2–3 examples, and end with a short takeaway:\n\n"
            f"{chunk}"
        )
    if mode == "Generate Quiz Questions":
        return (
            "Create exactly 5 quiz questions from this content. "
            "For each question provide 4 options (A–D) and mark the correct answer:\n\n"
            f"{chunk}"
        )
    if mode == "Create Flashcards":
        return (
            "Create 10 flashcards in this exact format (one per line):\n"
            "Q: ...\nA: ...\n\n"
            f"{chunk}"
        )
    return f"Process this text:\n\n{chunk}"


def run_ai_pipeline(text: str, mode: str) -> str:
    chunks = chunks_by_words(text)
    outputs = []
    for c in chunks:
        out = gemini_generate(build_mode_prompt(mode, c))
        if out.strip():
            outputs.append(out.strip())

    # If multiple chunks, synthesize into one clean final answer
    if len(outputs) > 1:
        stitched = "\n\n".join(outputs)
        final_prompt = (
            "You are a helpful assistant. Condense the following multiple partial outputs into one "
            "coherent final response (preserve key details, remove repetition):\n\n" + stitched
        )
        return gemini_generate(final_prompt)
    return outputs[0] if outputs else ""


# =========================
# 7) WORD CLOUD + PDF REPORT
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
        textColor=colors.HexColor("#004AAD")
    )

    story = [Paragraph("🧠 AI Document Analyzer Report", title_style), Spacer(1, 18)]

    story.append(Paragraph("📑 Summary", styles["Heading2"]))
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
        story.append(ListFlowable([ListItem(Paragraph(e, styles["BodyText"])) for e in entities[:30]], bulletType="bullet"))
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
# 8) APP UI
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

    st.subheader("📄 Extracted Text (preview)")
    st.text_area("Preview (first ~1500 chars)", text[:1500] + ("..." if len(text) > 1500 else ""), height=220)

    st.subheader("🎛️ Choose AI Task")
    mode = st.selectbox("What should the AI do?", ["Summarize", "Explain Like a Teacher", "Generate Quiz Questions", "Create Flashcards"])

    if init_gemini() is None:
        st.warning(
            "To enable AI outputs, add your Gemini key in **Streamlit Cloud → Settings → Secrets** "
            "as `GEMINI_API_KEY`."
        )
        st.stop()

    with st.spinner("🤖 Generating AI output (Gemini)..."):
        ai_output = run_ai_pipeline(text, mode)

    st.subheader(f"🎯 AI Output — {mode}")
    st.write(ai_output)

    st.subheader("📊 Additional Analysis (local)")
    sentiment, keywords, entities, words = analyze_text(text)
    st.write(f"😊 Sentiment polarity: **{sentiment:.3f}**")
    st.write("🔑 Top keywords:", ", ".join([f"{k}({c})" for k, c in keywords[:10]]))
    st.write("🏷️ Example entities:", ", ".join(entities[:20]) if entities else "None detected")

    st.subheader("☁️ Word Cloud")
    wc_buf = generate_wordcloud_image(words)
    st.image(wc_buf, caption="Word cloud (from extracted text)")

    st.subheader("📄 Download Report (PDF)")
    pdf_buf = create_pdf_report(ai_output, sentiment, keywords, entities, wc_buf)
    st.download_button("📥 Download PDF Report", pdf_buf, "AI_Document_Report.pdf", mime="application/pdf")
