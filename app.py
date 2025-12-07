import streamlit as st
import PyPDF2
import docx
from textblob import TextBlob
from collections import Counter
import re
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
import spacy


# -----------------------------
# Load SpaCy Model (simple English model for sentence splitting)
# -----------------------------
@st.cache_resource
def load_spacy():
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


nlp = load_spacy()


# -----------------------------
# Load Summarizer
# -----------------------------
@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )


summarizer = load_summarizer()


# --------------------------------
# Extract Text
# --------------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        ptext = page.extract_text()
        if ptext:
            text += ptext + "\n"
    return text


def extract_text_from_docx(uploaded_file):
    doc_file = docx.Document(uploaded_file)
    return "\n".join([p.text for p in doc_file.paragraphs])


def clean_text(text):
    return re.sub(r"\s+", " ", text.strip())


# --------------------------------
# Topic Segmentation
# --------------------------------
def segment_topics(text, chunk_size=300):
    sentences = list(nlp(text).sents)
    topics = []
    chunk = ""
    for s in sentences:
        if len(chunk) + len(s.text) < chunk_size:
            chunk += " " + s.text
        else:
            topics.append(chunk.strip())
            chunk = s.text
    if chunk:
        topics.append(chunk.strip())
    return topics


# --------------------------------
# Text Analysis
# --------------------------------
def analyze_text(text):
    doc = nlp(text)
    blob = TextBlob(text)

    sentiment = blob.sentiment.polarity

    stopwords = nlp.Defaults.stop_words
    words = [w.lower() for w in re.findall(r"\b\w+\b", text) if w.lower() not in stopwords]

    keywords = Counter(words).most_common(10)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return sentiment, keywords, entities, words


# --------------------------------
# AI Prompt Generator
# --------------------------------
def create_prompt(text, mode):
    if mode == "Summarize":
        return f"Summarize this clearly:\n{text}"

    if mode == "Explain Like a Teacher":
        return f"Explain this like a teacher with examples:\n{text}"

    if mode == "Generate Quiz Questions":
        return f"Generate 5 quiz questions from this content:\n{text}"

    if mode == "Create Flashcards":
        return f"Make flashcards in Q:A format:\n{text}"


# --------------------------------
# Chunk-based summarizer
# --------------------------------
def split_into_chunks(text, max_words=350):
    words = text.split()
    chunks = []
    current = []

    for w in words:
        current.append(w)
        if len(current) >= max_words:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


def generate_ai_output(text, mode):
    chunks = split_into_chunks(text)
    final_output = ""

    for chunk in chunks:
        prompt = create_prompt(chunk, mode)

        response = summarizer(
            prompt,
            max_length=200,
            min_length=50,
            do_sample=False
        )

        final_output += response[0]["summary_text"] + "\n\n"

    return final_output.strip()


# --------------------------------
# Word Cloud
# --------------------------------
def generate_wordcloud(words):
    wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(words))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf


# --------------------------------
# PDF Report
# --------------------------------
def create_pdf(summary, sentiment, keywords, entities, wordcloud_buf):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Title",
        fontSize=22,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#004AAD")
    )

    story = [Paragraph("🧠 AI Document Analyzer Report", title), Spacer(1, 20)]

    story.append(Paragraph("📑 Summary", styles["Heading2"]))
    story.append(Paragraph(summary, styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("😊 Sentiment", styles["Heading2"]))
    story.append(Paragraph(f"Polarity Score: {sentiment}", styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("🔑 Keywords", styles["Heading2"]))
    story.append(ListFlowable([ListItem(Paragraph(f"{w}: {c}", styles["BodyText"])) for w, c in keywords]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("🏷️ Named Entities", styles["Heading2"]))
    if entities:
        story.append(ListFlowable([
            ListItem(Paragraph(f"{e} ({l})", styles["BodyText"])) for e, l in entities
        ]))
    else:
        story.append(Paragraph("No entities found.", styles["BodyText"]))

    story.append(Spacer(1, 12))

    if wordcloud_buf:
        story.append(Paragraph("☁️ Word Cloud", styles["Heading2"]))
        story.append(Image(wordcloud_buf, width=5.5 * inch))

    doc.build(story)
    buffer.seek(0)
    return buffer


# --------------------------------
# STREAMLIT UI
# --------------------------------
st.set_page_config(page_title="AI Document Analyzer", page_icon="🧠", layout="wide")

st.title("🧠 AI Document Analyzer")
st.markdown("Upload a document and get **AI-powered summaries, quizzes, explanations, keywords, and more.**")

uploaded_file = st.file_uploader("📁 Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"])

if uploaded_file:
    ext = uploaded_file.name.split(".")[-1].lower()

    with st.spinner("📥 Extracting text..."):
        if ext == "pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif ext == "docx":
            text = extract_text_from_docx(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8")

    text = clean_text(text)

    if len(text) < 50:
        st.error("Not enough content to analyze.")
        st.stop()

    st.text_area("📄 Extracted Text", text[:1000] + "...", height=250)

    topics = segment_topics(text)

    st.markdown("### 🧩 Detected Topics")
    for i, t in enumerate(topics[:5]):
        st.write(f"**Topic {i+1}:** {t[:200]}...")

    st.markdown("### 🎛️ Choose AI Mode")
    mode = st.selectbox(
        "What should the AI do?",
        ["Summarize", "Explain Like a Teacher", "Generate Quiz Questions", "Create Flashcards"]
    )

    with st.spinner("🤖 AI Thinking..."):
        result = generate_ai_output(text, mode)

    st.markdown(f"### 🎯 AI Output — *{mode}*")
    st.write(result)

    st.markdown("### 📊 Additional Analysis")

    sentiment, keywords, entities, words = analyze_text(text)

    st.write("#### 😊 Sentiment Polarity:", sentiment)

    st.write("#### 🔑 Keywords:")
    for w, c in keywords:
        st.write(f"- {w}: {c}")

    st.write("#### 🏷️ Named Entities:")
    if entities:
        for e, l in entities:
            st.write(f"- {e} ({l})")
    else:
        st.write("No entities found.")

    st.write("#### ☁️ Word Cloud:")
    wordcloud_buf = generate_wordcloud(words)

    pdf_buf = create_pdf(result, sentiment, keywords, entities, wordcloud_buf)

    st.download_button(
        "📥 Download PDF Report",
        pdf_buf,
        "AI_Document_Report.pdf",
        "application/pdf"
    )
