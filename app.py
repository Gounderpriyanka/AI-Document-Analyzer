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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle

# Configure page first
st.set_page_config(page_title="AI Document Analyzer", page_icon="🧠", layout="wide")

# -----------------------------
# Model loading with error handling
# -----------------------------

@st.cache_resource
def load_spacy_model():
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("spaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")
        return None
    except Exception as e:
        st.error(f"Error loading spaCy model: {e}")
        return None

@st.cache_resource
def load_summarizer():
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        st.error(f"Error loading summarizer: {e}")
        return None

# Initialize models
nlp = load_spacy_model()
summarizer = load_summarizer()

# -----------------------------
# Helper functions
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(uploaded_file):
    try:
        doc = docx.Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

def analyze_text(text):
    blob = TextBlob(text)
    
    # Sentiment
    sentiment = blob.sentiment.polarity

    # Keywords
    words = [word.lower() for word in re.findall(r'\b[a-zA-Z]{3,}\b', text)]  # Only words with 3+ letters
    common_words = Counter(words).most_common(10)

    # Named Entities (only if spaCy is available)
    entities = []
    if nlp:
        try:
            # Process only first 10,000 characters for performance
            doc = nlp(text[:10000])
            entities = [(ent.text, ent.label_) for ent in doc.ents[:20]]  # Limit to 20 entities
        except Exception as e:
            st.warning(f"Could not extract named entities: {e}")

    return sentiment, common_words, entities, words

def ai_summary(text):
    if not summarizer:
        return "Summary not available - model failed to load."
    
    try:
        # Limit text length for performance
        max_text_length = 3000
        if len(text) > max_text_length:
            text = text[:max_text_length]
            st.info(f"Text truncated to {max_text_length} characters for faster processing.")
        
        max_chunk = 800  # Reduced chunk size for better performance
        text_chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
        summary = ""
        
        progress_bar = st.progress(0)
        for i, chunk in enumerate(text_chunks):
            if len(chunk.strip()) > 50:  # Only process substantial chunks
                result = summarizer(chunk, max_length=120, min_length=30, do_sample=False)
                summary += result[0]['summary_text'] + " "
            progress_bar.progress((i + 1) / len(text_chunks))
            
        return summary.strip()
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "Summary generation failed. Please try with a shorter document."

def generate_wordcloud(words):
    if not words:
        return None
        
    try:
        text = " ".join(words[:200])  # Limit words for performance
        wordcloud = WordCloud(width=800, height=400, background_color="white", max_words=50).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', dpi=150)
        plt.close(fig)  # Close the figure to free memory
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Error generating word cloud: {e}")
        return None

def create_pdf(summary, sentiment, keywords, entities, wordcloud_buf):
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )

        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            name="TitleStyle",
            parent=styles["Title"],
            fontSize=22,
            textColor=colors.HexColor("#004AAD"),
            alignment=TA_CENTER,
            spaceAfter=20
        )

        section_header = ParagraphStyle(
            name="SectionHeader",
            parent=styles["Heading2"],
            fontSize=14,
            textColor=colors.HexColor("#004AAD"),
            spaceBefore=10,
            spaceAfter=8
        )

        normal_text = ParagraphStyle(
            name="NormalText",
            parent=styles["Normal"],
            fontSize=11,
            leading=15
        )

        story = []

        # Header Section
        story.append(Paragraph("🧠 AI Document Analyzer Report", title_style))
        story.append(Paragraph("Generated by AI Document Analyzer", normal_text))
        story.append(Spacer(1, 15))
        story.append(Spacer(1, 10))

        # Summary Section
        story.append(Paragraph("📑 Summary", section_header))
        story.append(Paragraph(summary, normal_text))
        story.append(Spacer(1, 12))

        # Sentiment Section
        story.append(Paragraph("😊 Sentiment Analysis", section_header))
        story.append(Paragraph(f"<b>Polarity Score:</b> {sentiment:.2f}", normal_text))
        if sentiment > 0:
            story.append(Paragraph("Overall Sentiment: <font color='green'><b>Positive</b></font>", normal_text))
        elif sentiment < 0:
            story.append(Paragraph("Overall Sentiment: <font color='red'><b>Negative</b></font>", normal_text))
        else:
            story.append(Paragraph("Overall Sentiment: <font color='gray'><b>Neutral</b></font>", normal_text))
        story.append(Spacer(1, 12))

        # Keywords Section
        story.append(Paragraph("🔑 Top Keywords", section_header))
        if keywords:
            keyword_text = "<br/>".join([f"{w}: {f}" for w, f in keywords])
            story.append(Paragraph(keyword_text, normal_text))
        else:
            story.append(Paragraph("No significant keywords found.", normal_text))
        story.append(Spacer(1, 12))

        # Entities Section
        story.append(Paragraph("🏷️ Named Entities", section_header))
        if entities:
            entity_text = "<br/>".join([f"{ent} ({label})" for ent, label in entities])
            story.append(Paragraph(entity_text, normal_text))
        else:
            story.append(Paragraph("No named entities found.", normal_text))
        story.append(Spacer(1, 12))

        # Word Cloud Section
        if wordcloud_buf:
            story.append(Paragraph("☁️ Word Cloud", section_header))
            img = Image(wordcloud_buf, width=5.5*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 12))

        # Footer
        story.append(Spacer(1, 20))
        story.append(Paragraph(
            "<hr/><br/><b>Report generated using AI Document Analyzer</b>",
            ParagraphStyle(name="Footer", alignment=TA_CENTER, fontSize=9, textColor=colors.gray)
        ))

        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error creating PDF: {e}")
        return None

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🧠 AI Document Analyzer")
st.markdown("Upload a document (PDF, DOCX, or TXT) and get **AI-powered insights** — including summary, sentiment, keywords, named entities, a word cloud, and downloadable PDF report!")

# Check if models loaded successfully
if not nlp:
    st.warning("⚠️ spaCy model not available. Named entity recognition will be disabled.")
if not summarizer:
    st.warning("⚠️ Summarization model not available. AI summary will be disabled.")

uploaded_file = st.file_uploader("📁 Upload your document", type=["pdf", "docx", "txt"])

if uploaded_file:
    # Check file size
    if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
        st.error("File size too large. Please upload a file smaller than 10MB.")
        st.stop()
    
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    with st.spinner("Reading document..."):
        if file_type == "pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif file_type == "docx":
            text = extract_text_from_docx(uploaded_file)
        elif file_type == "txt":
            try:
                text = uploaded_file.read().decode("utf-8")
            except:
                text = uploaded_file.read().decode("latin-1")
        else:
            st.error("Unsupported file type!")
            st.stop()
    
    text = clean_text(text)
    
    if len(text) < 50:
        st.warning("The document seems too short for analysis.")
    else:
        st.success(f"✅ Document loaded! ({len(text)} characters)")
        
        with st.expander("📄 Extracted Text Preview", expanded=False):
            st.text_area("Text Preview", text[:1000] + "..." if len(text) > 1000 else text, height=200, label_visibility="collapsed")

        # Analysis in tabs for better organization
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📑 Summary", "😊 Sentiment", "🔑 Keywords", "🏷️ Entities", "☁️ Word Cloud"])

        with tab1:
            st.subheader("AI Summary")
            summary = ai_summary(text)
            st.write(summary)

        with tab2:
            st.subheader("Sentiment Analysis")
            sentiment, keywords, entities, all_words = analyze_text(text)
            st.write(f"Polarity Score: `{sentiment:.2f}`")
            if sentiment > 0:
                st.success("Overall Sentiment: Positive")
            elif sentiment < 0:
                st.error("Overall Sentiment: Negative")
            else:
                st.info("Overall Sentiment: Neutral")

        with tab3:
            st.subheader("Top Keywords")
            for word, freq in keywords:
                st.write(f"- `{word}`: {freq}")

        with tab4:
            st.subheader("Named Entities")
            if entities:
                for ent, label in entities:
                    st.write(f"- `{ent}` ({label})")
            else:
                st.info("No named entities found or NER disabled.")

        with tab5:
            st.subheader("Word Cloud")
            wordcloud_buf = generate_wordcloud(all_words)
            if wordcloud_buf:
                st.image(wordcloud_buf, use_column_width=True)
            else:
                st.info("Could not generate word cloud.")

        # PDF Generation
        st.markdown("---")
        st.subheader("📊 Download Report")
        
        if st.button("Generate PDF Report", type="primary"):
            with st.spinner("Creating PDF report..."):
                pdf_buffer = create_pdf(summary, sentiment, keywords, entities, wordcloud_buf)

            if pdf_buffer:
                st.download_button(
                    label="📥 Download Full Report as PDF",
                    data=pdf_buffer,
                    file_name="AI_Document_Analysis_Report.pdf",
                    mime="application/pdf",
                    type="primary"
                )
            else:
                st.error("Failed to generate PDF report.")
