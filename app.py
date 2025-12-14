import os
import streamlit as st
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from pypdf import PdfReader
from deepmultilingualpunctuation import PunctuationModel

os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

st.set_page_config(
    page_title="BriefAI Enterprise",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at center, #1e293b, #0f172a);
        color: #ffffff;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .hero-title {
        text-align: center;
        font-size: 60px;
        font-weight: 800;
        background: -webkit-linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: -50px;
        margin-bottom: 10px;
    }
    
    .hero-subtitle {
        text-align: center;
        font-size: 18px;
        color: #94a3b8;
        margin-bottom: 40px;
    }

    .stTextInput > div > div > input {
        border-radius: 50px;
        border: 2px solid #334155;
        background-color: #1e293b;
        color: white;
        padding: 25px 20px;
        font-size: 16px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #84fab0;
        box-shadow: 0 0 15px rgba(132, 250, 176, 0.3);
    }

    div.stButton > button {
        width: 100%;
        border-radius: 50px;
        height: 55px;
        background: linear-gradient(90deg, #84fab0 0%, #8fd3f4 100%);
        color: #0f172a;
        font-weight: 700;
        font-size: 18px;
        border: none;
        box-shadow: 0 4px 15px rgba(132, 250, 176, 0.4);
        transition: 0.3s;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(132, 250, 176, 0.6);
        color: black;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px;
        margin-top: 20px;
    }
    
    .metric-container {
        text-align: center;
        padding: 20px;
        background: rgba(0,0,0,0.2);
        border-radius: 15px;
    }

</style>
""", unsafe_allow_html=True)

#  Model Functions
@st.cache_resource
def load_models():
    # Load all models 
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")
    sentiment = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")
    punctuation = PunctuationModel()
    return summarizer, sentiment, punctuation

# ETL Functions
def get_youtube_text(url):
    try:
        if "v=" in url: video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be" in url: video_id = url.split("/")[-1]
        else: return None, "Invalid URL"
        
        ytt = YouTubeTranscriptApi()
        try: transcript = ytt.list(video_id).find_transcript(['en']).fetch()
        except: transcript = next(iter(ytt.list(video_id))).fetch()
        
        return " ".join([t.text for t in transcript]), None
    except Exception as e: return None, str(e)

def get_pdf_text(file):
    try:
        reader = PdfReader(file)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e: return str(e)

def summarize(text, model):
    chunk_size = 2000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    res = []
    
    # Custom Progress 
    status_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        status_text.caption(f"⚡ Neural Network Processing: Segment {i+1}/{len(chunks)}")
        try:
            out = model(chunk, max_length=130, min_length=30, do_sample=False, truncation=True)
            res.append(out[0]['summary_text'])
        except: continue
    
    status_text.empty()
    return " ".join(res) if res else "Error"

# Main UI Layout 

st.markdown('<div class="hero-title">BriefAI <span style="font-size:30px; vertical-align:top;">PRO</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Turn hours of content into seconds of insight. Powered by Advanced NLP.</div>', unsafe_allow_html=True)


col_spacer1, col_main, col_spacer2 = st.columns([1, 2, 1])

with col_main:
    inputType = st.selectbox("Select Content Source", ["YouTube Video Link", "PDF Document Upload"], label_visibility="collapsed")
    
    text_input = ""
    
    if inputType == "YouTube Video Link":
        link = st.text_input("YouTube Link", placeholder="Paste your YouTube link here...", label_visibility="collapsed")
        if link:
            text_input, err = get_youtube_text(link)
            if err: st.error(err)
            
    else:
        uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
        if uploaded:
            text_input = get_pdf_text(uploaded)

    st.write("")
    analyze = st.button("GENERATE INTELLIGENCE")

# Results
if analyze and text_input:
    summ_model, sent_model, punct_model = load_models()
    
    # Punctuation
    with st.spinner("✨ AI is refining the text..."):
        if len(text_input) < 3000:
            text_input = punct_model.restore_punctuation(text_input)
        else:
            # Uzunsa sadece başını düzelt
            fixed = punct_model.restore_punctuation(text_input[:2000])
            text_input = fixed + text_input[2000:]

    # Summarization
    with st.spinner(" Synthesizing abstractive summary..."):
        summary = summarize(text_input, summ_model)

    # Sentiment
    sent = sent_model(summary[:512])[0]
    
    # Result Display
    st.markdown("---")
    
    col_res1, col_res2 = st.columns([2, 1])
    
    with col_res1:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="color:#84fab0; margin-top:0;">📝 Executive Summary</h3>
            <p style="font-size: 16px; line-height: 1.7; color: #e2e8f0;">{summary}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_res2:
        color = "#4ade80" if sent['label'] == 'POSITIVE' else "#f87171"
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="color:#8fd3f4; margin-top:0;">📊 Analysis</h3>
            <div class="metric-container">
                <div style="font-size:14px; color:#94a3b8;">Detected Tone</div>
                <div style="font-size:32px; font-weight:bold; color:{color};">{sent['label']}</div>
            </div>
            <div class="metric-container" style="margin-top:10px;">
                <div style="font-size:14px; color:#94a3b8;">Confidence Score</div>
                <div style="font-size:32px; font-weight:bold; color:white;">{sent['score']:.1%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Original Text
        with st.expander("View Original Source Text"):
            st.text_area("", text_input, height=200)

if not analyze:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    
    with f1:
        st.markdown("""
        <div style="text-align:center;">
            <h1>🎥</h1>
            <h3>Video to Text</h3>
            <p style="color:#94a3b8;">Instantly transcribe and process YouTube content.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with f2:
        st.markdown("""
        <div style="text-align:center;">
            <h1>🧠</h1>
            <h3>Deep Understanding</h3>
            <p style="color:#94a3b8;">BART Large model captures context, not just keywords.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with f3:
        st.markdown("""
        <div style="text-align:center;">
            <h1>📈</h1>
            <h3>Sentiment AI</h3>
            <p style="color:#94a3b8;">Detect emotional tone and hidden patterns.</p>
        </div>

        """, unsafe_allow_html=True)

