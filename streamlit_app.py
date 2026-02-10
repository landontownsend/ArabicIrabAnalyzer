import streamlit as st
import os
import json
import time
import subprocess
import sys
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from google import genai

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Arabic Irab Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stTextArea textarea {
        direction: rtl;
        font-size: 20px;
        font-family: 'Amiri', 'Traditional Arabic', serif;
    }
</style>
""", unsafe_allow_html=True)

# ── Download CAMeL Tools data (required for Streamlit Cloud) ──────────────────
@st.cache_resource
def download_camel_data():
    try:
        subprocess.run(
            [sys.executable, "-m", "camel_tools.data", "download", "-y", "morphology-db-msa-r13"],
            check=False,
            capture_output=True
        )
    except Exception:
        pass

download_camel_data()

# ── Load resources ────────────────────────────────────────────────────────────
@st.cache_resource
def load_analyzer():
    db = MorphologyDB.builtin_db()
    return Analyzer(db)

@st.cache_resource
def load_gemini_client():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found. Add it to .env locally or Streamlit Secrets for deployment.")
        st.stop()
    client = genai.Client(api_key=api_key)
    return client

# ── Core functions ────────────────────────────────────────────────────────────
def tokenize_arabic(text):
    tokens = simple_word_tokenize(text)
    return [t for t in tokens if t.strip() and not all(c in ".,!?;:،؛" for c in t)]

def analyze_word_morphology(word, analyzer, max_analyses=1):
    analyses = analyzer.analyze(word)
    results = []
    for analysis in analyses[:max_analyses]:
        results.append({
            "word"    : word,
            "diac"    : analysis.get("diac", word),
            "lex"     : analysis.get("lex",  ""),
            "pos"     : analysis.get("pos",  ""),
            "gloss"   : analysis.get("gloss",""),
            "features": {
                "gender": analysis.get("gen", ""),
                "number": analysis.get("num", ""),
                "person": analysis.get("per", ""),
                "case"  : analysis.get("cas", ""),
                "state" : analysis.get("stt", ""),
            }
        })
    return results

def analyze_sentence(sentence, analyzer):
    tokens = tokenize_arabic(sentence)
    return [
        {"original": token, "morphology": analyze_word_morphology(token, analyzer)}
        for token in tokens
    ]

SYSTEM_PROMPT = """أنت خبير في النحو العربي والإعراب.
You are an expert in Arabic grammar and irab. For each word provide:
1. الإعراب (grammatical role)
2. علامة الإعراب (grammatical marker)
3. التفاصيل (gender, number, definiteness)
4. A brief English explanation
Return a JSON array only, no extra text."""

def create_prompt(sentence, morphology_data):
    lines = []
    for w in morphology_data:
        pos   = w["morphology"][0]["pos"] if w["morphology"] else "?"
        lemma = w["morphology"][0]["lex"] if w["morphology"] else "?"
        orig  = w["original"]
        lines.append("- " + orig + " : POS=" + pos + ", lemma=" + lemma)
    morph_hints = "\n".join(lines)
    return (
        SYSTEM_PROMPT + "\n\n"
        "Sentence: " + sentence + "\n\n"
        "Morphology hints:\n" + morph_hints + "\n\n"
        "Return ONLY a JSON array:\n"
        "[{\"word\":\"...\",\"irab\":\"...\",\"sign\":\"...\",\"details\":\"...\",\"explanation\":\"...\"}]"
    )

def get_irab(sentence, morphology_data, client):
    prompt = create_prompt(sentence, morphology_data)
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            text = response.text.strip()
            if text.startswith("```json"):
                text = text.split("```json")[1].split("```")[0].strip()
            elif text.startswith("```"):
                text = text.split("```")[1].split("```")[0].strip()
            return {"success": True, "data": json.loads(text)}
        except Exception as e:
            err = str(e)
            if "429" in err:
                if "PerDay" in err:
                    return {"success": False, "error": "Daily quota reached. Try again tomorrow or check billing."}
                time.sleep(60 * (attempt + 1))
            else:
                return {"success": False, "error": err}
    return {"success": False, "error": "Max retries exceeded."}

@st.cache_data(show_spinner=False)
def run_full_analysis(sentence):
    analyzer = load_analyzer()
    client   = load_gemini_client()
    morphology = analyze_sentence(sentence, analyzer)
    irab_resp  = get_irab(sentence, morphology, client)
    return {
        "original"  : sentence,
        "morphology": morphology,
        "irab"      : irab_resp.get("data", []),
        "success"   : irab_resp["success"],
        "error"     : irab_resp.get("error")
    }

def get_color(irab_type):
    colors = {
        "فاعل"      : "#2e7d32",
        "مفعول به"  : "#1565c0",
        "مبتدأ"     : "#e65100",
        "خبر"       : "#ad1457",
        "مضاف إليه" : "#6a1b9a",
        "نعت"       : "#00838f",
        "حال"       : "#f9a825",
        "فعل"       : "#4e342e",
        "حرف"       : "#546e7a",
        "ظرف"       : "#558b2f",
    }
    for key, color in colors.items():
        if key in irab_type:
            return color
    return "#37474f"

def word_card(word_data):
    color       = get_color(word_data.get("irab", ""))
    word        = word_data.get("word", "")
    irab        = word_data.get("irab", "")
    sign        = word_data.get("sign", "")
    details     = word_data.get("details", "")
    explanation = word_data.get("explanation", "")
    return (
        "<div style=\"background:#fff;border-right:6px solid " + color + ";border-radius:10px;"
        "padding:16px 20px;margin:10px 0;box-shadow:0 2px 6px rgba(0,0,0,0.08);direction:rtl;\">"
        "<div style=\"font-size:26px;font-weight:bold;color:" + color + ";margin-bottom:8px;\">" + word + "</div>"
        "<p style=\"margin:4px 0\"><strong>الإعراب:</strong> " + irab + "</p>"
        "<p style=\"margin:4px 0\"><strong>العلامة:</strong> " + sign + "</p>"
        "<p style=\"margin:4px 0\"><strong>التفاصيل:</strong> " + details + "</p>"
        "<p style=\"margin:4px 0;color:#666;font-style:italic\">" + explanation + "</p>"
        "</div>"
    )

def render_sidebar():
    with st.sidebar:
        st.markdown("## 📚 Arabic Irab Analyzer")
        st.markdown("### محلل الإعراب العربي")
        st.markdown("---")
        st.markdown("""
        **What this tool does:**
        Paste any Arabic sentence — voweled or unvoweled —
        and get a full grammatical breakdown of every word.

        **Powered by:**
        - CAMeL Tools (morphology)
        - Gemini 2.0 Flash (irab analysis)
        """)
        st.markdown("---")
        st.markdown("### أمثلة — Try an example")
        examples = [
            "ذهب الولد إلى المدرسة",
            "قرأ الطالب الكتاب",
            "كتب المعلم الدرس على السبورة",
            "جاء الرجل من السوق",
            "إن الله غفور رحيم",
            "تفتح الأزهار في الربيع",
        ]
        for ex in examples:
            key = "ex_" + ex
            if st.button(ex, key=key, use_container_width=True):
                st.session_state.input_text = ex
                st.rerun()

def main():
    render_sidebar()

    st.title("📚 Arabic Irab Analyzer")
    st.markdown(
        "<div style=\"direction:rtl;font-size:22px;color:#555;\">محلل الإعراب العربي</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    input_text = st.text_area(
        "أدخل الجملة العربية — Enter Arabic sentence:",
        value=st.session_state.get("input_text", ""),
        height=120,
        placeholder="اكتب جملة عربية هنا...",
    )

    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        analyze = st.button("🔍 تحليل | Analyze", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ مسح | Clear", use_container_width=True):
            st.session_state.input_text = ""
            st.rerun()

    if analyze and input_text.strip():
        with st.spinner("جاري التحليل... | Analyzing..."):
            result = run_full_analysis(input_text.strip())

        if result["success"]:
            st.success("✅ Analysis complete | اكتمل التحليل")

            tab1, tab2, tab3 = st.tabs([
                "📊 الإعراب | Irab",
                "🔤 الصرف | Morphology",
                "📝 Raw Data"
            ])

            with tab1:
                st.markdown("#### التحليل النحوي الكامل")
                rows = []
                for w in result["irab"]:
                    rows.append({
                        "الكلمة"     : w.get("word", ""),
                        "الإعراب"    : w.get("irab", ""),
                        "العلامة"    : w.get("sign", ""),
                        "التفاصيل"   : w.get("details", ""),
                        "Explanation": w.get("explanation", "")
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.markdown("---")
                st.markdown("#### تفصيل كل كلمة")
                for w in result["irab"]:
                    st.markdown(word_card(w), unsafe_allow_html=True)

            with tab2:
                st.markdown("#### التحليل الصرفي")
                for word_data in result["morphology"]:
                    if word_data["morphology"]:
                        m        = word_data["morphology"][0]
                        original = word_data["original"]
                        diac     = m.get("diac", "N/A")
                        label    = original + " ← " + diac
                        with st.expander(label):
                            st.write("**Lemma:** "  + m.get("lex",   "N/A"))
                            st.write("**POS:** "    + m.get("pos",   "N/A"))
                            st.write("**Gloss:** "  + m.get("gloss", "N/A"))
                            feats = {k: v for k, v in m.get("features", {}).items() if v}
                            if feats:
                                st.write("**Features:**")
                                for k, v in feats.items():
                                    st.write("  - " + k + ": " + v)

            with tab3:
                st.json(result)

        else:
            st.error("❌ " + str(result["error"]))

    elif analyze:
        st.warning("⚠️ Please enter a sentence first | الرجاء إدخال جملة")

    st.markdown("---")
    st.markdown(
        "<div style=\"text-align:center;color:#aaa;font-size:13px;\">"
        "Built with CAMeL Tools + Gemini API · Powered by Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
