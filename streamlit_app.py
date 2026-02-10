import streamlit as st
import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai
import pyarabic.araby as araby

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

# ── Load Gemini ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_gemini_client():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found.")
        st.stop()
    return genai.Client(api_key=api_key)

# ── PyArabic Preprocessing ────────────────────────────────────────────────────
def preprocess_arabic(sentence):
    """
    Extract rich linguistic features from Arabic text using PyArabic.
    These features are passed as structured context to Gemini.
    """
    tokens = araby.tokenize(sentence)
    word_features = []

    for token in tokens:
        # Strip diacritics for base form
        stripped     = araby.strip_tashkeel(token)
        # Remove only last haraka (useful for case detection)
        no_last      = araby.strip_lastharaka(token)
        # Normalize letter variants (alef, hamza, etc.)
        normalized   = araby.normalize_ligature(araby.normalize_hamza(stripped))
        # Detect definite article
        has_al       = araby.has_alef_lam(stripped) if hasattr(araby, 'has_alef_lam') else stripped.startswith(araby.ALEF + araby.LAM)
        # Sun/moon letter detection for words with ال
        is_sun       = False
        if has_al and len(stripped) > 2:
            is_sun = araby.is_sun(stripped[2])
        # Check if pure Arabic
        is_arabic    = araby.is_arabicrange(token[0]) if token else False

        word_features.append({
            "token"     : token,
            "stripped"  : stripped,
            "normalized": normalized,
            "no_last"   : no_last,
            "has_al"    : has_al,
            "is_sun"    : is_sun,
            "is_arabic" : is_arabic,
        })

    return word_features

def format_features_for_prompt(word_features):
    """
    Format PyArabic features into a readable string for the Gemini prompt.
    """
    lines = []
    for f in word_features:
        al_info = ""
        if f["has_al"]:
            al_info = " | has ال (definite)" + (" + sun letter assimilation" if f["is_sun"] else " + moon letter")
        lines.append(
            "- " + f["token"] +
            " | base: "       + f["stripped"] +
            " | normalized: " + f["normalized"] +
            al_info
        )
    return "\n".join(lines)

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """أنت خبير في النحو العربي والإعراب.
You are an expert in Arabic grammar and irab (grammatical analysis).

You will be given an Arabic sentence along with preprocessed linguistic features
extracted by PyArabic (tokenization, normalization, definite article detection,
sun/moon letter classification). Use these features as additional context.

For each word provide:
1. الإعراب   — grammatical role (فاعل، مفعول به، مبتدأ، خبر، فعل ماض، حرف جر، etc.)
2. العلامة   — grammatical marker (مرفوع بالضمة، منصوب بالفتحة، مجرور بالكسرة، مبني، etc.)
3. التفاصيل  — details (definiteness, gender, number, verb tense, etc.)
4. explanation — one clear sentence in English

Return ONLY a valid JSON array, no markdown, no text outside the array."""

def create_prompt(sentence, word_features):
    features_str = format_features_for_prompt(word_features)
    return (
        SYSTEM_PROMPT
        + "\n\nSentence: " + sentence
        + "\n\nPyArabic linguistic features:\n" + features_str
        + "\n\nReturn ONLY a JSON array:\n"
        + "[{\"word\":\"...\",\"irab\":\"...\",\"sign\":\"...\",\"details\":\"...\",\"explanation\":\"...\"}]"
    )

# ── Analysis ──────────────────────────────────────────────────────────────────
def get_irab(sentence, word_features, client):
    prompt = create_prompt(sentence, word_features)
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
        except json.JSONDecodeError:
            return {"success": False, "error": "Could not parse Gemini response. Try again."}
        except Exception as e:
            err = str(e)
            if "429" in err:
                if "PerDay" in err:
                    return {"success": False, "error": "Daily quota reached. Try again tomorrow."}
                time.sleep(60 * (attempt + 1))
            else:
                return {"success": False, "error": err}
    return {"success": False, "error": "Max retries exceeded."}

@st.cache_data(show_spinner=False)
def run_full_analysis(sentence):
    client        = load_gemini_client()
    word_features = preprocess_arabic(sentence)
    irab_resp     = get_irab(sentence, word_features, client)
    return {
        "original"    : sentence,
        "word_features": word_features,
        "irab"        : irab_resp.get("data", []),
        "success"     : irab_resp["success"],
        "error"       : irab_resp.get("error")
    }

# ── UI Helpers ────────────────────────────────────────────────────────────────
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
        "<div style=\"background:#fff;border-right:6px solid " + color + ";"
        "border-radius:10px;padding:16px 20px;margin:10px 0;"
        "box-shadow:0 2px 6px rgba(0,0,0,0.08);direction:rtl;\">"
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

        **Pipeline:**
        - PyArabic → tokenization, normalization,
          definite article & sun/moon letter detection
        - Gemini 2.0 Flash → full irab analysis
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
            "كان الطقس جميلاً في الصباح",
        ]
        for ex in examples:
            if st.button(ex, key="ex_" + ex, use_container_width=True):
                st.session_state.input_text = ex
                st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
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
                "🔬 Preprocessing | المعالجة",
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
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True
                )
                st.markdown("---")
                st.markdown("#### تفصيل كل كلمة")
                for w in result["irab"]:
                    st.markdown(word_card(w), unsafe_allow_html=True)

            with tab2:
                st.markdown("#### PyArabic Preprocessing Features")
                st.markdown("These features were extracted before sending to Gemini:")
                pre_rows = []
                for f in result["word_features"]:
                    pre_rows.append({
                        "Token"      : f["token"],
                        "Base Form"  : f["stripped"],
                        "Normalized" : f["normalized"],
                        "Definite ال": "✓" if f["has_al"] else "",
                        "Sun Letter" : "✓" if f["is_sun"] else "",
                    })
                st.dataframe(
                    pd.DataFrame(pre_rows),
                    use_container_width=True,
                    hide_index=True
                )

            with tab3:
                st.json(result)

        else:
            st.error("❌ " + str(result["error"]))

    elif analyze:
        st.warning("⚠️ Please enter a sentence first | الرجاء إدخال جملة")

    st.markdown("---")
    st.markdown(
        "<div style=\"text-align:center;color:#aaa;font-size:13px;\">"
        "Built with PyArabic + Gemini 2.0 Flash · Powered by Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
