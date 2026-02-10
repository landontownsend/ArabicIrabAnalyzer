# 📚 Arabic I'rab Analyzer | محلل الإعراب العربي

> A web application that breaks down Arabic sentences word by word, explaining the grammatical role of every word — like a grammar teacher that never sleeps.

**[🚀 Live Demo](https://arabicirabanalyzer.streamlit.app)** &nbsp;|&nbsp; **[💻 GitHub](https://github.com/landontownsend/ArabicIrabAnalyzer)**

---

## What Is This?

Arabic grammar has a concept called **إعراب (I'rab)** — a system where every single word in a sentence is assigned a precise grammatical role, and the ending of the word actually *changes* depending on that role. Think of it like grammatical case endings in Latin or German, but more intricate and applied to every word in the sentence.

For example, take the sentence:

> **ذهب الولد إلى المدرسة**
> *"The boy went to school"*

In English grammar, we might say *"went"* is a verb and *"boy"* is the subject. Arabic I'rab takes this much further — for each word you need to state:

- Its **grammatical role** (subject, object, predicate, etc.)
- The **case marker** that proves that role (a vowel sound at the end of the word)
- Extra details like whether it's definite or indefinite, masculine or feminine, singular or plural
- How it connects to other words in the sentence

For students of Arabic, writing out this analysis — called *doing the i'rab* — is a foundational skill taught in every Arabic grammar course, from beginner to advanced. It's also notoriously time-consuming and difficult to self-check.

This tool automates that process for any Arabic sentence you throw at it.

---

## Who Is This For?

- **Arabic language students** at any level who want to check their grammatical analysis
- **Teachers and tutors** looking to quickly generate example analyses
- **Researchers** working with Arabic text who need grammatical breakdowns
- **Linguists and NLP practitioners** interested in Arabic morphology
- **Anyone curious** about how Arabic grammar works

You don't need to know Arabic to appreciate what this tool is doing technically — the pipeline is described in full below.

---

## Features

- **Paste any Arabic sentence** — voweled (with diacritics) or unvoweled (without)
- **Full I'rab breakdown** for every word in the sentence
- **Color-coded word cards** grouped by grammatical role
- **Summary table** for quick scanning
- **Preprocessing tab** showing exactly what the NLP layer extracted before AI analysis
- **Bilingual output** — Arabic grammatical terms with English explanations
- **Example sentences** to explore right away in the sidebar

---

## How It Works

The app uses a two-stage NLP pipeline. Rather than just sending the sentence directly to an AI model, it first runs the text through a dedicated Arabic language processing library to extract structured linguistic features. Those features then accompany the sentence into the AI model as enriched context, improving the accuracy of the analysis.

### Stage 1 — PyArabic Preprocessing

[PyArabic](https://github.com/linuxscout/pyarabic) is a Python library built specifically for Arabic text processing. It handles low-level operations that are unique to Arabic as a language:

| Feature | What It Does | Why It Matters |
|---|---|---|
| **Tokenization** | Splits the sentence into individual words | Arabic script is cursive — words connect, making splitting non-trivial |
| **Diacritic stripping** | Removes vowel markers to get the base form | The same word appears differently when voweled vs. unvoweled |
| **Normalization** | Standardizes letter variants (e.g. different hamza forms: أ إ آ → ا) | The same word can be spelled multiple ways |
| **Definite article detection** | Identifies the Arabic equivalent of "the" (ال) | Definiteness is a core grammatical category in Arabic |
| **Sun/Moon letter classification** | Detects how ال assimilates to the following letter | Affects pronunciation and is a marker of certain grammatical structures |

The output of this stage is a structured feature set for every word — not a guess, just facts about the text extracted by rule-based linguistic algorithms.

### Stage 2 — Gemini 2.0 Flash Analysis

The preprocessed features plus the original sentence are passed to [Google's Gemini 2.0 Flash](https://deepmind.google/models/gemini/) model with a carefully engineered prompt that instructs it to act as an Arabic grammar expert. The model returns a structured JSON response with the full I'rab for each word.

Using PyArabic's features as context means Gemini receives grounded linguistic information rather than having to infer everything from the raw text alone — this is especially important for unvoweled Arabic, where a single word can have multiple valid readings depending on context.

### Pipeline Diagram

```
User Input (Arabic sentence)
          │
          ▼
┌─────────────────────┐
│   PyArabic Layer    │
│  - Tokenization     │
│  - Normalization    │
│  - Feature Extract  │
└─────────┬───────────┘
          │  Structured features
          ▼
┌─────────────────────┐
│  Gemini 2.0 Flash   │
│  - Grammatical role │
│  - Case markers     │
│  - Full I'rab       │
└─────────┬───────────┘
          │  JSON response
          ▼
┌─────────────────────┐
│   Streamlit UI      │
│  - Color cards      │
│  - Summary table    │
│  - Preprocessing    │
│    tab              │
└─────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | [Streamlit](https://streamlit.io) | Web UI framework |
| **Arabic NLP** | [PyArabic](https://github.com/linuxscout/pyarabic) | Morphological preprocessing |
| **AI Analysis** | [Gemini 2.0 Flash](https://ai.google.dev) | I'rab generation |
| **Language** | Python 3.10+ | Core application |
| **Deployment** | Streamlit Community Cloud | Hosting |

---

## Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/landontownsend/ArabicIrabAnalyzer.git
cd ArabicIrabAnalyzer
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set your API key**

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
```
Get a free API key at [aistudio.google.com](https://aistudio.google.com).

**4. Run the app**
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
ArabicIrabAnalyzer/
├── streamlit_app.py      # Main application — all logic and UI
├── requirements.txt      # Python dependencies
├── .env.example          # API key template
├── .gitignore            # Keeps secrets out of GitHub
└── .streamlit/
    └── config.toml       # Theme and server configuration
```

---

## A Note on Arabic Grammar

For those unfamiliar with Arabic, here is a quick primer on what I'rab actually is and why it's significant.

Arabic is a **synthetic language**, meaning grammatical relationships between words are expressed through changes to the words themselves (usually their endings) rather than through word order. This is the opposite of English, which relies heavily on word order — "the dog bit the man" means something very different from "the man bit the dog," and the words themselves don't change.

In Arabic, you could theoretically reorder the words in a sentence and still convey the same meaning, because the grammatical role of each word is encoded in its ending. The subject of a sentence takes a **damma** (ُ) — a small "u" sound — at the end. The object takes a **fatha** (َ) — an "a" sound. The object of a preposition takes a **kasra** (ِ) — an "i" sound.

I'rab is the formal practice of identifying and explaining these endings and roles for every word. It is taught in Arabic schools from a young age and is considered a mark of fluency and education. Classical texts — the Quran, poetry, literature — are particularly analyzed this way, since precise grammatical knowledge affects both meaning and recitation.

---

## Acknowledgements

- [PyArabic](https://github.com/linuxscout/pyarabic) by Taha Zerrouki — the foundational Arabic NLP library that powers the preprocessing stage
- [Google Gemini](https://deepmind.google/models/gemini/) for AI-powered grammatical analysis
- [Streamlit](https://streamlit.io) for making Python web apps genuinely enjoyable to build
- Every Arabic grammar teacher whose patience made this feel worth building

---

## License

Free to use, modify, and build on.

## Contact

landon20@umd.edu | landontownsend20@gmail.com
