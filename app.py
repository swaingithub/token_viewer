import re
import time
import hashlib
import numpy as np
import streamlit as st
from transformers import AutoTokenizer
import tiktoken
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import chromadb


# --- Sample sentences used for the map and search ---
SAMPLE_SENTENCES = [
    # Food
    ("I love eating pizza.", "Food"),
    ("Sushi is delicious with soy sauce.", "Food"),
    ("Chocolate cake is my favorite dessert.", "Food"),
    ("Fresh fruit salad is healthy and sweet.", "Food"),
    ("Pasta with tomato sauce is a classic meal.", "Food"),
    ("My mom makes the best apple pie on Sundays.", "Food"),
    ("Coffee with a good book starts every morning.", "Food"),
    ("The bakery opens early with warm bread.", "Food"),
    # Sports
    ("I play soccer every weekend.", "Sports"),
    ("Basketball is a fast-paced game.", "Sports"),
    ("Swimming is great exercise for the whole body.", "Sports"),
    ("Tennis requires good hand-eye coordination.", "Sports"),
    ("Running a marathon is a big challenge.", "Sports"),
    ("Cycling around the park feels refreshing.", "Sports"),
    ("Yoga helps me stretch and breathe.", "Sports"),
    ("The local team won the championship.", "Sports"),
    # Technology
    ("I bought a new laptop yesterday.", "Technology"),
    ("Smartphones keep us connected everywhere.", "Technology"),
    ("Artificial intelligence is changing the world.", "Technology"),
    ("Cloud computing stores data online.", "Technology"),
    ("Video games are fun to play with friends.", "Technology"),
    ("Our app crashed, so we debugged it.", "Technology"),
    ("The Wi-Fi password is written on the router.", "Technology"),
    ("Windows Update restarted the computer yesterday.", "Technology"),
    # Learning
    ("Practice makes perfect when you study daily.", "Learning"),
    ("A quiet desk helps you focus while studying.", "Learning"),
    ("Learn a new language with flashcards.", "Learning"),
    ("Taking notes in class improves memory.", "Learning"),
    ("Ask questions when something is unclear.", "Learning"),
    ("Review material before every test.", "Learning"),
    ("Reading aloud is a good way to memorize.", "Learning"),
]

TOPIC_COLORS = {
    "Food": "#059669",
    "Sports": "#2563eb",
    "Technology": "#7c3aed",
    "Learning": "#f59e0b",
    "Your sentence": "#4f46e5",
}


# ==================== Page setup & styling ====================

st.set_page_config(page_title="Follow one sentence", layout="wide")

# Dark theme, one purple accent, neutral grays. Card styling targets the
# st-key-card_* classes that Streamlit adds to keyed containers.
st.markdown(
    """
    <style>
    /* Professional dark palette: deep slate + single violet accent, light neutrals. */
    :root {
      --accent:#4f46e5;
      --accent-hover:#6366f1;
      --accent-soft:rgba(79,70,229,0.16);
      --line:#27304a;
      --text:#e5e7eb;
      --text-dim:#9ca3af;
      --text-muted:#6b7280;
      --bg:#0b0f18;
      --surface:#111727;
      --surface-2:#162037;
      --surface-3:#1a2540;
    }
    :root { color-scheme: dark; }
    body, .stApp { background:var(--bg); }
    .block-container { padding-top:2rem; padding-bottom:2.5rem; max-width:900px; margin:0 auto; }
    .stApp { font-family:'Inter', system-ui, -apple-system, sans-serif; }
    h1 { font-size:2rem; font-weight:800; letter-spacing:-0.02em; margin-bottom:0.25rem; animation:titleIn 0.8s cubic-bezier(.34,1.56,.64,1) both; }
    h2, h3, h4 { letter-spacing:-0.01em; }
    .subtitle { color:var(--text-dim); font-size:1.1rem; margin-bottom:1rem; animation:fade 0.7s ease both; }

    /* Entrances */
    @keyframes cardIn {
      0% {opacity:0; transform:translateY(24px) scale(0.985);}
      70% {transform:translateY(-6px) scale(1.02);}
      100% {opacity:1; transform:translateY(0) scale(1);}
    }
    @keyframes cardPop {
      0% { box-shadow:0 1px 2px rgba(0,0,0,0.35), 0 4px 14px rgba(0,0,0,0.18), 0 0 0 0 rgba(79,70,229,0); }
      45% { box-shadow:0 1px 2px rgba(0,0,0,0.35), 0 4px 14px rgba(0,0,0,0.18), 0 0 0 18px rgba(79,70,229,0.4); }
      100% { box-shadow:0 1px 2px rgba(0,0,0,0.35), 0 4px 14px rgba(0,0,0,0.18), 0 0 0 0 rgba(79,70,229,0); }
    }
    @keyframes dotDrop {
      from {opacity:0; transform:translateY(-8px);}
      to {opacity:1; transform:translateY(0);}
    }
    @keyframes titleIn { from {opacity:0; transform:translateY(18px);} to {opacity:1; transform:translateY(0);} }
    @keyframes pillIn { from {opacity:0; transform:translateY(8px);} to {opacity:1; transform:translateY(0);} }
    @keyframes fade { from {opacity:0;} to {opacity:1;} }
    @keyframes pulse {
      0%,100% { box-shadow:0 0 0 rgba(79,70,229,0); opacity:1; }
      50% { box-shadow:0 0 10px rgba(79,70,229,0.55); opacity:1; }
    }

    /* Progress stepper */
    .stepper { display:flex; align-items:center; gap:5px; flex-wrap:wrap; margin:0 0 0.75rem 0; }
    .stepper .step { display:flex; align-items:center; gap:7px; color:var(--text-muted); font-size:0.8rem; white-space:nowrap; transition:color 0.2s; }
    .stepper .step .dot { width:12px; height:12px; border-radius:50%; background:#27304a; border:1px solid #36435a; transition:background 0.3s, box-shadow 0.3s, transform 0.3s; }
    .stepper .step.active { color:#ede9fe; font-weight:700; }
    .stepper .step.active .dot { background:var(--accent); border-color:var(--accent-hover); animation:pulse 2.2s ease-in-out infinite; }
    .stepper .step.done { color:#c4b5fd; }
    .stepper .step.done .dot { background:var(--accent-soft); border-color:var(--accent); }
    .stepper .line { flex:1 1 16px; min-width:12px; height:2px; background:#27304a; border-radius:2px; }
    .stepper .line.done { background:var(--accent); }

    /* Cards (keyed containers): entrances staggered per card, with a ripple pop on arrival */
    div[class*="st-key-card_"] {
        border:1px solid var(--line) !important;
        background:var(--surface) !important;
        border-radius:12px !important;
        box-shadow:0 1px 2px rgba(0,0,0,0.35), 0 4px 14px rgba(0,0,0,0.18);
        animation:cardIn 0.85s cubic-bezier(.34,1.56,.64,1) both, cardPop 1.6s ease-out;
        margin-bottom:0.45rem;
    }
    div[class*="st-key-card_tokens"]    { animation-delay:0.28s; }
    div[class*="st-key-card_embedding"] { animation-delay:0.42s; }
    div[class*="st-key-card_map"]       { animation-delay:0.55s; }
    div[class*="st-key-card_search"]    { animation-delay:0.68s; }
    div[class*="st-key-card_how"]       { animation-delay:0.22s; }
    div[class*="st-key-recap_box"]      { animation:cardIn 0.5s cubic-bezier(.34,1.56,.64,1) 0.12s both, cardPop 1.6s ease-out 0.3s; }
    div[class*="st-key-card_"]:hover, div[class*="st-key-recap_box"]:hover {
        border-color:var(--accent-soft);
        box-shadow:0 1px 2px rgba(0,0,0,0.35), 0 8px 24px rgba(0,0,0,0.28);
        transform:translateY(-2px);
    }
    .card-head { display:flex; align-items:flex-start; gap:13px; margin-bottom:0.8rem; }
    .card-head .num { flex:0 0 34px; width:34px; height:34px; border-radius:50%; background:var(--accent); color:#fff; font-weight:800; display:flex; align-items:center; justify-content:center; font-size:1.1rem; box-shadow:0 2px 8px rgba(79,70,229,0.35); }
    .card-title { font-size:1.25rem; font-weight:700; color:var(--text); }
    .card-desc { color:var(--text-dim); font-size:0.9rem; margin-top:4px; }

    /* Vertical connector between cards */
    .connector { display:flex; flex-direction:column; align-items:center; padding:5px 0; animation:fade 0.5s ease both; }
    .connector .cdot { width:7px; height:7px; border-radius:50%; background:var(--accent); box-shadow:0 0 8px rgba(79,70,229,0.4); animation:dotDrop 0.5s cubic-bezier(.34,1.56,.64,1) 0.1s both; }
    .connector .cline { width:2px; height:26px; background:var(--line); border-radius:2px; }
    .connector .cdot + .cdot + .cline { animation: pulse 2.2s ease-in-out infinite; }

    /* Flow diagrams for the How it works card */
    .flow { display:flex; align-items:center; justify-content:center; gap:0; flex-wrap:wrap; margin:1rem 0 1.5rem; }
    .flow-box { border:1px solid var(--line); background:var(--surface-2); border-radius:8px; padding:5px 11px; font-size:0.75rem; font-weight:600; text-align:center; min-width:80px; }
    .flow-box .flow-sub { display:block; font-size:0.55rem; font-weight:500; color:var(--text-muted); margin-top:1px; }
    .flow-box.highlight { border-color:var(--accent); color:var(--accent); }
    .flow-arrow { padding:0 5px; font-size:1.2rem; color:var(--accent); }
    /* Sequential reveal of flow boxes and arrows — the "animated way" */
    @keyframes flowReveal {
      from { opacity:0; transform: translateX(-10px); }
      to { opacity:1; transform: translateX(0); }
    }
    .flow-box, .flow-arrow { animation: flowReveal 0.55s cubic-bezier(.22,1,.36,1) both; }
    @keyframes linePulse {
      0%,100% { background: var(--line); }
      50% { background: var(--accent); }
    }
    .connector .cline { animation: linePulse 2.2s ease-in-out infinite; }
    details[open] { animation: fade 0.3s ease both; }

    /* Step 4 (search + recap) with comfortable spacing */
    div[class*="st-key-card_search"] { padding:0.75rem 1.25rem; }
    div[class*="st-key-card_search"] .stMarkdown, div[class*="st-key-card_search"] .stCaption { margin:3px 0; }
    div[class*="st-key-card_search"] h4 { font-size:1.05rem; }
    div[class*="st-key-recap_box"] { padding:0.55rem 1.4rem 0.45rem; }
    div[class*="st-key-recap_box"] .stMarkdown, div[class*="st-key-recap_box"] .stCaption { margin:3px 0; }
    div[class*="st-key-recap_box"] h4 { font-size:1.05rem; }
    div[data-testid="stProgress"] { height:0.3rem !important; margin:1px 0; }
    div[class*="st-key-card_search"] strong { font-size:0.98rem; }
    /* Step 6 (how card) with comfortable spacing */
    div[class*="st-key-card_how"] .flow { margin:0.6rem 0 0.9rem; }
    div[class*="st-key-card_how"] .stMarkdown { margin:4px 0; }
    div[class*="st-key-card_how"] .card-desc { margin-top:4px; }

    /* Token pills */
    .pill { display:inline-block; padding:8px 15px; margin:6px 8px 6px 0; border-radius:8px; font-weight:600; font-size:14px; color:#111827; font-family:ui-monospace, monospace; border:1px solid rgba(0,0,0,0.10); cursor:help; animation:pillIn 0.5s ease both; transition:transform 0.18s, box-shadow 0.18s, border-color 0.18s; }
    .pill:hover { transform:scale(1.05); box-shadow:0 0 12px rgba(79,70,229,0.35); border-color:var(--accent); }
    .pill-id { font-size:10px; opacity:0.75; margin-top:4px; display:block; }

    /* Buttons */
    .stButton>button { background:var(--accent); border:1px solid var(--accent) !important; color:#fff !important; border-radius:8px; font-weight:600; transition:transform 0.18s, box-shadow 0.2s, filter 0.2s; }
    .stButton>button:hover { transform:translateY(-2px); box-shadow:0 6px 20px rgba(79,70,229,0.3); }
    /* Charts fade in when they appear */
    div[data-testid="stPlotlyChart"] { animation:fade 0.8s ease both; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Follow one sentence")
st.markdown('<div class="subtitle">Watch one sentence move through tokens, an embedding, a meaning map, and search.</div>', unsafe_allow_html=True)

# Keep the page from scrolling down on every step: when you click "Next", the
# new card appears right there instead of the page jumping to the bottom.
st.components.v1.html(
    """
    <script>
    (function () {
      function resetScroll() { window.scrollTo(0, 0); }
      // On any DOM change (Streamlit reruns after button clicks), stay at top.
      var observer = new MutationObserver(function () {
        requestAnimationFrame(function () {
          setTimeout(resetScroll, 60);
          setTimeout(resetScroll, 150);
        });
      });
      observer.observe(document.documentElement, { childList: true, subtree: true, attributes: true });
      window.addEventListener("load", resetScroll);
      resetScroll();
    })();
    </script>
    """,
    height=0,
)


# ==================== Helper functions ====================

# --- Tokenizers ---
def get_tokenizer(name):
    """Return a tokenizer object based on the chosen name."""
    if name == "cl100k_base":
        return tiktoken.get_encoding("cl100k_base")
    return AutoTokenizer.from_pretrained(name)


@st.cache_resource
def load_tokenizer(name):
    """Load and cache a tokenizer so it is not reloaded on every run."""
    return get_tokenizer(name)


def clean_token(token):
    """Turn a raw token into readable text (GPT-2 marks spaces with 'Ġ')."""
    token = token.replace("Ġ", " ").replace("Ċ", "↵")
    return token.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tokenize(text, tokenizer, name):
    """Return a list of (token_string, token_id) pairs."""
    if name == "cl100k_base":
        ids = tokenizer.encode(text)
        tokens = [tokenizer.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in ids]
    else:
        ids = tokenizer.encode(text)
        tokens = tokenizer.convert_ids_to_tokens(ids)
    return list(zip(tokens, ids))


def color_for(token_id):
    """Pick a soft pastel pill color from the token id."""
    r = 200 + (token_id * 37) % 55
    g = 200 + (token_id * 101) % 55
    b = 200 + (token_id * 151) % 55
    return f"rgb({r},{g},{b})"


def render_tokens(pairs):
    """Render tokens as pills with a native hover tooltip and staggered entrance."""
    boxes = []
    for idx, (token, tid) in enumerate(pairs):
        safe_token = clean_token(token)
        display = safe_token.replace(" ", "␣") if safe_token.strip() else "␣"
        boxes.append(
            f'<span class="pill" style="background:{color_for(tid)};animation:pillIn 0.5s ease both;animation-delay:{0.05 + idx * 0.10}s" '
            f'title="Token: {display}  |  ID: {tid}">'
            f'{display}<span class="pill-id">ID {tid}</span></span>'
        )
    st.markdown("".join(boxes), unsafe_allow_html=True)


# --- Embeddings ---
@st.cache_resource
def load_embedding_model():
    """Load and cache the sentence-transformers embedding model."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(model, texts):
    """Turn strings into embedding vectors (as Python lists)."""
    return model.encode(texts).tolist()


@st.cache_resource
def get_sample_embeddings():
    """Embeddings for the fixed sample sentences (cached once)."""
    return load_embedding_model().encode([s for s, _ in SAMPLE_SENTENCES])


# --- ChromaDB ---
@st.cache_resource
def get_chroma_collection():
    """Get or create the 'sentences' collection with cosine distance."""
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_or_create_collection(name="sentences", metadata={"hnsw:space": "cosine"})


def load_samples_into_chroma(collection, model):
    """Upsert sample sentences; fixed ids prevent duplicates on re-click."""
    docs = [s for s, _ in SAMPLE_SENTENCES]
    topics = [t for _, t in SAMPLE_SENTENCES]
    ids = [f"sample-{i}" for i in range(len(docs))]
    metadatas = [{"topic": t} for t in topics]
    collection.upsert(documents=docs, ids=ids, embeddings=embed_texts(model, docs), metadatas=metadatas)
    return len(docs)


def reset_database(collection, model):
    """Delete every document in the collection, then reload the samples."""
    _, _, ids = get_all_sentences(collection)
    if ids:
        collection.delete(ids=ids)
    load_samples_into_chroma(collection, model)
    return collection.count()


def add_user_sentences(collection, model, sentences):
    """Add the user's sentences to Chroma with stable ids."""
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0
    ids = ["user-" + hashlib.md5(s.lower().encode()).hexdigest() for s in sentences]
    metadatas = [{"topic": "User"} for _ in sentences]
    collection.upsert(documents=sentences, ids=ids, embeddings=embed_texts(model, sentences), metadatas=metadatas)
    return len(sentences)


def get_all_sentences(collection):
    """Return (documents, metadatas, ids) for everything in Chroma."""
    result = collection.get(include=["documents", "metadatas"])
    return result.get("documents", []), result.get("metadatas", []), result.get("ids", [])


def semantic_search(collection, query_embedding, top_k=3):
    """Return the top_k most similar (doc, meta, distance) tuples."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    return list(zip(docs, metas, dists))


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "i", "you", "it", "to", "of",
    "and", "or", "in", "on", "for", "with", "my", "me", "we", "us", "be", "at",
}


def keyword_search(documents, metadatas, query, top_k=3):
    """Simple word-overlap search that ignores common filler words."""
    query_words = set(re.findall(r"\w+", query.lower())) - STOPWORDS
    if not query_words:
        return []
    scored = []
    for doc, meta in zip(documents, metadatas):
        doc_words = set(re.findall(r"\w+", doc.lower())) - STOPWORDS
        overlap = len(query_words & doc_words)
        if overlap > 0:
            scored.append((overlap, doc, meta))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# ==================== Small UI building blocks ====================

def render_stepper(step, show_all):
    """Draw the Input - Tokens - Embedding - Map - Search - How it works progress stepper."""
    stages = ["Input", "Tokens", "Embedding", "Map", "Search", "How it works"]
    current = 5 if show_all else step
    html = '<div class="stepper">'
    for i, name in enumerate(stages):
        if show_all or i < current:
            cls = "step done"
        elif i == current:
            cls = "step active"
        else:
            cls = "step"
        html += f'<div class="{cls}"><span class="dot"></span><span>{name}</span></div>'
        if i < len(stages) - 1:
            line_cls = "line done" if (show_all or i < current) else "line"
            html += f'<div class="{line_cls}"></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def connector():
    """Thin vertical line with dots between two cards."""
    st.markdown(
        '<div class="connector"><span class="cdot"></span><span class="cline"></span><span class="cdot"></span></div>',
        unsafe_allow_html=True,
    )


def card(number, title, description, key):
    """Open a card container with a numbered circle, title and one-line note."""
    c = st.container(border=True, key=key)
    c.markdown(
        f'<div class="card-head"><span class="num">{number}</span>'
        f'<div><div class="card-title">{title}</div>'
        f'<div class="card-desc">{description}</div></div></div>',
        unsafe_allow_html=True,
    )
    return c


def flow_diagram(items, base_delay=0.03):
    """Render a horizontal flow of boxes connected by arrows.

    items: list of dicts with 'label', optional 'sub' and 'highlight'.
    Elements (boxes and arrows) reveal sequentially with a small delay.
    """
    flat = []
    for i, item in enumerate(items):
        if i > 0:
            flat.append("arrow")
        flat.append(("box", item))
    parts = []
    for n, el in enumerate(flat):
        delay = base_delay + n * 0.09
        if el == "arrow":
            parts.append(f'<span class="flow-arrow" style="animation-delay:{delay:.2f}s">→</span>')
        else:
            item = el[1]
            cls = "flow-box highlight" if item.get("highlight") else "flow-box"
            html = f'<div class="{cls}" style="animation-delay:{delay:.2f}s"><span>{item["label"]}</span>'
            if item.get("sub"):
                html += f'<span class="flow-sub">{item["sub"]}</span>'
            html += "</div>"
            parts.append(html)
    return '<div class="flow">' + "".join(parts) + "</div>"


def embedding_figure(vec, height=90):
    """Render the user's 384-number vector as a hoverable heatmap strip."""
    vec = np.array(vec).reshape(1, -1)
    v = float(np.abs(vec).max()) if np.abs(vec).max() > 0 else 1.0
    fig = go.Figure(
        data=go.Heatmap(
            z=vec,
            colorscale=[[0, "#3b82f6"], [0.5, "#ffffff"], [1, "#ef4444"]],
            zmin=-v,
            zmax=v,
            showscale=False,
            x=list(range(vec.shape[1])),
            y=[""],
            hovertemplate="dimension %{x}<br>value %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    return fig


# ==================== Session state & resets ====================

# Defaults.
if "step" not in st.session_state:
    st.session_state.step = 0
if "sentence_input" not in st.session_state:
    st.session_state.sentence_input = "I love pizza."

# Deferred resets run before any widget is created (safe to touch widget keys).
if st.session_state.pop("do_try_another", False):
    st.session_state.step = 0
    st.session_state.show_all = False
    st.session_state.sentence_input = ""
if st.session_state.pop("do_start_over", False):
    st.session_state.step = 0
    st.session_state.show_all = False
    st.session_state.sentence_input = "I love pizza."

step = st.session_state.step
show_all = bool(st.session_state.get("show_all", False))


# ==================== Top: stepper & controls ====================

render_stepper(step, show_all)

ctrl1, ctrl2 = st.columns([1, 1])
with ctrl1:
    if st.button("Start over", use_container_width=True):
        st.session_state.do_start_over = True
        st.rerun()
with ctrl2:
    st.toggle("Show all", key="show_all")


# ==================== Step 1: Input ====================

input_card = card(1, "Your sentence", "Type a sentence, or pick an example to start.", "card_input")
with input_card:
    ex1, ex2, ex3 = st.columns(3)
    if ex1.button("I love pizza.", use_container_width=True):
        st.session_state.sentence_input = "I love pizza."
        st.rerun()
    if ex2.button("The cat sat on the mat.", use_container_width=True):
        st.session_state.sentence_input = "The cat sat on the mat."
        st.rerun()
    if ex3.button("Learning AI step by step", use_container_width=True):
        st.session_state.sentence_input = "Learning AI step by step"
        st.rerun()
    sentence = st.text_input("Sentence", key="sentence_input")

# Load shared resources once.
model = load_embedding_model()
collection = get_chroma_collection()
count = collection.count()

# Auto-load samples on first run so the database is never empty.
if count == 0:
    load_samples_into_chroma(collection, model)
    count = collection.count()

# Accordion flow: only the current step card is visible unless "Show all"
# is on — clicking Next makes the new card slide in right where you are.
tokens_visible = show_all or step == 1
embed_visible = show_all or step == 2
map_visible = show_all or step == 3
search_visible = show_all or step == 4

pairs = []
query_embedding = None

# Reveal button after the input card.
if not show_all and step == 0:
    if st.button("Next: tokenize", type="primary"):
        st.session_state.step = 1
        st.rerun()


# ==================== Step 2: Tokens ====================

if tokens_visible:
    connector()
    tokens_card = card(2, "Tokens", "Your sentence split into pieces by a tokenizer.", "card_tokens")
    with tokens_card:
        tokenizer_name = st.selectbox("Tokenizer", ["gpt2", "bert-base-uncased", "cl100k_base"], key="tokenizer_select")
        try:
            pairs = tokenize(sentence, load_tokenizer(tokenizer_name), tokenizer_name)
            render_tokens(pairs)
            st.caption("␣ means the token starts with a space. Hover a pill to see its ID.")
            st.markdown(f"**Total tokens:** {len(pairs)}")
        except Exception as e:
            st.error(f"Could not load tokenizer '{tokenizer_name}': {e}")

        with st.expander("Why?"):
            st.markdown("**Why are some words split into pieces?**")
            st.write(
                "Tokenizers break text into smaller units so the model can handle rare or long words. "
                "Common words often stay whole, while less common ones split into pieces. "
                "A smaller vocabulary keeps the model efficient."
            )

        if not show_all and step == 1:
            if st.button("Next: embed", type="primary"):
                st.session_state.step = 2
                st.rerun()


# ==================== Step 3: Embedding ====================

if embed_visible:
    connector()
    query_embedding = embed_texts(model, [sentence])[0]

    embed_card = card(3, "Embedding", "One list of numbers that captures the meaning of your sentence.", "card_embedding")
    with embed_card:
        st.plotly_chart(embedding_figure(query_embedding), use_container_width=True)
        st.caption("Blue = negative, white = zero, red = positive. Hover a cell to see its value.")

        with st.expander("Why?"):
            st.markdown("**What do these 384 numbers mean?**")
            st.write(
                "Each number is one coordinate in a 384-dimension space. "
                "The model learns coordinates so that sentences with similar meaning land near each other. "
                "No single number has a simple meaning on its own."
            )

        if not show_all and step == 2:
            if st.button("Next: map", type="primary"):
                st.session_state.step = 3
                st.rerun()


# ==================== Step 4: Meaning map ====================

if map_visible:
    connector()
    map_card = card(4, "Meaning map", "A flat view of where sentences sit in meaning space.", "card_map")
    with map_card:
        if query_embedding is None:
            query_embedding = embed_texts(model, [sentence])[0]
        sample_embeddings = get_sample_embeddings()
        pca = PCA(n_components=2)
        coords = pca.fit_transform(sample_embeddings)
        user_coords = pca.transform(np.array([query_embedding]))

        fig = go.Figure()
        for topic in ["Food", "Sports", "Technology", "Learning"]:
            indices = [i for i, (_, t) in enumerate(SAMPLE_SENTENCES) if t == topic]
            fig.add_trace(
                go.Scatter(
                    x=coords[indices, 0],
                    y=coords[indices, 1],
                    mode="markers",
                    customdata=[SAMPLE_SENTENCES[i][0] for i in indices],
                    hovertemplate="%{customdata}<extra></extra>",
                    marker=dict(size=14, color=TOPIC_COLORS[topic], line=dict(width=1.5, color="#2a3445")),
                    name=topic,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[user_coords[0, 0]],
                y=[user_coords[0, 1]],
                mode="markers+text",
                text=[sentence],
                textposition="top center",
                textfont=dict(size=13, color="#ffffff", family="Arial Black"),
                marker=dict(size=30, color="#4f46e5", symbol="star", line=dict(width=3, color="#ffffff")),
                name="Your sentence",
            )
        )
        fig.update_layout(
            height=300,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="system-ui, -apple-system, sans-serif", size=12, color="#e5e7eb"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, tickfont=dict(size=11, color="#9ca3af")),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, tickfont=dict(size=11, color="#9ca3af")),
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)", font=dict(size=12, color="#e5e7eb")),
            margin=dict(l=60, r=20, t=30, b=80),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Simplified 2D view. Real embeddings have hundreds of dimensions.")

        with st.expander("Why?"):
            st.markdown("**Why do similar sentences sit close together?**")
            st.write(
                "The embedding model places sentences so that related meanings share similar coordinates. "
                "When we flatten those coordinates to 2D, near neighbors on the map tend to be near neighbors in meaning. "
                "The map is a rough sketch, not an exact distance."
            )

        if not show_all and step == 3:
            if st.button("Next: search", type="primary"):
                st.session_state.step = 4
                st.rerun()


# ==================== Step 5: Search ====================

if search_visible:
    connector()
    # In accordion mode the embedding card is hidden, but the search card
    # needs the same data — compute it on demand.
    if query_embedding is None:
        query_embedding = embed_texts(model, [sentence])[0]
    if not pairs:
        _tok = st.session_state.get("tokenizer_select", "gpt2")
        pairs = tokenize(sentence, load_tokenizer(_tok), _tok)
    docs, metadatas, _ = get_all_sentences(collection)
    top_k = min(3, count)

    # Measure just the vector search so the recap shows a real number.
    started = time.perf_counter()
    results = semantic_search(collection, query_embedding, top_k=top_k)
    search_time = time.perf_counter() - started

    keyword_results = keyword_search(docs, metadatas, sentence, top_k=3)

    search_card = card(5, "Search", "Find the closest sentences by meaning, next to a keyword search.", "card_search")
    with search_card:
        st.markdown("#### You just ran a tiny semantic search engine.")

        col_meaning, col_keyword = st.columns(2)
        with col_meaning:
            st.markdown("**By meaning**")
            if results:
                for n, (doc, meta, dist) in enumerate(results):
                    topic = meta.get("topic", "?") if meta else "?"
                    similarity = round(1 - dist, 2)
                    d = 0.15 + n * 0.07
                    st.markdown(
                        f'<div style="animation:fade 0.4s ease both; animation-delay:{d:.2f}s">'
                        f'<b>{doc}</b> — {topic} · similarity {similarity:.2f}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No results.")

        with col_keyword:
            st.markdown("**By keyword**")
            if keyword_results:
                for n, (overlap, doc, meta) in enumerate(keyword_results):
                    topic = meta.get("topic", "?") if meta else "?"
                    d = 0.15 + n * 0.07
                    st.markdown(
                        f'<div style="animation:fade 0.4s ease both; animation-delay:{d:.2f}s">'
                        f'<b>{doc}</b> — {topic} · {overlap} shared word(s)</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No shared words found")

        with st.expander("Why?"):
            st.markdown("**Why did meaning search find results that keywords missed?**")
            st.write(
                "Meaning search compares vectors, so it can match paraphrases that share no exact words. "
                "Keyword search only counts exact word overlaps, so it misses synonyms and rephrasing. "
                "That is why the two columns can disagree."
            )

        # Recap card with real measured values (part of Step 5, so it only
        # shows when the search card is visible).
        n_matches = len(results)
        time_str = "<0.01 s" if search_time < 0.005 else f"{search_time:.2f} s"
        recap = st.container(border=True, key="recap_box")
        with recap:
            st.markdown("#### Recap")
            st.markdown(f"### {len(pairs)} tokens → {len(query_embedding)} numbers → {n_matches} matches in {time_str}")
            next_col, try_col = st.columns([1.3, 1])
            with next_col:
                if not show_all and step == 4 and st.button("Next: how it works", type="primary"):
                    st.session_state.step = 5
                    st.rerun()
            with try_col:
                if st.button("Try another sentence", key="try_another"):
                    st.session_state.do_try_another = True
                    st.rerun()


# ==================== Step 6: How it works ====================

how_visible = show_all or step == 5

if how_visible:
    connector()
    how_card = card(6, "How it works", "The pipeline behind your sentence, visualized.", "card_how")
    with how_card:
        if query_embedding is None:
            query_embedding = embed_texts(model, [sentence])[0]
        if not pairs:
            _tok = st.session_state.get("tokenizer_select", "gpt2")
            pairs = tokenize(sentence, load_tokenizer(_tok), _tok)
        st.markdown(
            "The map in Step 4 shows every sentence in 2D — near neighbors on the plot tend to be similar in meaning. "
            "Everything below is what happens behind that, step by step. All of it runs locally: the tokenizer, "
            "the embedding model and the ChromaDB database are on your machine.",
            unsafe_allow_html=True,
        )

        st.markdown("**The pipeline**", unsafe_allow_html=True)
        st.markdown(
            flow_diagram([
                {"label": "Sentence", "sub": "your words"},
                {"label": "Tokenizer", "sub": "pieces"},
                {"label": "Token IDs", "sub": "numbers"},
                {"label": "Embedding", "sub": "384-number vector"},
                {"label": "Vector DB", "sub": "ChromaDB, cosine"},
                {"label": "Search", "sub": "top matches"},
            ], base_delay=0.03),
            unsafe_allow_html=True,
        )

        st.markdown("**Meaning search flow**", unsafe_allow_html=True)
        st.markdown(
            flow_diagram([
                {"label": "Your vector"},
                {"label": "ChromaDB query", "sub": "cosine similarity", "highlight": True},
                {"label": "Top 3", "sub": "nearest by similarity", "highlight": True},
            ], base_delay=0.65),
            unsafe_allow_html=True,
        )
        results = semantic_search(collection, query_embedding, top_k=3)
        if results:
            for n, (doc, meta, dist) in enumerate(results):
                topic = meta.get("topic", "?") if meta else "?"
                similarity = round(1 - dist, 2)
                d = 1.7 + n * 0.08
                st.markdown(
                    f'<div style="animation:fade 0.4s ease both; animation-delay:{d:.2f}s">'
                    f"<b>{doc}</b> — {topic} · similarity {similarity:.2f}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No results.")

        st.markdown("**Keyword search flow**", unsafe_allow_html=True)
        st.markdown(
            flow_diagram([
                {"label": "Query words", "highlight": True},
                {"label": "Document words", "highlight": True},
                {"label": "Shared words", "sub": "exact matches", "highlight": True},
            ], base_delay=2.2),
            unsafe_allow_html=True,
        )
        docs, metadatas, _ = get_all_sentences(collection)
        keyword_results = keyword_search(docs, metadatas, sentence, top_k=3)
        if keyword_results:
            for n, (overlap, doc, meta) in enumerate(keyword_results):
                topic = meta.get("topic", "?") if meta else "?"
                d = 3.1 + n * 0.08
                st.markdown(
                    f'<div style="animation:fade 0.4s ease both; animation-delay:{d:.2f}s">'
                    f"<b>{doc}</b> — {topic} · {overlap} shared word(s)</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("**Your sentence, step by step**", unsafe_allow_html=True)
        st.markdown(f"Sentence: `{sentence}`", unsafe_allow_html=True)
        if not pairs:
            pairs = tokenize(sentence, load_tokenizer(tokenizer_name), tokenizer_name)
        render_tokens(pairs)
        st.caption("␣ means the token starts with a space.")
        st.markdown(f"**Total tokens:** {len(pairs)}", unsafe_allow_html=True)

        vec = np.array(query_embedding if query_embedding is not None else embed_texts(model, [sentence])[0])
        st.plotly_chart(embedding_figure(vec, height=50), use_container_width=True)
        st.caption("Your vector as stored in the database — blue = negative, red = positive, hover for the value.")
        st.caption(f"ChromaDB: {collection.count()} documents, each with its 384-number vector.")

        with st.expander("Why?"):
            st.markdown("**Why are some words split into pieces?**")
            st.write(
                "Tokenizers break text into smaller units so the model can handle rare or long words. "
                "Common words often stay whole, while less common ones split into pieces. "
                "A smaller vocabulary keeps the model efficient."
            )
        with st.expander("Why?"):
            st.markdown("**What do these 384 numbers mean?**")
            st.write(
                "Each number is one coordinate in a 384-dimension space. "
                "The model learns coordinates so that sentences with similar meaning land near each other. "
                "No single number has a simple meaning on its own."
            )
        with st.expander("Why?"):
            st.markdown("**Why do similar sentences sit close together?**")
            st.write(
                "The embedding model places sentences so that related meanings share similar coordinates. "
                "When we flatten those coordinates to 2D, near neighbors on the map tend to be near neighbors in meaning. "
                "The map is a rough sketch, not an exact distance."
            )
        with st.expander("Why?"):
            st.markdown("**Why did meaning search find results that keywords missed?**")
            st.write(
                "Meaning search compares vectors, so it can match paraphrases that share no exact words. "
                "Keyword search only counts exact word overlaps, so it misses synonyms and rephrasing. "
                "That is why the two columns can disagree."
            )


# ==================== Manage database ====================

with st.expander("Manage database", expanded=False):
    st.markdown("Load, add or reset sentences in the vector database.")
    col_load, col_reset, col_add = st.columns(3)

    with col_load:
        if st.button("Load sample sentences", key="load_samples", use_container_width=True):
            n = load_samples_into_chroma(collection, model)
            st.success(f"Loaded/updated {n} sample sentences.")
            st.rerun()

    with col_reset:
        if st.button("Reset database", key="reset_db", use_container_width=True):
            n = reset_database(collection, model)
            # Shown outside the expander so it stays visible after the rerun.
            st.session_state.reset_msg = f"Deleted everything and reloaded {n} sample sentences."
            st.rerun()

    with col_add:
        new_text = st.text_area(
            "Add your sentences (one per line)",
            placeholder="I love pizza.\nPython is great.",
            key="add_sentences",
            height=90,
        )
        if st.button("Add to database", key="add_to_db", use_container_width=True):
            n = add_user_sentences(collection, model, new_text.splitlines())
            if n:
                st.success(f"Added {n} sentence(s).")
                st.rerun()
            else:
                st.warning("No sentences to add.")

# Feedback for the database reset, shown outside the expander.
_reset_msg = st.session_state.pop("reset_msg", None)
if _reset_msg:
    st.success(_reset_msg)
