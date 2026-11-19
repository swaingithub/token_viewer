# token-viewer

A small Streamlit app that follows **one sentence** through the whole pipeline:
tokens, embedding, a 2D meaning map, and finally a real vector search compared
with a keyword search.

The page is a single dark, professional theme — deep slate background
(`#0b0f18`), one indigo accent (`#4f46e5`), light-gray card outlines — with
entrances and hover effects throughout.

```
text --> tokenizer --> token IDs --> embedding --> vector database --> results
```

The page is a single dark-themed flow made of five numbered cards:

```
1 Input  ->  2 Tokens  ->  3 Embedding  ->  4 Map  ->  5 Search + Recap
```

A progress stepper at the top (Input · Tokens · Embedding · Map · Search) shows
where you are. Cards are revealed one at a time with a "Next" button, or all at
once with the "Show all" toggle. "Start over" and "Try another sentence" reset
the flow.

## What each card shows

**1. Your sentence**
- Type a sentence, or pick one of three example buttons.
- "Next: tokenize" reveals the next card.

**2. Tokens**
- Choose a tokenizer: `gpt2`, `bert-base-uncased`, or `cl100k_base`.
- Tokens appear as colored pills with their IDs; hover a pill for a tooltip.
- `␣` marks a token that starts with a space.

**3. Embedding**
- Uses `all-MiniLM-L6-v2` to turn the sentence into 384 numbers.
- Shown as a heatmap strip: blue = negative, white = zero, red = positive.
- Hover a cell to see the exact value.

**4. Meaning map**
- The 31 sample sentences plus your sentence are reduced to 2D with PCA and
  plotted, colored by topic (Food, Sports, Technology, Learning).
- Your sentence is the star; hover a dot to read its sentence.
- Caption: *"Simplified 2D view. Real embeddings have hundreds of dimensions."*

**5. Search + Recap**
- "By meaning" (ChromaDB, cosine distance) side by side with "By keyword"
  (simple word overlap). If keywords find nothing, it says
  "No shared words found".
- A recap card shows the real measured numbers, for example:
  `4 tokens → 384 numbers → 3 matches in <0.01 s`
- Search time is measured with `time.perf_counter`.

Every step also has a "Why?" expander answering one question in two or three
sentences (token splitting, what the numbers mean, why similar sentences sit
close together, and why meaning search beats keywords).

Cards fade in and slide up as you advance (each one on its own staggered
delay), token pills pop in one by one, chart visuals fade in, the active
stepper dot pulses, buttons lift on hover, and cards lift slightly on hover.
That "animated way" is the point: you can watch each stage appear.

## Database

- ChromaDB is stored locally in `chroma_db/` (cosine distance).
- On first run, if the collection is empty, the 31 sample sentences are loaded
  automatically so the app never starts empty.
- The "Manage database" expander still lets you reload samples or add your own
  sentences (one per line). Sample ids are fixed, so reloading does not create
  duplicates.
- "Reset database" deletes everything in the collection and reloads the 31
  sample sentences. A success message appears at the top of the page.

## Install

macOS uses an externally managed Python, so create a virtual environment inside
the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

To stop the app, press `Ctrl+C` in the terminal.

## Theme

`.streamlit/config.toml` forces the dark theme:

```toml
[theme]
base = "dark"
primaryColor = "#4f46e5"
backgroundColor = "#0b0f18"
secondaryBackgroundColor = "#111727"
textColor = "#e5e7eb"
font = "sans serif"
```

## Models and tools

- `gpt2` — Hugging Face `transformers` AutoTokenizer
- `bert-base-uncased` — Hugging Face `transformers` AutoTokenizer
- `cl100k_base` — `tiktoken` (the tokenizer used by GPT-3.5 / GPT-4)
- `all-MiniLM-L6-v2` — `sentence-transformers` embedding model (384 dimensions)
- `chromadb` — local vector database (persisted in `chroma_db/`)
- `plotly` — the embedding strip and the meaning map

## Limitations

- The sample data is small (15 sentences), so clusters and search results are
  illustrative rather than representative.
- PCA flattens 384 dimensions to 2D, so plot distances are only an
  approximation of real meaning distance.
- Keyword search only counts exact word overlaps (common words like "the" are
  ignored), so it misses synonyms and paraphrases by design.
- Deleting the `chroma_db/` folder resets the database; it will be refilled with
  the samples on the next run.
