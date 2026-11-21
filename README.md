# Follow One Sentence

A small Streamlit app that follows **one sentence** through the steps that happen before an AI model can search by meaning: tokens, an embedding, a 2D meaning map, and a vector search shown next to a keyword search.

**Live demo:** https://tokenviewer.streamlit.app/
*(The first load can take a few seconds while the models start up.)*

![Demo](docs/demo.gif)
<!-- Add a short GIF or screenshot at docs/demo.gif -->

```
text -> tokenizer -> token IDs -> embedding -> vector database -> results
```

## What you will see

The flow is **accordion-style**: the input card is always visible, and each step
shows one card at a time — clicking "Next" makes the new step card slide in
right below the input card, and the page never scrolls. Use the "Show all"
toggle to see every card at once. Click "Start over" or "Try another sentence"
to begin again.

| Step | What happens |
|---|---|
| 1. Your sentence | Type a sentence or pick an example. |
| 2. Tokens | The sentence is split into tokens. Choose `gpt2`, `bert-base-uncased` or `cl100k_base`. Each pill shows the token and its ID (`␣` means the token starts with a space). |
| 3. Embedding | `all-MiniLM-L6-v2` turns the whole sentence into 384 numbers. Blue is negative, white is zero, red is positive. Hover a cell for its value. |
| 4. Meaning map | About 30 sample sentences and yours are flattened to 2D with PCA and colored by topic (Food, Sports, Technology, Learning). Your sentence is the star. |
| 5. Search + recap | Search by meaning (ChromaDB, cosine similarity) next to a simple keyword search, plus a recap of the real measured numbers. |
| 6. How it works | A visual pipeline of the data flow: Sentence → Tokenizer → Token IDs → Embedding → Vector DB → Search. Plus flow diagrams for the meaning/keyword search paths, the real search results, the tokenizer comparison (gpt2/bert/cl100k), and the live results. |

Each step has a "Why?" panel with a short explanation.

## Animations — the app is deliberately animated to feel like learning

- New step cards arrive with a **spring bounce slide-in** and a **ripple pop** around the card ("milestone reach" feel); page titles, subtitles and connector dots fade in as well.
- Flow diagrams (in the "How it works" card) reveal their boxes and arrows one after another, sliding in from the left.
- The result lists fade in staggered — the search results, the keyword results, and the real results inside "How it works".
- Connector lines pulse, and the active stepper dot pulses.
- Cards already slide up as you advance.

## Good to know

- The Tokens card uses three popular tokenizers so you can compare them. The embedding model has its own tokenizer internally, so the token count in the recap comes from the tokenizer you selected, not necessarily the one inside `all-MiniLM-L6-v2`.
- The embedding card shows one vector for the whole sentence. The model works with a vector per token internally and combines them into this one.
- This app shows the retrieval side of many AI apps (search and RAG). It does not generate text like a chatbot.

## Run it locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). The first run downloads the models, so it needs an internet connection. Stop the app with `Ctrl+C`.

## Database

- ChromaDB stores the sentences in a per-visitor in-memory database (`chromadb.EphemeralClient()`, cosine distance) — nothing is shared between visitors or written to disk.
- On first run, if the collection is empty, the sample sentences load automatically.
- Open "Manage database" to reload the samples, add your own sentences (one per line), or reset everything with the "Reset database" button (it deletes everything and reloads the 31 samples for this visitor).

## Built with

- `transformers` (AutoTokenizer) for `gpt2` and `bert-base-uncased`
- `tiktoken` for `cl100k_base`
- `sentence-transformers` with `all-MiniLM-L6-v2` (384 dimensions)
- `chromadb` for the vector database
- `scikit-learn` (PCA) and `plotly` for the charts
- `streamlit` for the interface

## Limitations

- The sample data is small (about 30 sentences), so the map and search results are illustrative, not representative.
- PCA flattens 384 dimensions to 2D, so distances on the map only roughly reflect real meaning distance.
- Keyword search counts exact word overlaps (ignoring common words), so it misses synonyms and paraphrases by design.
- Similarity scores depend on the embedding model and the sample data. Treat them as a demonstration, not a benchmark.

## Inspiration

Inspired by [Transformer Explainer](https://poloclub.github.io/transformer-explainer/), which visualizes what happens *inside* a model. This project covers the steps before and around it.

## License

Add a license here (for example MIT).
