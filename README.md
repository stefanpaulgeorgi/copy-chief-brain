# 🧠 Copy Chief Brain

A RAG-powered copy review tool that provides instant feedback in Stefan's voice.

## What It Does

1. **Ingests** your copy review transcripts
2. **Indexes** them in a vector database (ChromaDB)
3. **Retrieves** relevant past feedback when you submit new copy
4. **Generates** feedback in Stefan's voice using Claude

## Quick Start

### 1. Install Dependencies

```bash
cd copy_chief_brain
pip install -r requirements.txt
```

### 2. Set Up API Keys

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API keys:
# - ANTHROPIC_API_KEY (for Claude)
# - OPENAI_API_KEY (for embeddings)
```

### 3. Ingest Transcripts

```bash
# From .srt files
python -m src.ingest -i "../CA Pro Copy Reviews and Feedback" --clear

# OR from the pre-processed JSONL file
python -m src.ingest -i "../processed_transcripts/all_transcripts.jsonl" --clear
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Usage

### Web Interface

1. Open the Streamlit app
2. Paste your copy in the left panel
3. (Optional) Select copy type and niche
4. Click "Get Feedback"
5. Stefan's feedback appears in the right panel

### Options

- **Use RAG context**: Toggle to include/exclude past review examples
- **Context chunks**: How many examples to retrieve (more = richer context, slower)
- **Copy type**: Help the system understand what you're submitting
- **Niche**: Help retrieve more relevant examples

---

## Adding More Transcripts

When you have more review transcripts to add:

```bash
# Add to existing index (don't use --clear)
python -m src.ingest -i "/path/to/new/transcripts"

# Or replace everything
python -m src.ingest -i "/path/to/all/transcripts" --clear
```

The system is designed to scale to 1,500+ transcripts.

---

## Project Structure

```
copy_chief_brain/
├── app.py                 # Streamlit web interface
├── requirements.txt       # Python dependencies
├── .env.example          # API keys template
├── .env                  # Your API keys (create this)
├── data/
│   └── chroma_db/        # Vector database storage
└── src/
    ├── __init__.py
    ├── config.py         # Settings and system prompt
    ├── ingest.py         # Transcript processing
    ├── retriever.py      # RAG retrieval logic
    └── generator.py      # Claude response generation
```

---

## Customization

### Modify the System Prompt

Edit `src/config.py` → `STEFAN_SYSTEM_PROMPT` to adjust:
- Voice and tone
- Feedback structure
- Methodology emphasis

### Adjust Retrieval

In `src/config.py`:
- `CHUNK_SIZE`: How big each indexed chunk is (default: 1000 words)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 150 words)
- `TOP_K_RESULTS`: Default number of chunks to retrieve (default: 8)

### Change the Model

In `src/config.py`:
- `LLM_MODEL`: Change Claude model (e.g., `claude-3-opus-20240229`)
- `EMBEDDING_MODEL`: Change embedding model

---

## Troubleshooting

### "No transcripts indexed yet"

Run the ingestion script:
```bash
python -m src.ingest -i "../CA Pro Copy Reviews and Feedback" --clear
```

### "ANTHROPIC_API_KEY not set"

Make sure your `.env` file exists and contains:
```
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx
```

### "OPENAI_API_KEY not set"

Add to your `.env` file:
```
OPENAI_API_KEY=sk-xxxxx
```

### Slow responses

- Reduce "Context chunks" in sidebar
- Use a faster Claude model in `src/config.py`

---

## Sharing with Clients

To let clients use this:

1. **Self-hosted**: Deploy on a server with your API keys
2. **Local setup**: Give them this folder + instructions (they need their own API keys)
3. **Streamlit Cloud**: Deploy to Streamlit's free hosting

For client deployment, consider adding:
- Rate limiting
- Usage tracking
- Authentication

---

## Cost Estimates

Per review (approximate):
- **Embeddings**: ~$0.0001 (query embedding)
- **Claude API**: ~$0.01-0.05 (depending on response length)
- **Total**: ~$0.01-0.05 per review

Ingestion (one-time):
- **Embeddings**: ~$0.02 per 1000 chunks
- **52 transcripts**: ~$0.10-0.20 total

---

## Future Improvements

- [ ] Add user authentication
- [ ] Track feedback quality/usefulness
- [ ] Export feedback to PDF/doc
- [ ] Compare multiple copy versions
- [ ] Add feedback categories/tags
- [ ] Fine-tune a model on the data

---

*Built for Stefan's Copy Chiefing workflow*
