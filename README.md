# Agentic Data Science Assistant (Flask edition)

Same EDA → ML → Insight → Chat LangGraph pipeline as the original Streamlit
app, now served from a Flask backend with a plain HTML/CSS/JS frontend so it
can be deployed on Render.

## What changed vs. the Streamlit version
- **Nothing about the agent logic** (EDA_Agent, ML_Agent, Insight_Agent,
  the LangGraph wiring, the FAISS retrieval chain) was altered.
- Streamlit UI → Flask routes (`/api/upload`, `/api/chat`,
  `/api/download_model`) + `templates/index.html` + `static/js/app.js`.
- `st.session_state` → an in-memory `SESSIONS` dict keyed by a Flask
  session cookie (works with a single Gunicorn worker — see below).
- The trained `RandomForestClassifier`/`RandomForestRegressor` is now saved
  to disk with `joblib` as a `.pkl` (in `models_store/`) after every upload,
  and can be downloaded via the "Download trained model" button.
- **The hardcoded Google API key was removed.** Set `GOOGLE_API_KEY` as an
  environment variable instead — it was previously committed in plaintext
  in the source, so if that key is real, rotate it in Google AI Studio.

## Run locally

```bash
cd app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in GOOGLE_API_KEY
export $(cat .env | xargs)
python app.py
```

Visit http://localhost:5000

## Deploy on Render

1. Push this folder to a GitHub repo.
2. In Render: **New → Web Service**, connect the repo (or use the included
   `render.yaml` as a Blueprint for one-click setup).
3. Environment: **Python 3**
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT`
   (both already set in `Procfile` / `render.yaml`)
4. Add an environment variable `GOOGLE_API_KEY` with your key in the Render
   dashboard (Environment tab) — never commit it to the repo.
5. Deploy. First build will take a while since `sentence-transformers` /
   `faiss-cpu` pull in a fair amount (including a CPU torch build).

### Notes / things worth knowing
- **Single worker**: chat/session state lives in server memory
  (`SESSIONS` dict), so the app is configured for 1 Gunicorn worker. If you
  scale to multiple workers/instances, move that state to Redis or similar.
- **Free tier disk**: Render's free plan disk isn't guaranteed to persist
  across deploys/restarts — `models_store/*.pkl` is fine for
  "train now, download now" but don't rely on it as long-term storage.
- **Cold starts**: `HuggingFaceEmbeddings` downloads
  `sentence-transformers/all-MiniLM-L6-v2` on first use; the first upload
  after a deploy/restart will be slower while that model loads.
