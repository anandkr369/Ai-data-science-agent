import os
import io
import uuid
import pickle
from typing import TypedDict

import numpy as np
import pandas as pd
import joblib
 
from dotenv import load_dotenv
load_dotenv() 

from flask import Flask, request, jsonify, session, render_template, send_file

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END

# -----------------------------
# APP SETUP
# -----------------------------

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
# Where trained models (.pkl) get written to, per session.
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models_store")
os.makedirs(MODEL_DIR, exist_ok=True)

# In-memory store keyed by session_id. Holds things that can't/shouldn't be
# round-tripped through the Flask session cookie (dataframes, FAISS index,
# the LCEL retrieval chain, etc). This mirrors what st.session_state was
# doing in the Streamlit version. NOTE: this only works correctly with a
# single Gunicorn worker (see README) since it's process-local memory.
SESSIONS = {}

# -----------------------------
# API KEY
# -----------------------------
# NEVER hardcode API keys in source. Set GOOGLE_API_KEY as an environment
# variable (in Render: Dashboard -> your service -> Environment).
google_api_key = os.getenv("GOOGLE_API_KEY")
llm = None
if google_api_key:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0
    )

# Embeddings model is loaded once at startup and reused across requests.
_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


# -----------------------------
# STATE
# -----------------------------

class AgentState(TypedDict):
    df: pd.DataFrame
    eda_summary: str
    ml_summary: str
    insight_summary: str
    model_path: str
    feature_importance: dict
    problem_type: str
    metrics: dict


# -----------------------------
# EDA AGENT
# -----------------------------

def eda_agent(state):
    df = state["df"]
    summary = []

    summary.append(f"Rows: {df.shape[0]}")
    summary.append(f"Columns: {df.shape[1]}")

    missing = df.isnull().sum()
    summary.append(f"Missing Values:\n{missing.to_string()}")

    numeric = df.select_dtypes(include=np.number)
    if not numeric.empty:
        summary.append(numeric.describe().to_string())

    state["eda_summary"] = "\n".join(summary)
    return state


# -----------------------------
# ML AGENT
# -----------------------------

def ml_agent(state):
    df = state["df"].copy()
    state["model_path"] = ""
    state["feature_importance"] = {}
    state["problem_type"] = ""
    state["metrics"] = {}

    if len(df.columns) < 2:
        state["ml_summary"] = "Not enough columns to build a model."
        return state

    # Drop rows where the target is missing; fill remaining NaNs so the
    # model doesn't blow up on real-world messy data.
    target = df.columns[-1]
    df = df.dropna(subset=[target])

    if df.empty:
        state["ml_summary"] = "No rows left after dropping missing target values."
        return state

    encoders = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            # Catches object, pandas "string", category, bool, etc.
            df[col] = df[col].fillna("missing").astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    X = df.drop(columns=[target])
    y = df[target]

    if X.empty or y.nunique() < 2:
        state["ml_summary"] = "Not enough variation in the data to train a model."
        return state

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        unique_values = y.nunique()

        if unique_values <= 20:
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)

            feature_importance = dict(zip(X.columns, model.feature_importances_))
            state["problem_type"] = "classification"
            state["metrics"] = {"accuracy": float(acc)}

            summary = f"""
Classification Problem

Target: {target}

Accuracy: {acc:.4f}

Feature Importance:
{feature_importance}
"""
        else:
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            r2 = r2_score(y_test, pred)
            mae = mean_absolute_error(y_test, pred)

            feature_importance = dict(zip(X.columns, model.feature_importances_))
            state["problem_type"] = "regression"
            state["metrics"] = {"r2": float(r2), "mae": float(mae)}

            summary = f"""
Regression Problem

Target: {target}

R2 Score: {r2:.4f}

MAE: {mae:.4f}

Feature Importance:
{feature_importance}
"""

        # Persist the trained model (+ encoders + feature/target metadata) as
        # a .pkl so it can be reused for inference or downloaded, without
        # retraining every time.
        model_id = uuid.uuid4().hex
        model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
        joblib.dump(
            {
                "model": model,
                "encoders": encoders,
                "feature_columns": list(X.columns),
                "target_column": target,
                "problem_type": state["problem_type"],
            },
            model_path,
        )
        state["model_path"] = model_path
        state["feature_importance"] = {
            k: float(v) for k, v in feature_importance.items()
        }

    except Exception as e:
        summary = f"ML Agent failed: {e}"

    state["ml_summary"] = summary
    return state


# -----------------------------
# INSIGHT AGENT
# -----------------------------

def insight_agent(state):
    if llm is None:
        state["insight_summary"] = (
            "Insight Agent skipped: no GOOGLE_API_KEY configured on the server."
        )
        return state

    prompt = f"""
You are a senior data scientist.

EDA Results:
{state['eda_summary']}

ML Results:
{state['ml_summary']}

Generate:
1. Key findings
2. Business insights
3. Recommendations
"""
    try:
        response = llm.invoke(prompt)
        state["insight_summary"] = response.content
    except Exception as e:
        state["insight_summary"] = f"Insight Agent failed: {e}"

    return state


# -----------------------------
# LANGGRAPH
# -----------------------------

graph = StateGraph(AgentState)
graph.add_node("EDA_Agent", eda_agent)
graph.add_node("ML_Agent", ml_agent)
graph.add_node("Insight_Agent", insight_agent)
graph.set_entry_point("EDA_Agent")
graph.add_edge("EDA_Agent", "ML_Agent")
graph.add_edge("ML_Agent", "Insight_Agent")
graph.add_edge("Insight_Agent", END)

workflow = graph.compile()


# -----------------------------
# HELPERS
# -----------------------------

def get_session_id():
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]


def build_retrieval_chain(eda_summary, ml_summary, insight_summary):
    documents = [
        Document(page_content=eda_summary),
        Document(page_content=ml_summary),
        Document(page_content=insight_summary),
    ]

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()

    chat_prompt = ChatPromptTemplate.from_template("""
You are an expert Data Scientist.

Answer ONLY using the retrieved context.

Context:
{context}

Question:
{input}
""")

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    return (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | chat_prompt
        | llm
        | StrOutputParser()
    )


# -----------------------------
# ROUTES
# -----------------------------

@app.route("/")
def index():
    return render_template("index.html", llm_configured=llm is not None)


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400

    result = workflow.invoke({
        "df": df,
        "eda_summary": "",
        "ml_summary": "",
        "insight_summary": "",
        "model_path": "",
        "feature_importance": {},
        "problem_type": "",
        "metrics": {},
    })

    sid = get_session_id()

    retrieval_chain = None
    if llm is not None:
        try:
            retrieval_chain = build_retrieval_chain(
                result["eda_summary"], result["ml_summary"], result["insight_summary"]
            )
        except Exception as e:
            result["insight_summary"] += f"\n\n(Chat agent unavailable: {e})"

    SESSIONS[sid] = {
        "df": df,
        "result": result,
        "retrieval_chain": retrieval_chain,
    }

    # Correlation matrix for the frontend heatmap.
    numeric = df.select_dtypes(include=np.number)
    corr_payload = None
    if not numeric.empty and numeric.shape[1] > 1:
        corr = numeric.corr().round(4)
        corr_payload = {
            "labels": list(corr.columns),
            "matrix": corr.values.tolist(),
        }

    preview = df.head().replace({np.nan: None}).to_dict(orient="records")
    columns = list(df.columns)

    response = {
        "columns": columns,
        "preview": preview,
        "eda_summary": result["eda_summary"],
        "ml_summary": result["ml_summary"],
        "insight_summary": result["insight_summary"],
        "problem_type": result.get("problem_type", ""),
        "metrics": result.get("metrics", {}),
        "feature_importance": result.get("feature_importance", {}),
        "correlation": corr_payload,
        "model_available": bool(result.get("model_path")),
        "chat_available": retrieval_chain is not None,
    }
    return jsonify(response)


@app.route("/api/chat", methods=["POST"])
def chat():
    sid = get_session_id()
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is empty"}), 400

    sess = SESSIONS.get(sid)
    if not sess or sess.get("retrieval_chain") is None:
        return jsonify({"error": "Upload a CSV first (and configure GOOGLE_API_KEY) before chatting."}), 400

    try:
        answer = sess["retrieval_chain"].invoke(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": f"Chat agent failed: {e}"}), 500


@app.route("/api/download_model", methods=["GET"])
def download_model():
    sid = get_session_id()
    sess = SESSIONS.get(sid)
    if not sess:
        return jsonify({"error": "No trained model for this session"}), 400

    model_path = sess["result"].get("model_path")
    if not model_path or not os.path.exists(model_path):
        return jsonify({"error": "No trained model available"}), 400

    return send_file(model_path, as_attachment=True, download_name="trained_model.pkl")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "llm_configured": llm is not None})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
