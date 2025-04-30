#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag.py
------

• Keeps the original RAG pipeline **unchanged**  
• Adds an optional Streamlit front-end (same file)  
• If the script is launched with **`streamlit run rag.py`** the UI is
  shown; otherwise it falls back to the small CLI demo.

Dependencies
------------
pip install streamlit sentence-transformers transformers faiss-cpu torch
"""

# --------------------------------------------------------------------------- #
# Original RAG pipeline (steps 1-11) – **unchanged**                          #
# --------------------------------------------------------------------------- #
import json, os, re, unicodedata
from typing import List, Dict, Tuple

import torch, faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KB_PATH            = "knowledge_base.json"
EMBED_MODEL_NAME   = "all-MiniLM-L6-v2"
GEN_MODEL_NAME     = "google/flan-t5-small"

INDEX_FILE         = "faiss.index"
EMB_FILE           = "embeddings.npy"

TOP_K              = 3
DISTANCE_THRESHOLD = 1.0
HISTORY_MAX        = 3

# ---------- text cleaner ----------------------------------------------------
PUNCT_RE = re.compile(r"[^\w\s]")

def clean(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode()
    text = text.lower()
    text = PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

# ---------- model loading ---------------------------------------------------
embedder  = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
generator = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(DEVICE)

# ---------- knowledge base --------------------------------------------------
def load_kb(path: str) -> List[Dict]:
    data = json.load(open(path, encoding="utf-8"))
    return list(data.values()) if isinstance(data, dict) else data

restaurants = load_kb(KB_PATH)

# ---------- flatten menu items ---------------------------------------------
texts, meta = [], []
for r in restaurants:
    rname, rloc = r["restaurant_name"], r["location"]
    for it in r["items"]:
        txt_raw = (
            f"{rname} {rloc} {it['category']} "
            f"{it['item_name']} {it['description']} "
            f"{' '.join(it['special_features'] or [])}"
        )
        texts.append(clean(txt_raw))
        meta.append(
            {
                "restaurant": rname,
                "category"  : it["category"],
                "item_name" : it["item_name"],
                "url"       : it["product_url"],
                "price"     : it["price"],
                "features"  : it["special_features"],
            }
        )

# ---------- FAISS index -----------------------------------------------------
if os.path.exists(INDEX_FILE) and os.path.exists(EMB_FILE):
    faiss_index = faiss.read_index(INDEX_FILE)
else:
    emb = embedder.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )
    faiss_index = faiss.IndexFlatL2(emb.shape[1])
    faiss_index.add(emb)
    faiss.write_index(faiss_index, INDEX_FILE)
    emb.tofile(EMB_FILE)

def retrieve(query: str, k: int = TOP_K) -> List[Dict]:
    q_emb = embedder.encode([clean(query)], convert_to_numpy=True)
    dist, idx = faiss_index.search(q_emb, k)
    return [
        {
            "text"    : texts[i],
            "meta"    : meta[i],
            "distance": float(dist[0][rank]),
        }
        for rank, i in enumerate(idx[0])
    ]

# ---------- conversation memory --------------------------------------------
class Conversation:
    def __init__(self, max_turns: int = HISTORY_MAX):
        self.max = max_turns
        self.memory: List[Tuple[str, str]] = []

    def add(self, user: str, assistant: str) -> None:
        self.memory.append((user, assistant))
        if len(self.memory) > self.max:
            self.memory.pop(0)

    def format_history(self) -> str:
        if not self.memory:
            return ""
        return "\n".join(f"User: {u}\nAssistant: {a}" for u, a in self.memory) + "\n"

# ---------- prompt builder --------------------------------------------------
SYSTEM_PROMPT = (
    "You answer questions about restaurant menus using ONLY the CONTEXT. "
    "If the answer cannot be found, say you do not know."
)

def make_prompt(query: str, ctx_chunks: List[str], history: str) -> str:
    ctx = "\n".join(ctx_chunks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{history}"
        f"CONTEXT:\n{ctx}\n\n"
        f"Question: {query}\nAnswer:"
    )

def dedupe_tokens(text: str) -> str:
    toks = text.split()
    out = [toks[0]] if toks else []
    for tok in toks[1:]:
        if tok != out[-1]:
            out.append(tok)
    return " ".join(out)

# ---------- main RAG answer -------------------------------------------------
def answer(query: str, conv: Conversation,
           top_k: int = TOP_K) -> Tuple[str, List[Dict]]:
    retrieved = retrieve(query, top_k)

    if (not retrieved) or (retrieved[0]["distance"] > DISTANCE_THRESHOLD):
        response = (
            "I do not know. The knowledge base does not contain "
            "information relevant to this question."
        )
        conv.add(query, response)
        return response, []

    prompt = make_prompt(query,
                         [r["text"] for r in retrieved],
                         conv.format_history())

    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=512).to(DEVICE)

    out_ids = generator.generate(
        **inputs,
        max_length=220,
        num_beams=4,
        temperature=0.7,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15
    )

    response = dedupe_tokens(tokenizer.decode(out_ids[0], skip_special_tokens=True))
    conv.add(query, response)
    return response, retrieved

# --------------------------------------------------------------------------- #
# Optional Streamlit front-end                                                #
# --------------------------------------------------------------------------- #
def run_streamlit() -> None:
    import streamlit as st

    if "conv" not in st.session_state:
        st.session_state.conv = Conversation()

    st.set_page_config(page_title="Restaurant Menu Chatbot", page_icon="🍽️")
    st.title("Restaurant Menu Chatbot")

    st.markdown(
        """
        **Ask me about:**
        * Menu item details  
        * Restaurant comparisons  
        * Price ranges (if available)  
        * Dietary restrictions
        """
    )

    user_query = st.text_input(
        "Your question",
        placeholder="Suggest a spicy vegetarian snack",
        key="user_query"
    )

    if user_query:
        with st.spinner("Retrieving answer…"):
            reply, ctx_list = answer(user_query, st.session_state.conv)

        st.subheader("Answer")
        st.write(reply)

        st.subheader(f"Top-{TOP_K} retrieved context")
        for ctx in ctx_list:
            st.markdown(
                f"- **{ctx['meta']['restaurant']}**: {ctx['text']} "
                f"(distance {ctx['distance']:.3f})"
            )

    if st.sidebar.button("Reset conversation"):
        st.session_state.pop("conv", None)
        st.experimental_rerun()

# --------------------------------------------------------------------------- #
# CLI demo or Streamlit detection                                             #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Detect whether running inside Streamlit (`streamlit run rag.py`)
    import sys
    if any("streamlit" in arg for arg in sys.argv) or "streamlit" in sys.modules:
        run_streamlit()
    else:
        convo = Conversation()
        demo_question = "Is Big Mac® vegetarian food?"
        ans, ctx = answer(demo_question, convo)
        print("Query:", demo_question)
        print("\nAnswer:", ans)
        print("\nRetrieved context:")
        for c in ctx:
            print(f"- {c['text']}  (distance={c['distance']:.3f})")
