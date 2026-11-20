"""A tiny RAG system, written to be read and understood.

Pipeline:   question -> retriever -> (optional defense) -> LLM -> answer

Pieces in this file
  1. helpers          : load data, read numbers (also Bangla digits)
  2. retrievers       : TF-IDF (default, no download) or multilingual-e5 (neural, optional)
  3. LLM backends     : "none" (no LLM), "hf" (Hugging Face model), "ollama" (local server)
  4. defense          : a simple "authority + agreement" filter
  5. answer()         : glues everything together

Everything is SYNTHETIC demo data. Nothing here is official advice.
"""
import json
import os
import re
import urllib.request
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


# --------------------------------------------------------------------------- 1. helpers
BN_TO_EN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def load_jsonl(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def normalize_digits(text):
    """Turn Bangla digits into 0-9 and drop thousands commas ('45,000' -> '45000')."""
    text = text.translate(BN_TO_EN_DIGITS)
    return re.sub(r"(?<=\d),(?=\d{3})", "", text)


def extract_numbers(text):
    return re.findall(r"\d+", normalize_digits(text))


# --------------------------------------------------------------------------- 2. retrievers
class TfidfRetriever:
    """Character n-gram TF-IDF. Works offline and copes with Bangla / Banglish spelling noise."""
    name = "tfidf"

    def __init__(self, docs):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.docs = docs
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                   lowercase=True, sublinear_tf=True)
        self.matrix = self.vec.fit_transform([normalize_digits(d["text"]) for d in docs])

    def search(self, query, k=3):
        q = self.vec.transform([normalize_digits(query)])
        scores = (self.matrix @ q.T).toarray().ravel()
        top = np.argsort(-scores)[:k]
        return [(self.docs[i], float(scores[i])) for i in top]


class E5Retriever:
    """Neural multilingual embeddings (needs: pip install sentence-transformers; downloads ~470 MB)."""
    name = "e5"

    def __init__(self, docs, model_name="intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer
        self.docs = docs
        self.model = SentenceTransformer(model_name)
        self.emb = self.model.encode(["passage: " + d["text"] for d in docs],
                                     normalize_embeddings=True)

    def search(self, query, k=3):
        q = self.model.encode(["query: " + query], normalize_embeddings=True)[0]
        scores = self.emb @ q
        top = np.argsort(-scores)[:k]
        return [(self.docs[i], float(scores[i])) for i in top]


def make_retriever(kind, docs):
    return {"tfidf": TfidfRetriever, "e5": E5Retriever}[kind](docs)


# --------------------------------------------------------------------------- 3. LLM backends
SYSTEM_PROMPT = (
    "You answer questions about citizen services using ONLY the context below. "
    "If the context does not contain the answer, say you do not know. "
    "Reply with one short sentence."
)


def build_messages(question, chunks):
    context = "\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks))
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


class NoLLM:
    """No language model: just repeat the best retrieved chunk.
    Lets you study the retrieval stage on its own, and runs on any laptop."""
    name = "none"

    def generate(self, question, chunks):
        return chunks[0]["text"] if chunks else "I do not know."


class HFLLM:
    """A small chat model from Hugging Face, run on your own computer.
    Needs: pip install transformers torch   (first run downloads the model)."""

    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens=60):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = f"hf:{model_name}"
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

    def generate(self, question, chunks):
        prompt = self.tok.apply_chat_template(build_messages(question, chunks),
                                              tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt")
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()


class OllamaLLM:
    """Talks to a local Ollama server (https://ollama.com). Start it, then: ollama pull qwen2.5:1.5b"""

    def __init__(self, model_name="qwen2.5:1.5b", url="http://localhost:11434/api/chat"):
        self.name = f"ollama:{model_name}"
        self.model_name, self.url = model_name, url

    def generate(self, question, chunks):
        body = json.dumps({"model": self.model_name, "stream": False,
                           "messages": build_messages(question, chunks),
                           "options": {"temperature": 0}}).encode("utf-8")
        req = urllib.request.Request(self.url, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())["message"]["content"].strip()


def make_llm(kind, model=None):
    if kind == "none":
        return NoLLM()
    if kind == "hf":
        return HFLLM(model) if model else HFLLM()
    if kind == "ollama":
        return OllamaLLM(model) if model else OllamaLLM()
    raise ValueError(f"unknown llm backend: {kind}")


# --------------------------------------------------------------------------- 4. defense
AUTHORITY_WEIGHT = {"official": 3.0, "secondary": 1.0, "unverified": 1.0}


def consensus_filter(hits, same_topic_ratio=0.7):
    """Simple defense: 'authority + agreement'.

    1. Look only at retrieved chunks that are about the same topic as the best hit
       (score >= same_topic_ratio * best score).
    2. Every number in a chunk gets a vote worth the chunk's authority weight.
    3. Keep chunks that contain a winning number, drop the rest.

    Known weaknesses (see README): it cannot help if only the poisoned chunk is retrieved,
    and it fails if the attacker can fake the 'official' label.
    """
    if not hits:
        return hits, []
    best = hits[0][1]
    group = [h for h in hits if best > 0 and h[1] >= same_topic_ratio * best]
    rest = [h for h in hits if h not in group]
    votes = defaultdict(float)
    for doc, _ in group:
        for n in set(extract_numbers(doc["text"])):
            votes[n] += AUTHORITY_WEIGHT.get(doc.get("authority", "unverified"), 1.0)
    if not votes:
        return hits, []
    top_vote = max(votes.values())
    winners = {n for n, v in votes.items() if v == top_vote}
    kept = [h for h in group if set(extract_numbers(h[0]["text"])) & winners]
    dropped = [h for h in group if h not in kept]
    return kept + rest, dropped


# --------------------------------------------------------------------------- 5. answer
def answer(question, retriever, llm, k=3, defense=False):
    hits = retriever.search(question, k)
    dropped = []
    if defense:
        hits, dropped = consensus_filter(hits)
    text = llm.generate(question, [d for d, _ in hits])
    return {"answer": text, "retrieved": hits, "dropped": dropped}
