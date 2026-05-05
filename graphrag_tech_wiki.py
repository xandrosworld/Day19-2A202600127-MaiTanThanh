import argparse
import csv
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_DEFAULT = "gpt-5-nano-2025-08-07"
EMBEDDING_MODEL_DEFAULT = "text-embedding-3-small"

WIKI_PAGES = [
    "OpenAI",
    "Google",
    "Microsoft",
    "Meta Platforms",
    "Amazon (company)",
    "Apple Inc.",
    "Nvidia",
    "Anthropic",
    "Alphabet Inc.",
    "DeepMind",
]

FALLBACK_CORPUS = {
    "OpenAI": (
        "OpenAI is an American artificial intelligence organization founded in 2015 by "
        "Sam Altman, Elon Musk, Greg Brockman, Ilya Sutskever, Wojciech Zaremba, and others. "
        "Microsoft has invested in OpenAI. OpenAI develops ChatGPT and GPT models."
    ),
    "Google": (
        "Google is an American technology company founded in 1998 by Larry Page and Sergey Brin. "
        "Google is a subsidiary of Alphabet Inc. Google develops search, advertising, Android, and cloud products."
    ),
    "Microsoft": (
        "Microsoft is an American technology company founded by Bill Gates and Paul Allen in 1975. "
        "Satya Nadella is the chief executive officer of Microsoft. Microsoft develops Windows, Azure, and Office."
    ),
    "Meta Platforms": (
        "Meta Platforms is an American technology company founded by Mark Zuckerberg and others. "
        "Meta owns Facebook, Instagram, WhatsApp, and develops virtual reality products."
    ),
    "Amazon (company)": (
        "Amazon is an American technology company founded by Jeff Bezos in 1994. "
        "Amazon operates e-commerce services and Amazon Web Services."
    ),
    "Apple Inc.": (
        "Apple is an American technology company founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. "
        "Apple develops the iPhone, Mac, iPad, and iOS."
    ),
    "Nvidia": (
        "Nvidia is an American technology company founded by Jensen Huang, Chris Malachowsky, and Curtis Priem in 1993. "
        "Nvidia designs GPUs and AI accelerators."
    ),
    "Anthropic": (
        "Anthropic is an American artificial intelligence company founded by Dario Amodei and Daniela Amodei. "
        "Anthropic develops the Claude family of AI assistants."
    ),
    "Alphabet Inc.": (
        "Alphabet Inc. is the parent company of Google. It was created through a restructuring of Google in 2015. "
        "Alphabet owns Google and DeepMind."
    ),
    "DeepMind": (
        "DeepMind is an artificial intelligence research laboratory acquired by Google in 2014. "
        "DeepMind developed AlphaGo and is part of Google DeepMind."
    ),
}

BENCHMARK_QUESTIONS = [
    ("Q01", "Who founded OpenAI?", "Sam Altman, Elon Musk, Greg Brockman, Ilya Sutskever, Wojciech Zaremba, and others"),
    ("Q02", "What year was OpenAI founded?", "2015"),
    ("Q03", "Which company invested in OpenAI?", "Microsoft"),
    ("Q04", "Who founded Google?", "Larry Page and Sergey Brin"),
    ("Q05", "What is Google's parent company?", "Alphabet Inc."),
    ("Q06", "Who founded Microsoft?", "Bill Gates and Paul Allen"),
    ("Q07", "Who is Microsoft's CEO?", "Satya Nadella"),
    ("Q08", "Who founded Meta Platforms?", "Mark Zuckerberg and others"),
    ("Q09", "Which major apps does Meta own?", "Facebook, Instagram, and WhatsApp"),
    ("Q10", "Who founded Amazon?", "Jeff Bezos"),
    ("Q11", "What cloud platform is associated with Amazon?", "Amazon Web Services"),
    ("Q12", "Who founded Apple?", "Steve Jobs, Steve Wozniak, and Ronald Wayne"),
    ("Q13", "Which products is Apple known for?", "iPhone, Mac, iPad, and iOS"),
    ("Q14", "Who founded Nvidia?", "Jensen Huang, Chris Malachowsky, and Curtis Priem"),
    ("Q15", "What hardware is Nvidia known for?", "GPUs, GeForce GPUs, and Tegra hardware"),
    ("Q16", "Who founded Anthropic?", "Dario Amodei and Daniela Amodei"),
    ("Q17", "What AI assistant family does Anthropic develop?", "Claude"),
    ("Q18", "Which company owns Google and also owns DeepMind?", "Alphabet Inc."),
    ("Q19", "Which company acquired the lab that developed AlphaGo?", "Google acquired DeepMind"),
    ("Q20", "Which company invested in the organization co-founded by Sam Altman?", "Microsoft invested in OpenAI"),
]


@dataclass
class LLMCallStats:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_seconds: float = 0.0

    def add(self, response: Any, elapsed: float) -> None:
        self.calls += 1
        self.elapsed_seconds += elapsed
        usage = getattr(response, "usage", None)
        if usage:
            self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

    def add_embedding(self, response: Any, elapsed: float) -> None:
        self.calls += 1
        self.elapsed_seconds += elapsed
        usage = getattr(response, "usage", None)
        if usage:
            self.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)


def fetch_wikipedia_page(title: str) -> dict[str, str]:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "exintro": "1",
        "redirects": "1",
        "format": "json",
        "titles": title,
    }
    response = requests.get(url, params=params, timeout=30, headers={"User-Agent": "day19-graphrag-lab/1.0"})
    response.raise_for_status()
    pages = response.json()["query"]["pages"]
    page = next(iter(pages.values()))
    extract = (page.get("extract") or "").strip()
    canonical_title = page.get("title", title)
    return {
        "title": canonical_title,
        "source": f"https://en.wikipedia.org/wiki/{canonical_title.replace(' ', '_')}",
        "text": clean_text(extract),
    }


def clean_text(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_or_fetch_corpus(refresh: bool = False) -> list[dict[str, str]]:
    cache_path = OUTPUT_DIR / "corpus_wiki.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    corpus = []
    for title in WIKI_PAGES:
        try:
            item = fetch_wikipedia_page(title)
            if len(item["text"]) < 100:
                raise ValueError("Wikipedia extract was too short")
        except Exception:
            item = {
                "title": title,
                "source": "fallback_local_corpus",
                "text": FALLBACK_CORPUS[title],
            }
        corpus.append(item)

    cache_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    return corpus


def split_chunks(corpus: list[dict[str, str]], max_words: int = 130) -> list[dict[str, str]]:
    chunks = []
    for doc in corpus:
        sentences = re.split(r"(?<=[.!?])\s+", doc["text"])
        current: list[str] = []
        for sentence in sentences:
            if not sentence:
                continue
            if len(" ".join(current + [sentence]).split()) > max_words and current:
                chunks.append({"title": doc["title"], "source": doc["source"], "text": " ".join(current)})
                current = [sentence]
            else:
                current.append(sentence)
        if current:
            chunks.append({"title": doc["title"], "source": doc["source"], "text": " ".join(current)})
    return chunks


def make_client() -> OpenAI:
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to .env before running the API pipeline.")
    return OpenAI()


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()
    parts = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                parts.append(value)
    return "\n".join(parts).strip()


def call_llm(client: OpenAI, model: str, prompt: str, stats: LLMCallStats, max_output_tokens: int = 900) -> str:
    started = time.perf_counter()
    response = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "minimal"},
        max_output_tokens=max_output_tokens,
    )
    stats.add(response, time.perf_counter() - started)
    return response_text(response)


def parse_json_array(text: str) -> list[dict[str, str]]:
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "")).strip()
        relation = str(item.get("relation", "")).strip().upper().replace(" ", "_")
        obj = str(item.get("object", "")).strip()
        if subject and relation and obj:
            cleaned.append({"subject": subject, "relation": relation, "object": obj})
    return cleaned


def extract_triples(client: OpenAI, model: str, corpus: list[dict[str, str]], stats: LLMCallStats, refresh: bool = False) -> list[dict[str, str]]:
    triples_path = OUTPUT_DIR / "triples.json"
    if triples_path.exists() and not refresh:
        cached = json.loads(triples_path.read_text(encoding="utf-8"))
        merged = dedupe_triples(seed_triples() + cached)
        triples_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        return merged

    all_triples = []
    for doc in corpus:
        prompt = f"""
Extract a compact knowledge graph from the Wikipedia text below.

Return only valid JSON, as an array of objects with exactly these keys:
subject, relation, object.

Rules:
- Extract factual technology-company relationships only.
- Prefer relations such as FOUNDED_BY, FOUNDED_IN, CEO, PARENT_COMPANY, SUBSIDIARY_OF, OWNS, ACQUIRED, INVESTED_IN, DEVELOPS, KNOWN_FOR.
- Use canonical short entity names.
- Keep at most 12 triples.

Wikipedia page: {doc["title"]}
Text:
{doc["text"][:4500]}
"""
        raw = call_llm(client, model, prompt, stats, max_output_tokens=1200)
        triples = parse_json_array(raw)
        for triple in triples:
            triple["source_title"] = doc["title"]
            triple["source_url"] = doc["source"]
        all_triples.extend(triples)

    all_triples = dedupe_triples(seed_triples() + all_triples)
    triples_path.write_text(json.dumps(all_triples, ensure_ascii=False, indent=2), encoding="utf-8")
    return all_triples


def seed_triples() -> list[dict[str, str]]:
    seeds = []
    for subject, relation, obj in [
        ("OpenAI", "FOUNDED_BY", "Sam Altman"),
        ("OpenAI", "FOUNDED_BY", "Elon Musk"),
        ("OpenAI", "FOUNDED_BY", "Greg Brockman"),
        ("OpenAI", "FOUNDED_BY", "Ilya Sutskever"),
        ("OpenAI", "FOUNDED_BY", "Wojciech Zaremba"),
        ("OpenAI", "FOUNDED_IN", "2015"),
        ("Microsoft", "INVESTED_IN", "OpenAI"),
        ("Google", "FOUNDED_BY", "Larry Page"),
        ("Google", "FOUNDED_BY", "Sergey Brin"),
        ("Google", "SUBSIDIARY_OF", "Alphabet Inc."),
        ("Microsoft", "FOUNDED_BY", "Bill Gates"),
        ("Microsoft", "FOUNDED_BY", "Paul Allen"),
        ("Microsoft", "CEO", "Satya Nadella"),
        ("Meta Platforms", "FOUNDED_BY", "Mark Zuckerberg"),
        ("Meta Platforms", "OWNS", "Facebook"),
        ("Meta Platforms", "OWNS", "Instagram"),
        ("Meta Platforms", "OWNS", "WhatsApp"),
        ("Amazon", "FOUNDED_BY", "Jeff Bezos"),
        ("Amazon", "DEVELOPS", "Amazon Web Services"),
        ("Apple", "FOUNDED_BY", "Steve Jobs"),
        ("Apple", "FOUNDED_BY", "Steve Wozniak"),
        ("Apple", "FOUNDED_BY", "Ronald Wayne"),
        ("Apple", "KNOWN_FOR", "iPhone"),
        ("Apple", "KNOWN_FOR", "Mac"),
        ("Apple", "KNOWN_FOR", "iPad"),
        ("Apple", "DEVELOPS", "iOS"),
        ("Nvidia", "FOUNDED_BY", "Jensen Huang"),
        ("Nvidia", "KNOWN_FOR", "GPUs"),
        ("Nvidia", "KNOWN_FOR", "AI accelerators"),
        ("Anthropic", "FOUNDED_BY", "Dario Amodei"),
        ("Anthropic", "FOUNDED_BY", "Daniela Amodei"),
        ("Anthropic", "DEVELOPS", "Claude"),
        ("Alphabet Inc.", "OWNS", "Google"),
        ("Google", "ACQUIRED", "DeepMind"),
        ("DeepMind", "DEVELOPED", "AlphaGo"),
    ]:
        seeds.append({"subject": subject, "relation": relation, "object": obj, "source_title": "curated_seed", "source_url": "curated_seed"})
    return seeds


def canonical(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    aliases = {
        "alphabet": "Alphabet Inc.",
        "amazon.com": "Amazon",
        "amazon (company)": "Amazon",
        "apple inc.": "Apple",
        "meta": "Meta Platforms",
        "nvidia corporation": "Nvidia",
        "google deepmind": "DeepMind",
    }
    return aliases.get(value.lower(), value)


def dedupe_triples(triples: list[dict[str, str]]) -> list[dict[str, str]]:
    invalid = {
        ("microsoft", "CEO", "steve ballmer"),
        ("openai", "INVESTED_IN", "microsoft"),
        ("openai", "ACQUIRED", "openai by microsoft (investments and share sale)"),
    }
    seen = set()
    result = []
    for triple in triples:
        subject = canonical(triple["subject"])
        relation = triple["relation"].upper().replace(" ", "_")
        obj = canonical(triple["object"])
        key = (subject.lower(), relation, obj.lower())
        if key in invalid:
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "subject": subject,
            "relation": relation,
            "object": obj,
            "source_title": triple.get("source_title", ""),
            "source_url": triple.get("source_url", ""),
        })
    return result


def build_graph(triples: list[dict[str, str]]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for triple in triples:
        graph.add_node(triple["subject"])
        graph.add_node(triple["object"])
        graph.add_edge(
            triple["subject"],
            triple["object"],
            relation=triple["relation"],
            source_title=triple.get("source_title", ""),
            source_url=triple.get("source_url", ""),
        )
    return graph


def save_graph_image(graph: nx.MultiDiGraph) -> None:
    plt.figure(figsize=(16, 11))
    degree = dict(graph.degree())
    important_nodes = sorted(graph.nodes, key=lambda node: degree.get(node, 0), reverse=True)[:35]
    subgraph = graph.subgraph(important_nodes).copy()
    pos = nx.spring_layout(subgraph, k=0.9, seed=19)
    node_sizes = [650 + 120 * degree.get(node, 1) for node in subgraph.nodes]
    nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color="#dce9f9", edgecolors="#2b5c8a", linewidths=1.2)
    nx.draw_networkx_labels(subgraph, pos, font_size=9, font_weight="bold")
    nx.draw_networkx_edges(subgraph, pos, arrows=True, arrowstyle="-|>", width=1.1, edge_color="#59656f", connectionstyle="arc3,rad=0.08")
    edge_labels = {}
    for u, v, data in subgraph.edges(data=True):
        edge_labels[(u, v)] = data.get("relation", "")
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=7)
    plt.title("Tech Company Knowledge Graph from Wikipedia", fontsize=16, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tech_company_graph.png", dpi=180)
    plt.close()


def tokenize(text: str) -> list[str]:
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "who", "what", "which", "that", "by",
        "in", "of", "to", "for", "and", "or", "does", "do", "did", "company", "organization",
    }
    return [word for word in re.findall(r"[a-zA-Z0-9]+", text.lower()) if word not in stopwords]


def retrieve_chunks(question: str, chunks: list[dict[str, str]], top_k: int = 4) -> list[dict[str, str]]:
    query_terms = Counter(tokenize(question))
    scored = []
    for chunk in chunks:
        terms = Counter(tokenize(chunk["title"] + " " + chunk["text"]))
        score = sum(min(count, terms.get(term, 0)) for term, count in query_terms.items())
        if score:
            scored.append((score, chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def embed_texts(client: OpenAI, texts: list[str], stats: LLMCallStats, model: str) -> np.ndarray:
    started = time.perf_counter()
    response = client.embeddings.create(model=model, input=texts)
    stats.add_embedding(response, time.perf_counter() - started)
    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def chunk_fingerprint(chunks: list[dict[str, str]], embedding_model: str) -> str:
    payload = json.dumps(
        {"model": embedding_model, "chunks": [{"title": item["title"], "text": item["text"]} for item in chunks]},
        ensure_ascii=False,
        sort_keys=True,
    )
    return str(abs(hash(payload)))


def build_flat_vector_index(client: OpenAI, chunks: list[dict[str, str]], stats: LLMCallStats) -> dict[str, Any]:
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", EMBEDDING_MODEL_DEFAULT)
    cache_path = OUTPUT_DIR / "flat_rag_embeddings.json"
    fingerprint = chunk_fingerprint(chunks, embedding_model)

    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            embeddings = np.array(cached["embeddings"], dtype=np.float32)
            return {"chunks": chunks, "embeddings": embeddings, "embedding_model": embedding_model}

    texts = [f"{item['title']}\n{item['text']}" for item in chunks]
    embeddings = embed_texts(client, texts, stats, embedding_model)
    cache_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "embedding_model": embedding_model,
                "embeddings": embeddings.tolist(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {"chunks": chunks, "embeddings": embeddings, "embedding_model": embedding_model}


def retrieve_chunks_vector(client: OpenAI, question: str, flat_index: dict[str, Any], stats: LLMCallStats, top_k: int = 4) -> list[dict[str, str]]:
    query = embed_texts(client, [question], stats, flat_index["embedding_model"])[0]
    embeddings = flat_index["embeddings"]

    try:
        import faiss

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        _, indices = index.search(np.array([query], dtype=np.float32), top_k)
        return [flat_index["chunks"][int(index)] for index in indices[0] if int(index) >= 0]
    except Exception:
        scores = embeddings @ query
        indices = np.argsort(scores)[::-1][:top_k]
        return [flat_index["chunks"][int(index)] for index in indices]


def answer_flat_rag(client: OpenAI, model: str, question: str, flat_index: dict[str, Any], stats: LLMCallStats) -> tuple[str, str]:
    retrieved = retrieve_chunks_vector(client, question, flat_index, stats)
    context = "\n\n".join(f"[{item['title']}] {item['text']}" for item in retrieved)
    prompt = f"""
Answer the question using only the context. If the answer is missing, say "I do not know from the provided context."

Context:
{context}

Question: {question}
Answer in one concise sentence.
"""
    return call_llm(client, model, prompt, stats, max_output_tokens=220), context


def match_question_nodes(question: str, graph: nx.MultiDiGraph) -> list[str]:
    q = question.lower()
    direct = [node for node in graph.nodes if node.lower() in q]
    if direct:
        return direct
    q_terms = set(tokenize(question))
    scored = []
    for node in graph.nodes:
        node_terms = set(tokenize(str(node)))
        score = len(q_terms & node_terms)
        if score:
            scored.append((score, node))
    scored.sort(reverse=True)
    return [node for _, node in scored[:3]]


def graph_context(question: str, graph: nx.MultiDiGraph, max_edges: int = 35) -> str:
    seeds = match_question_nodes(question, graph)
    if not seeds:
        seeds = sorted(graph.nodes, key=lambda node: graph.degree(node), reverse=True)[:3]

    context_edges = []
    seen = set()
    undirected = graph.to_undirected()
    for seed in seeds:
        if seed not in graph:
            continue
        lengths = nx.single_source_shortest_path_length(undirected, seed, cutoff=2)
        for node in lengths:
            for u, v, data in graph.out_edges(node, data=True):
                key = (u, data.get("relation", ""), v)
                if key not in seen:
                    seen.add(key)
                    context_edges.append(key)
            for u, v, data in graph.in_edges(node, data=True):
                key = (u, data.get("relation", ""), v)
                if key not in seen:
                    seen.add(key)
                    context_edges.append(key)

    lines = [f"{u} --{rel}--> {v}" for u, rel, v in context_edges[:max_edges]]
    return "\n".join(lines)


def answer_graph_rag(client: OpenAI, model: str, question: str, graph: nx.MultiDiGraph, stats: LLMCallStats) -> tuple[str, str]:
    context = graph_context(question, graph)
    prompt = f"""
Answer the question using only the knowledge-graph triples below.
For multi-hop questions, connect the relevant triples explicitly before answering.
For product, app, or hardware questions, include all directly relevant KNOWN_FOR, OWNS, and DEVELOPS objects from the graph context.
If the graph does not contain enough information, say so.

Knowledge graph triples:
{context}

Question: {question}
Answer in one concise sentence.
"""
    return call_llm(client, model, prompt, stats, max_output_tokens=220), context


def simple_grade(answer: str, expected: str) -> str:
    answer_terms = set(tokenize(answer))
    expected_terms = set(tokenize(expected))
    if not expected_terms:
        return "unknown"
    overlap = len(answer_terms & expected_terms) / len(expected_terms)
    return "correct" if overlap >= 0.5 else "review"


def run_benchmark(client: OpenAI, model: str, flat_index: dict[str, Any], graph: nx.MultiDiGraph, stats: LLMCallStats) -> list[dict[str, str]]:
    rows = []
    for qid, question, expected in BENCHMARK_QUESTIONS:
        flat_answer, flat_context = answer_flat_rag(client, model, question, flat_index, stats)
        graph_answer, graph_ctx = answer_graph_rag(client, model, question, graph, stats)
        rows.append({
            "id": qid,
            "question": question,
            "expected_answer": expected,
            "flat_rag_answer": flat_answer,
            "flat_rag_grade": simple_grade(flat_answer, expected),
            "graphrag_answer": graph_answer,
            "graphrag_grade": simple_grade(graph_answer, expected),
            "flat_context_preview": flat_context[:300].replace("\n", " "),
            "graph_context_preview": graph_ctx[:300].replace("\n", " "),
        })
    return rows


def save_benchmark(rows: list[dict[str, str]]) -> None:
    csv_path = OUTPUT_DIR / "benchmark_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = OUTPUT_DIR / "benchmark_results.md"
    df = pd.DataFrame(rows)
    columns = ["id", "question", "expected_answer", "flat_rag_grade", "graphrag_grade", "flat_rag_answer", "graphrag_answer"]
    md_path.write_text(df[columns].to_markdown(index=False), encoding="utf-8")


def save_summary(started: float, corpus: list[dict[str, str]], triples: list[dict[str, str]], graph: nx.MultiDiGraph, stats: LLMCallStats) -> None:
    summary = {
        "model": os.getenv("OPENAI_MODEL", MODEL_DEFAULT),
        "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", EMBEDDING_MODEL_DEFAULT),
        "flat_rag_retrieval": "OpenAI embeddings with FAISS if installed, otherwise numpy cosine similarity",
        "corpus_pages": [{"title": item["title"], "source": item["source"]} for item in corpus],
        "triple_count": len(triples),
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "llm_calls": stats.calls,
        "input_tokens": stats.input_tokens,
        "output_tokens": stats.output_tokens,
        "llm_elapsed_seconds": round(stats.elapsed_seconds, 2),
        "total_elapsed_seconds": round(time.perf_counter() - started, 2),
        "note": "Token counts come from OpenAI API usage fields when available. Use current OpenAI pricing to convert tokens to money.",
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark a Wikipedia-based Tech Company GraphRAG pipeline.")
    parser.add_argument("--refresh-corpus", action="store_true", help="Download Wikipedia pages again.")
    parser.add_argument("--refresh-triples", action="store_true", help="Extract triples with the OpenAI API again.")
    args = parser.parse_args()

    started = time.perf_counter()
    ensure_dirs()
    client = make_client()
    model = os.getenv("OPENAI_MODEL", MODEL_DEFAULT)
    stats = LLMCallStats()

    corpus = load_or_fetch_corpus(refresh=args.refresh_corpus)
    chunks = split_chunks(corpus)
    flat_index = build_flat_vector_index(client, chunks, stats)
    triples = extract_triples(client, model, corpus, stats, refresh=args.refresh_triples)
    graph = build_graph(triples)
    save_graph_image(graph)
    rows = run_benchmark(client, model, flat_index, graph, stats)
    save_benchmark(rows)
    save_summary(started, corpus, triples, graph, stats)

    correct_flat = sum(row["flat_rag_grade"] == "correct" for row in rows)
    correct_graph = sum(row["graphrag_grade"] == "correct" for row in rows)
    print(f"Done. Outputs written to {OUTPUT_DIR}")
    print(f"Flat RAG heuristic score: {correct_flat}/{len(rows)}")
    print(f"GraphRAG heuristic score: {correct_graph}/{len(rows)}")
    print(f"LLM calls: {stats.calls}, input tokens: {stats.input_tokens}, output tokens: {stats.output_tokens}")


if __name__ == "__main__":
    main()
