from fastapi import FastAPI
from qdrant_client import QdrantClient, models
from llama_index.llms.ollama import Ollama
import ollama
import pdfplumber
import json
import csv
import tiktoken
from uuid import uuid4
from nltk.tokenize import sent_tokenize
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import re
import nltk

input_csv_file = "input.csv"
output_json_file = "output.json"

from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
collection_name = "data_1"
llm_model = "llama3.1:latest"
embedding_model = "qwen3-embedding:latest"

input_csv_file = "res/input.csv"
output_json_file = "output.json"

llm = Ollama(
    model=llm_model,
    request_timeout=1000,
    #context_window=8000,
)

def create_collect():

    if collection_name not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name = collection_name,
            vectors_config = models.VectorParams(
                size=4,
                distance=models.Distance.COSINE) 
        )

        client.create_payload_index(
            collection_name=collection_name,
            field_name="category",
            field_schema=models.PayloadSchemaType.KEYWORD
        )

    return client

def load_pdf(file_path: str) -> str:
    full_text = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text.append(f"\n\n ---Page {i+1}---\n{text}\n")

    
    # with open("example.txt", "a") as file:
    #     file.writelines(full_text)

    with open("output.json", 'w', encoding='utf-8') as f:
        json.dump(("\n".join(full_text)), f, ensure_ascii=False, indent=4)
        
    return full_text


def embed_text(text:str)->list[float]:
    return ollama.embeddings(
        model=embedding_model,
        prompt=text
    )["embedding"]

def embed_json(text:str):

    # "id": "",
    # "equipment": {
    #   "problem": "",
    #   "trouble_code": "",
    #   "error": "",
    #   "symptom_short": "",
    #   "sympton": "",
    #   "signs": "",
    #   "condition": "",
    #   "root_cause": "",
    #   "Most_Probable_Cause": "",
    #   "Troubleshoot_Procedure

    with open(text,"r", encoding="utf-8") as f:
        data = json.load(f)

    embedded_points = []

    for item in data:
        combined_text = f"{item['equipment']}"
        vector = embed_text(combined_text)

        embedded_points.append(
            {
                "id":item["id"],
                "vector":vector,
                "payload": item
            }
        )

    client.upsert(
            collection_name=collection_name,
            points=embedded_points
        )
    print("emded done")

def call_llm(query, context):
    prompt = f"""
    You are a technical assistant.
    Answer using ONLY the context below.

    Context:
    {context}

    Question:
    {query}
    """
    resp = ollama.chat(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp["message"]["content"]

def index_chunks(client, chunks, batch_size=16):
    from qdrant_client.models import PointStruct

    batch = []

    for i, c in enumerate(chunks, 1):
        vec = embed_text(c["text"])

        batch.append(
            PointStruct(
                id=c["id"],
                vector=vec,
                payload={
                    "text": c["text"],
                    **c["metadata"]
                }
            )
        )

        if len(batch) >= batch_size:
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
            batch.clear()
            print(f"Indexed {i}/{len(chunks)} chunks")

    if batch:
        client.upsert(
            collection_name=collection_name,
            points=batch
        )

def retrieve(client, query, top_k=5):
    qvec = embed_text(query)

    hits = client.query_points(
        collection_name=collection_name,
        query=qvec,
        limit=top_k,
        with_payload=True
    )
    print(f"Retrieved hits for query: '{hits}'")

    return [
        point.payload["id"]
        for point in hits.points
        if point.payload and "id" in point.payload
    ]


class Chunker:
    def __init__(self):
        self.encoder = tiktoken.encoding_for_model("gpt-4o-mini")

    def tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def split_sections(self, text: str):
        pattern = r"(?:^|\n)(#+\s.*|\n[A-Z][^\n]{5,}\n)"
        parts = re.split(pattern, text)

        sections = []
        title = "General"

        for p in parts:
            if p.strip().startswith("#") or p.strip().isupper():
                title = p.strip()
            elif p.strip():
                sections.append({"title": title, "content": p.strip()})

        return sections
    
    def chunk(self, text: str):
        chunks = []

        for section in self.split_sections(text):
            sentences = sent_tokenize(section["content"])
            buf, buf_tokens = [], 0

            for s in sentences:
                t = self.tokens(s)

                if buf_tokens + t > MAX_TOKENS:
                    chunks.append(self.make_chunk(section["title"], buf))
                    buf = buf[-OVERLAP_SENTENCES:]
                    buf_tokens = self.tokens(" ".join(buf))

                buf.append(s)
                buf_tokens += t

            if buf:
                chunks.append(self.make_chunk(section["title"], buf))

        return chunks
    
    def make_chunk(self, title, sentences):
        text = " ".join(sentences)
        return {
            "id": str(uuid4()),
            "text": text,
            "metadata": {
                "section": title,
                "tokens": self.tokens(text)
            }
        }    


if __name__ == "__main__":

    # text = load_pdf("res/Mdt_error_manual.pdf")
    # chunker = Chunker()
    # chunks = chunker.chunk(text)
    # print(f"chunks created {len(chunks)}")

    qdrant = create_collect()
    embed_json("output.json")
    

    while True:

        q = input("ask or exit:")
        if q.lower() == "exit":
            break
        ctx = "\n\n".join(retrieve(qdrant, q))
        print("\nContext:\n", ctx)
        print("\nAnswer:\n", call_llm(q, ctx))

    
