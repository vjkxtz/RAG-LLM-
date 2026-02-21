import os
import ollama
import pdfplumber
import tiktoken
from uuid import uuid4
from nltk.tokenize import sent_tokenize
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import re
import nltk

EMBEDDING_MODEL = "llama3.1:latest"
VECTOR_SIZE = 4096
COLLECTION_NAME = "maintenance_data"

MAX_TOKENS =500
OVERLAP_SENTENCES = 2

QDRANT_PATH ="./qdrant_data"
nltk.download("punkt")
nltk.download("punkt_tab")

def load_pdf(file_path: str) -> str:
    full_text = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text.append(f"\n\n ---Page {i + 1}---\n{text}\n")
    return "\n".join(full_text)



def embed_text(text: str) -> list[float]:
    return ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text
    )["embedding"]

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


def init_qdrant():
    client = QdrantClient(path=QDRANT_PATH)

    if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
    return client


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
                collection_name=COLLECTION_NAME,
                points=batch
            )
            batch.clear()
            print(f"Indexed {i}/{len(chunks)} chunks")

    if batch:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )

def retrieve(client, query, top_k=5):
    qvec = embed_text(query)

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,
        limit=top_k,
        with_payload=True
    )
    print(f"Retrieved hits for query: '{hits}'")

    return [
        point.payload["text"]
        for point in hits.points
        if point.payload and "text" in point.payload
    ]


def answer(query, context):
    prompt = f"""
    You are a technical assistant.
    Answer using ONLY the context below.

    Context:
    {context}

    Question:
    {query}
    """

    resp = ollama.chat(
        model=EMBEDDING_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp["message"]["content"]




def main():

    #data = embed_text("This is a sample text to embed.")
    
    #print(len(data))
    
    text = load_pdf("ACOPOS error list.pdf")

    chunker = Chunker()
    chunks = chunker.chunk(text)

    print(f"Chunks created: {len(chunks)}")

    qdrant = init_qdrant()

    index_chunks(qdrant, chunks)

    print("Indexing complete ✅")

    while True:
        q = input("\nAsk (or 'exit'): ")
        if q.lower() == "exit":
            break

        ctx = "\n\n".join(retrieve(qdrant, q))
        print("\nContext:\n", ctx)
        print("\nAnswer:\n", answer(q, ctx))


if __name__ == "__main__":
    main()



