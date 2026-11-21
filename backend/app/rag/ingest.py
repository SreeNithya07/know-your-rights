
from pathlib import Path
import pymupdf
import chromadb
from sentence_transformers import SentenceTransformer

# Project folders
ROOT = Path(__file__).resolve().parents[3]
KNOWLEDGE_BASE = ROOT / "knowledge_base"
DATABASE = ROOT / "vector_database"

# Embedding model supports English and Hindi
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Chunk settings
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

COLLECTION_NAME = "indian_laws"


def split_text(text):
    """Split extracted text into overlapping chunks."""

    text = " ".join(text.split())

    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def process_pdfs():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(DATABASE))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Indian Constitution and legal documents"}
    )

    all_chunks = []
    all_metadata = []
    all_ids = []

    pdf_files = list(KNOWLEDGE_BASE.rglob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in knowledge_base!")
        return

    for pdf_path in pdf_files:
        print(f"\nReading: {pdf_path.name}")

        document = pymupdf.open(pdf_path)

        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text()

            if not page_text.strip():
                continue

            chunks = split_text(page_text)

            for chunk_number, chunk in enumerate(chunks, start=1):
                chunk_id = (
                    f"{pdf_path.stem}-"
                    f"page-{page_number}-"
                    f"chunk-{chunk_number}"
                )

                all_chunks.append(chunk)
                all_ids.append(chunk_id)

                all_metadata.append({
                    "document": pdf_path.name,
                    "category": pdf_path.parent.name,
                    "page": page_number,
                    "chunk": chunk_number,
                    "source": "https://www.legislative.gov.in/"
                })

        document.close()

    if not all_chunks:
        print("No readable text found!")
        return

    print(f"\nTotal chunks created: {len(all_chunks)}")
    print("Generating embeddings...")

    # Process in batches to manage memory
    batch_size = 64

    for start in range(0, len(all_chunks), batch_size):
        end = start + batch_size

        batch_texts = all_chunks[start:end]
        batch_ids = all_ids[start:end]
        batch_metadata = all_metadata[start:end]

        embeddings = model.encode(
            batch_texts,
            show_progress_bar=False
        ).tolist()

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metadata,
            embeddings=embeddings
        )

        print(f"Stored {min(end, len(all_chunks))}/{len(all_chunks)} chunks")

    print("\nSUCCESS!")
    print("Total stored chunks:", collection.count())
    print("Database location:", DATABASE)


if __name__ == "__main__":
    process_pdfs()
