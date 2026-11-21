
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATABASE = ROOT / "vector_database"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "indian_laws"

# These remain unloaded when the backend starts.
model = None
collection = None


def initialize_rag():
    global model, collection

    if model is not None and collection is not None:
        return True

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        print("Loading legal search database...")

        client = chromadb.PersistentClient(path=str(DATABASE))

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME
        )

        if collection.count() == 0:
            print("Legal database is empty.")
            return False

        print("Loading legal search model...")

        model = SentenceTransformer(MODEL_NAME)

        print("Legal search model loaded successfully.")
        return True

    except Exception as error:
        print("RAG initialization failed:", error)
        model = None
        collection = None
        return False


def get_search_queries(question):
    text = question.lower()

    queries = [question]

    if any(term in text for term in [
        "fundamental rights",
        "constitutional rights",
        "rights under constitution"
    ]):
        queries.extend([
            "Article 14 equality before law and equal protection",
            "Article 15 prohibition of discrimination",
            "Article 16 equality of opportunity in public employment",
            "Article 19 freedom of speech and expression",
            "Article 21 protection of life and personal liberty",
            "Article 21A right to education",
            "Article 22 protection against arrest and detention",
            "Article 23 prohibition of human trafficking and forced labour",
            "Article 24 prohibition of child labour",
            "Article 25 freedom of religion",
            "Article 32 constitutional remedies"
        ])

    return queries


def search_legal_documents(question, number_of_results=5):

    if not question or not question.strip():
        return []

    if not initialize_rag():
        return []

    try:
        if collection.count() == 0:
            return []

        queries = get_search_queries(question)

        # Encode all queries together instead of separately.
        embeddings = model.encode(queries)

        found = {}

        for search_query, query_embedding in zip(queries, embeddings):

            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(4, collection.count()),
                include=["documents", "metadatas", "distances"]
            )

            for i, document in enumerate(results["documents"][0]):

                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]

                key = (
                    metadata["document"],
                    metadata["page"],
                    metadata["chunk"]
                )

                if key not in found or distance < found[key]["distance"]:

                    found[key] = {
                        "text": document,
                        "document": metadata["document"],
                        "category": metadata["category"],
                        "page": metadata["page"],
                        "distance": distance
                    }

        ranked_results = sorted(
            found.values(),
            key=lambda item: item["distance"]
        )

        return ranked_results[:number_of_results]

    except Exception as error:
        print("Legal document search failed:", error)
        return []


if __name__ == "__main__":

    question = input("Enter your legal question: ")

    results = search_legal_documents(question)

    print("\nRELEVANT LEGAL INFORMATION")
    print("=" * 60)

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")
        print("Document:", result["document"])
        print("Page:", result["page"])
        print("Distance:", round(result["distance"], 4))
        print("Text:", result["text"][:1000])
        print("-" * 60)
