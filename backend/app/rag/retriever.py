
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[3]
DATABASE = ROOT / "vector_database"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "indian_laws"

print("Loading legal search model...")
model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path=str(DATABASE))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# Expand broad questions into specific constitutional topics
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

    if not question.strip():
        return []

    if collection.count() == 0:
        return []

    queries = get_search_queries(question)

    found = {}

    # Search each query and combine unique results
    for search_query in queries:

        query_embedding = model.encode(search_query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
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

    # Lower distance means a closer semantic match
    ranked_results = sorted(
        found.values(),
        key=lambda item: item["distance"]
    )

    # Return the best unique passages
    return ranked_results[:number_of_results]


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
