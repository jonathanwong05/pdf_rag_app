import redis
import numpy as np
import ollama
from redis.commands.search.query import Query
from .utils import get_redis_client, get_sentence_transformer, VECTOR_DIM, INDEX_NAME, DOC_PREFIX
from .utils import get_embedding_st_minilm 

def search_embeddings(query, top_k=5):

    redis_client = get_redis_client()

    query_embedding = get_embedding_st_minilm(query)

    # Convert embedding to bytes for Redis search
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        # Construct the vector similarity search query
        # Use a more standard RediSearch vector search syntax
        # q = Query("*").sort_by("embedding", query_vector)

        q = (
            Query(f"*=>[KNN {top_k} @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("file", "page", "chunk", "vector_distance")
            .dialect(2)
        )

        # Perform the search
        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )

        # Transform results into the expected format
        top_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": float(result.vector_distance),
            }
            for result in results.docs
        ][:top_k]

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, context_results, model_name, prompt_template):

    # Prepare context string
    context_str = "\n\n".join(
        [
            f"From {res['file']} (page {res['page']}):\n\"{res['chunk']}\"\n(similarity: {float(res['similarity']):.2f})"
            for res in context_results
        ]
    )

        # Default prompt if none provided
    if prompt_template is None:
        prompt_template = (
            "You are a helpful assistant. Use the following context excerpts from user‐uploaded PDFs to answer their question.\n"
            "If none of the context is relevant, say \"I don't know.\"\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{query}\n\n"
            "Answer:"
        )

    prompt = prompt_template.replace("{context}", context_str).replace("{query}", query)

    response = ollama.chat(
        model=model_name, messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        context_results = search_embeddings(query)

        # Generate RAG response
        response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(response)
