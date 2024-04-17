import os
import csv
import re
from typing import List
from src import CFG
from src.retrieval_qa import (
    #build_retrieval_qa,
    build_base_retriever,
    build_rerank_retriever,
    build_compression_retriever,
    build_retrieval_chain
)
from src.vectordb import load_faiss, load_chroma, load_pgvector,check_pgvector_connection
from streamlit_app.utils import load_base_embeddings, load_llm, load_reranker

csv_path = "/home/archie/retrieval-augmented-generation/rag_evaluation.csv"

def load_prompts(csv_path: str) -> List[str]:
    questions_list = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            questions_list.append(row['question'])
    return questions_list

LLM = load_llm()
BASE_EMBEDDINGS = load_base_embeddings()
RERANKER = load_reranker()
RETREIVAL_MODE = "Rerank"

def load_retriever(_vectordb, retrieval_mode):
    if retrieval_mode == "Base":
        return build_base_retriever(_vectordb)
    if retrieval_mode == "Rerank":
        return build_rerank_retriever(_vectordb, RERANKER)
    if retrieval_mode == "Contextual compression":
        return build_compression_retriever(_vectordb, BASE_EMBEDDINGS)
    raise NotImplementedError

def load_vectordb():
    """Load the vector database based on configuration."""
    if CFG.VECTORDB_TYPE == "faiss":
        return load_faiss(BASE_EMBEDDINGS)
    elif CFG.VECTORDB_TYPE == "chroma":
        return load_chroma(BASE_EMBEDDINGS)
    elif CFG.VECTORDB_TYPE == "pgvector":
        return load_pgvector(BASE_EMBEDDINGS, CFG.VECTORDB_PATH, CFG.COLLECTION_NAME)
    else:
        raise NotImplementedError

def clean_text(text):
    """Clean up text by removing extra spaces and trimming whitespace."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main(prompts):
    """Main function to handle conversational QA with a list of prompts."""
    vectordb_ready = False
    if CFG.VECTORDB_TYPE == "pgvector":
        vectordb_ready = check_pgvector_connection(CFG.VECTORDB_PATH)
    else: 
        vectordb_ready = os.path.exists(CFG.VECTORDB_PATH)

    if not vectordb_ready:
        print("VectorDB not found or connection failed. Please build VectorDB first.")
        return

    try:
        print("Loading retrieval chain...")
        vectordb = load_vectordb()
        #retriever = load_retriever(vectordb, RETREIVAL_MODE)
        #retrieval_chain = load_retriever(vectordb, RETREIVAL_MODE)
        #retrieval_qa = build_retrieval_qa(LLM, retriever)
        retrieval_chain = build_retrieval_chain(vectordb, RERANKER, LLM)
        print("Retrieval chain loaded successfully.")
    except Exception as e:
        print(f"Error loading retrieval chain: {e}")
        return

    results = []

    for prompt in prompts:
        response = retrieval_chain.invoke(
            {
                "question": prompt,
                "chat_history": [],
            }
        )

        # Embed the query
        query_embedding = BASE_EMBEDDINGS.embed_query(prompt)
        query_embedding_str = str(query_embedding)

        sources = [f"Source {index + 1}: {clean_text(row.page_content)}" for index, row in enumerate(response.get('source_documents', []))]
        relevance_scores = [row.metadata.get('relevance_score') for row in response.get('source_documents', [])]
        document_uuids = [row.metadata.get('uuid') for row in response.get('source_documents', [])]

        results.append({"text": prompt, "context_text_0": sources, "relevance_score": relevance_scores, "document_uuid": document_uuids, "response": response['answer'], "text_vector": query_embedding_str})

    # Write results to CSV
    with open('results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['text', 'context_text_0', 'response', 'text_vector', 'relevance_score', 'document_uuid']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print("Results written to results.csv.")

if __name__ == "__main__":
    prompts = load_prompts(csv_path)
    #prompts = ["What are the exemptions soldiers need in order to have a beard in the army?", "What is the army female hair standards?", "What is the composition of the Army Clemency and Parole Board Organization?"] 
    main(prompts)
