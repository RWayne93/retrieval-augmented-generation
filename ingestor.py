# import os
# from src.vectordb import build_vectordb
# from src.embeddings import build_base_embeddings
# from streamlit_app.utils import perform
# from tqdm import tqdm

# documents_directory = "documents"

# def ingest_documents(directory: str) -> None:
#     """Ingest documents from the specified directory, ignoring macOS .DS_Store files."""
#     document_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f != ".DS_Store"]
#     #total_documents = len(document_files)
#     #print(f"Total documents to ingest: {total_documents}")

#     embedding_model = build_base_embeddings()

#     for filename in tqdm(document_files):
#         file_path = os.path.join(directory, filename)
#      #   print(f"Ingesting {filename}...")
#         with open(file_path, 'rb') as file:
#             perform(build_vectordb, file.read(), embedding_function=embedding_model)
#         #print(f"Finished ingesting {filename}.")
#     print("Ingestion complete.")

# if __name__ == "__main__":
#     ingest_documents(documents_directory)

import os
import argparse
from src.vectordb import build_vectordb
from src.embeddings import build_base_embeddings
from streamlit_app.utils import perform
from tqdm import tqdm

def ingest_document(file_path: str, embedding_model) -> None:
    """Ingest a single document."""
    if os.path.isfile(file_path) and file_path != ".DS_Store":
        with open(file_path, 'rb') as file:
            perform(build_vectordb, file.read(), embedding_function=embedding_model)
        print(f"Finished ingesting {file_path}.")
    else:
        print(f"File {file_path} does not exist or is a directory.")

def ingest_multiple_documents(file_paths: list, embedding_model) -> None:
    """Ingest multiple documents."""
    for file_path in file_paths:
        ingest_document(file_path, embedding_model)

def ingest_directory(directory: str, embedding_model) -> None:
    """Ingest documents from the specified directory, ignoring macOS .DS_Store files."""
    document_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f != ".DS_Store"]
    for filename in tqdm(document_files):
        file_path = os.path.join(directory, filename)
        ingest_document(file_path, embedding_model)
    print("Ingestion complete.")

def main():
    parser = argparse.ArgumentParser(description="Ingest documents for vector database.")
    parser.add_argument("--directory", type=str, help="Directory containing documents to ingest.")
    parser.add_argument("--file", nargs='+', help="One or more document files to ingest.")
    
    args = parser.parse_args()
    
    embedding_model = build_base_embeddings()

    if args.file:
        ingest_multiple_documents(args.file, embedding_model)
    elif args.directory:
        ingest_directory(args.directory, embedding_model)
    else:
        print("No directory or file specified. Please use --directory or --file.")

if __name__ == "__main__":
    main()