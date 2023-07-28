"""
RetrievalQA
"""
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.vectorstores import FAISS

from src import CFG

qa_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def build_retrieval_qa(llm: CTransformers, vectordb: FAISS):
    """Builds a retrieval QA model.

    Args:
        llm (CTransformers): The language model to use.
        vectordb (FAISS): The vector database to use.

    Returns:
        RetrievalQA: The retrieval QA model.
    """
    prompt = PromptTemplate(
        template=qa_template,
        input_variables=["context", "question"],
    )

    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": CFG.SEARCH_K}),
        return_source_documents=CFG.RETURN_SOURCE_DOCUMENTS,
        chain_type_kwargs={"prompt": prompt},
    )
    return retrieval_qa
