# import os

# import streamlit as st
# from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
# from langchain_core.runnables import RunnableConfig

# from src import CFG
# from src.embeddings import build_hyde_embeddings
# from src.query_expansion import build_multiple_queries_expansion_chain
# from src.retrieval_qa import (
#     build_retrieval_qa,
#     build_base_retriever,
#     build_rerank_retriever,
#     build_compression_retriever,
# )
# from src.vectordb import build_vectordb, load_faiss, load_chroma, load_pgvector, check_pgvector_connection
# from streamlit_app.pdf_display import get_doc_highlighted, display_pdf
# from streamlit_app.utils import perform, load_base_embeddings, load_llm, load_reranker

# import phoenix as px
# from phoenix.trace.langchain import LangChainInstrumentor

# st.set_page_config(page_title="Retrieval QA", layout="wide")

# LLM = load_llm()
# BASE_EMBEDDINGS = load_base_embeddings()
# RERANKER = load_reranker()

# if 'phoenix_initialized' not in st.session_state:
#     session = px.launch_app()
#     phoenix_tracer = LangChainInstrumentor().instrument()
#     # Mark Phoenix as initialized in this session
#     st.session_state['phoenix_initialized'] = True


# @st.cache_resource
# def load_vectordb():
#     print("Loading vectordb")
#     if CFG.VECTORDB_TYPE == "faiss":
#         return load_faiss(BASE_EMBEDDINGS)
#     if CFG.VECTORDB_TYPE == "chroma":
#         return load_chroma(BASE_EMBEDDINGS)
#     if CFG.VECTORDB_TYPE == "pgvector":
#         return load_pgvector(BASE_EMBEDDINGS, CFG.VECTORDB_PATH, CFG.COLLECTION_NAME)
#     raise NotImplementedError


# @st.cache_resource
# def load_vectordb_hyde():
#     hyde_embeddings = build_hyde_embeddings(LLM, BASE_EMBEDDINGS)

#     if CFG.VECTORDB_TYPE == "faiss":
#         return load_faiss(hyde_embeddings)
#     if CFG.VECTORDB_TYPE == "chroma":
#         return load_chroma(hyde_embeddings)
#     if CFG.VECTORDB_TYPE == "pgvector":
#         return load_pgvector(hyde_embeddings, CFG.VECTORDB_PATH, CFG.COLLECTION_NAME)
#     raise NotImplementedError


# def load_retriever(_vectordb, retrieval_mode):
#     if retrieval_mode == "Base":
#         return build_base_retriever(_vectordb)
#     if retrieval_mode == "Rerank":
#         return build_rerank_retriever(_vectordb, RERANKER)
#     if retrieval_mode == "Contextual compression":
#         return build_compression_retriever(_vectordb, BASE_EMBEDDINGS)
#     raise NotImplementedError


# def init_sess_state():
#     if "uploaded_filename" not in st.session_state:
#         st.session_state["uploaded_filename"] = ""

#     if "last_form" not in st.session_state:
#         st.session_state["last_form"] = list()

#     if "last_query" not in st.session_state:
#         st.session_state["last_query"] = ""

#     if "last_response" not in st.session_state:
#         st.session_state["last_response"] = dict()

#     if "last_related" not in st.session_state:
#         st.session_state["last_related"] = list()


# def doc_qa():
#     init_sess_state()

#     with st.sidebar:
#         st.header("RAG with quantized LLM")

#         with st.expander("Models used"):
#             st.info(f"LLM: `{CFG.LLM_PATH}`")
#             st.info(f"Embeddings: `{CFG.EMBEDDINGS_PATH}`")
#             st.info(f"Reranker: `{CFG.RERANKER_PATH}`")

#         uploaded_file = st.file_uploader(
#             "Upload a PDF and build VectorDB", type=["pdf"]
#         )
#         if st.button("Build VectorDB"):
#             if uploaded_file is None:
#                 st.error("No PDF uploaded")
#             else:
#                 with st.spinner("Building VectorDB..."):
#                     perform(build_vectordb, uploaded_file.read())
#                 st.session_state.uploaded_filename = uploaded_file.name

#         if st.session_state.uploaded_filename != "":
#             st.info(f"Current document: {st.session_state.uploaded_filename}")

#         vectordb_ready = False
#         if CFG.VECTORDB_TYPE == "pgvector":
#             vectordb_ready = check_pgvector_connection(CFG.VECTORDB_PATH)
#         else:
#             vectordb_ready = os.path.exists(CFG.VECTORDB_PATH)

#         if not vectordb_ready:
#             st.info("Please build VectorDB first.")
#             st.stop()

#         try:
#             with st.status("Load VectorDB", expanded=False) as status:
#                 st.write("Loading VectorDB ...")
#                 vectordb = load_vectordb()
#                 st.write("Loading HyDE VectorDB ...")
#                 vectordb_hyde = load_vectordb_hyde()
#                 status.update(
#                     label="Loading complete!", state="complete", expanded=False
#                 )
#             st.success("Reading from existing VectorDB")
#         except Exception as e:
#             st.error(e)
#             st.stop()

#     c0, c1 = st.columns(2)

#     with c0.form("qa_form"):
#         user_query = st.text_area("Your query")
#         with st.expander("Settings"):
#             mode = st.radio(
#                 "Mode",
#                 ["Retrieval only", "Retrieval QA"],
#                 index=1,
#                 help="""Retrieval only will output extracts related to your query immediately, \
#                 while Retrieval QA will output an answer to your query and will take a while on CPU.""",
#             )
#             retrieval_mode = st.radio(
#                 "Retrieval method",
#                 ["Base", "Rerank", "Contextual compression"],
#                 index=1,
#             )
#             use_hyde = st.checkbox("Use HyDE (for Retrieval QA only)")

#         submitted = st.form_submit_button("Query")
#         if submitted:
#             if user_query == "":
#                 st.error("Please enter a query.")

#     if user_query != "" and (
#         st.session_state.last_query != user_query
#         or st.session_state.last_form != [mode, retrieval_mode, use_hyde]
#     ):
#         st.session_state.last_query = user_query
#         st.session_state.last_form = [mode, retrieval_mode, use_hyde]

#         if mode == "Retrieval only":
#             retriever = load_retriever(vectordb, retrieval_mode)
#             with c0:
#                 with st.spinner("Retrieving ..."):
#                     relevant_docs = retriever.get_relevant_documents(user_query)

#             st.session_state.last_response = {
#                 "query": user_query,
#                 "source_documents": relevant_docs,
#             }

#             chain = build_multiple_queries_expansion_chain(LLM)
#             res = chain.invoke(user_query)
#             st.session_state.last_related = [
#                 x.strip() for x in res.split("\n") if x.strip()
#             ]
#         else:
#             db = vectordb_hyde if use_hyde else vectordb
#             retriever = load_retriever(db, retrieval_mode)
#             retrieval_qa = build_retrieval_qa(LLM, retriever)

#             st_callback = StreamlitCallbackHandler(
#                 parent_container=c0.container(),
#                 expand_new_thoughts=True,
#                 collapse_completed_thoughts=True,
#             )
#             st.session_state.last_response = retrieval_qa.invoke(
#                 user_query, config=RunnableConfig(callbacks=[st_callback])
#             )
#             st_callback._complete_current_thought()

#     if st.session_state.last_response:
#         with c0:
#             st.warning(f"##### {st.session_state.last_query}")
#             if st.session_state.last_response.get("result") is not None:
#                 st.success(st.session_state.last_response["result"])

#             if st.session_state.last_related:
#                 st.write("#### Related")
#                 for r in st.session_state.last_related:
#                     st.write(f"```\n{r}\n```")

#         with c1:
#             st.write("#### Sources")
#             for row in st.session_state.last_response["source_documents"]:
#                 if 'page' in row.metadata:
#                     st.write("**Page {}**".format(row.metadata["page"] + 1))
#                 else:
#                     st.write("**Page information not available**")
#                 st.info(row.page_content.replace("$", r"\$"))

#             # Display PDF
#             st.write("---")
#             _display_pdf_from_docs(st.session_state.last_response["source_documents"])
    
    
    

# def _display_pdf_from_docs(source_documents):
#     n = len(source_documents)
#     i = st.radio(
#         "View in PDF", list(range(n)), format_func=lambda x: f"Extract {x + 1}"
#     )
#     row = source_documents[i]
#     try:
#         extracted_doc, page_nums = get_doc_highlighted(
#             row.metadata["source"], row.page_content
#         )
#         if extracted_doc is None:
#             st.error("No page found")
#         else:
#             display_pdf(extracted_doc, page_nums[0] + 1)
#     except Exception as e:
#         st.error(e)


# if __name__ == "__main__":
#     doc_qa()



import os

import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.runnables import RunnableConfig

from src import CFG
from src.embeddings import build_hyde_embeddings
from src.query_expansion import build_multiple_queries_expansion_chain
from src.retrieval_qa import (
    build_retrieval_qa,
    build_base_retriever,
    build_rerank_retriever,
    build_compression_retriever,
)
from src.vectordb import build_vectordb, load_faiss, load_chroma, load_pgvector, check_pgvector_connection
from streamlit_app.pdf_display import get_doc_highlighted, display_pdf
from streamlit_app.utils import perform, load_base_embeddings, load_llm, load_reranker

import phoenix as px
from phoenix.trace.langchain import LangChainInstrumentor
import mlflow
import json
import numpy as np

def serialize_object(obj):
    if isinstance(obj, np.floating):
        return float(obj)  # Convert numpy floats to Python floats
    elif isinstance(obj, np.integer):
        return int(obj)  # Convert numpy integers to Python integers
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif "Document" in str(type(obj)):  # Adjust condition based on actual type
        return {
            "page_content": getattr(obj, 'page_content', ''),
            "metadata": getattr(obj, 'metadata', {})
        }
    else:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    

mlflow.set_tracking_uri("http://localhost:5000")
#mlflow.set_tracking_uri("http://172.26.46.168:5000")
st.set_page_config(page_title="Retrieval QA", layout="wide")

LLM = load_llm()
BASE_EMBEDDINGS = load_base_embeddings()
RERANKER = load_reranker()

if 'phoenix_initialized' not in st.session_state:
    session = px.launch_app()
    phoenix_tracer = LangChainInstrumentor().instrument()
    # Mark Phoenix as initialized in this session
    st.session_state['phoenix_initialized'] = True


#@st.cache_resource
def load_vectordb():
    print("Loading vectordb")
    if CFG.VECTORDB_TYPE == "faiss":
        return load_faiss(BASE_EMBEDDINGS)
    if CFG.VECTORDB_TYPE == "chroma":
        return load_chroma(BASE_EMBEDDINGS)
    if CFG.VECTORDB_TYPE == "pgvector":
        return load_pgvector(BASE_EMBEDDINGS, CFG.VECTORDB_PATH, CFG.COLLECTION_NAME)
    raise NotImplementedError


#@st.cache_resource
def load_vectordb_hyde():
    hyde_embeddings = build_hyde_embeddings(LLM, BASE_EMBEDDINGS)

    if CFG.VECTORDB_TYPE == "faiss":
        return load_faiss(hyde_embeddings)
    if CFG.VECTORDB_TYPE == "chroma":
        return load_chroma(hyde_embeddings)
    if CFG.VECTORDB_TYPE == "pgvector":
        return load_pgvector(hyde_embeddings, CFG.VECTORDB_PATH, CFG.COLLECTION_NAME)
    raise NotImplementedError


def load_retriever(_vectordb, retrieval_mode):
    if retrieval_mode == "Base":
        return build_base_retriever(_vectordb)
    if retrieval_mode == "Rerank":
        return build_rerank_retriever(_vectordb, RERANKER)
    if retrieval_mode == "Contextual compression":
        return build_compression_retriever(_vectordb, BASE_EMBEDDINGS)
    raise NotImplementedError


def init_sess_state():
    if "uploaded_filename" not in st.session_state:
        st.session_state["uploaded_filename"] = ""

    if "last_form" not in st.session_state:
        st.session_state["last_form"] = list()

    if "last_query" not in st.session_state:
        st.session_state["last_query"] = ""

    if "last_response" not in st.session_state:
        st.session_state["last_response"] = dict()

    if "last_related" not in st.session_state:
        st.session_state["last_related"] = list()


def doc_qa():

    init_sess_state()

    if mlflow.active_run():
        mlflow.end_run()

    with st.sidebar:
        st.header("RAG with quantized LLM")

        with st.expander("Models used"):
            st.info(f"LLM: `{CFG.LLM_PATH}`")
            st.info(f"Embeddings: `{CFG.EMBEDDINGS_PATH}`")
            st.info(f"Reranker: `{CFG.RERANKER_PATH}`")

        uploaded_file = st.file_uploader(
            "Upload a PDF and build VectorDB", type=["pdf"]
        )
        if st.button("Build VectorDB"):
            if uploaded_file is None:
                st.error("No PDF uploaded")
            else:
                with st.spinner("Building VectorDB..."):
                    perform(build_vectordb, uploaded_file.read())
                st.session_state.uploaded_filename = uploaded_file.name

        if st.session_state.uploaded_filename != "":
            st.info(f"Current document: {st.session_state.uploaded_filename}")

        vectordb_ready = False
        if CFG.VECTORDB_TYPE == "pgvector":
            vectordb_ready = check_pgvector_connection(CFG.VECTORDB_PATH)
        else:
            vectordb_ready = os.path.exists(CFG.VECTORDB_PATH)

        if not vectordb_ready:
            st.info("Please build VectorDB first.")
            st.stop()

        try:
            with st.status("Load VectorDB", expanded=False) as status:
                st.write("Loading VectorDB ...")
                vectordb = load_vectordb()
                st.write("Loading HyDE VectorDB ...")
                vectordb_hyde = load_vectordb_hyde()
                status.update(
                    label="Loading complete!", state="complete", expanded=False
                )
            st.success("Reading from existing VectorDB")
        except Exception as e:
            st.error(e)
            st.stop()

    c0, c1 = st.columns(2)

    with c0.form("qa_form"):
        user_query = st.text_area("Your query")
        with st.expander("Settings"):
            mode = st.radio(
                "Mode",
                ["Retrieval only", "Retrieval QA"],
                index=1,
                help="""Retrieval only will output extracts related to your query immediately, \
                while Retrieval QA will output an answer to your query and will take a while on CPU.""",
            )
            retrieval_mode = st.radio(
                "Retrieval method",
                ["Base", "Rerank", "Contextual compression"],
                index=1,
            )
            use_hyde = st.checkbox("Use HyDE (for Retrieval QA only)")

        submitted = st.form_submit_button("Query")
        if submitted:
            if user_query == "":
                st.error("Please enter a query.")

    if user_query != "" and (
        st.session_state.last_query != user_query
        or st.session_state.last_form != [mode, retrieval_mode, use_hyde]
    ):
        st.session_state.last_query = user_query
        st.session_state.last_form = [mode, retrieval_mode, use_hyde]

        if mode == "Retrieval only":
            retriever = load_retriever(vectordb, retrieval_mode)
            with c0:
                with st.spinner("Retrieving ..."):
                    relevant_docs = retriever.get_relevant_documents(user_query)

            st.session_state.last_response = {
                "query": user_query,
                "source_documents": relevant_docs,
            }

            chain = build_multiple_queries_expansion_chain(LLM)
            res = chain.invoke(user_query)
            st.session_state.last_related = [
                x.strip() for x in res.split("\n") if x.strip()
            ]
        else:
            if mlflow.active_run():
                mlflow.end_run()
            with mlflow.start_run():
                mlflow.log_param("user_query", user_query)
                mlflow.log_param("retrieval_mode", retrieval_mode)
                mlflow.log_param("use_hyde", use_hyde)

                db = vectordb_hyde if use_hyde else vectordb
                retriever = load_retriever(db, retrieval_mode)
                retrieval_qa = build_retrieval_qa(LLM, retriever)

                st_callback = StreamlitCallbackHandler(
                    parent_container=c0.container(),
                    expand_new_thoughts=True,
                    collapse_completed_thoughts=True,
                )
                response = retrieval_qa.invoke(
                    user_query, config=RunnableConfig(callbacks=[st_callback])
                )
                st.session_state.last_response = response
                serialized_response = json.dumps(response, default=serialize_object)
                # Log the response from the retrieval QA
                mlflow.log_dict(serialized_response, "response.json")

                st_callback._complete_current_thought()
            
            mlflow.end_run()
            #st_callback._complete_current_thought()
    
    if st.session_state.last_response:
        with c0:
            st.warning(f"##### {st.session_state.last_query}")
            if st.session_state.last_response.get("result") is not None:
                st.success(st.session_state.last_response["result"])

            if st.session_state.last_related:
                st.write("#### Related")
                for r in st.session_state.last_related:
                    st.write(f"```\n{r}\n```")

        with c1:
            st.write("#### Sources")
            for row in st.session_state.last_response["source_documents"]:
                if 'page' in row.metadata:
                    st.write("**Page {}**".format(row.metadata["page"] + 1))
                else:
                    st.write("**Page information not available**")
                st.info(row.page_content.replace("$", r"\$"))

            # Display PDF
            st.write("---")
            _display_pdf_from_docs(st.session_state.last_response["source_documents"])
    
    
    

def _display_pdf_from_docs(source_documents):
    n = len(source_documents)
    i = st.radio(
        "View in PDF", list(range(n)), format_func=lambda x: f"Extract {x + 1}"
    )
    row = source_documents[i]
    try:
        extracted_doc, page_nums = get_doc_highlighted(
            row.metadata["source"], row.page_content
        )
        if extracted_doc is None:
            st.error("No page found")
        else:
            display_pdf(extracted_doc, page_nums[0] + 1)
    except Exception as e:
        st.error(e)


if __name__ == "__main__":
    doc_qa()
