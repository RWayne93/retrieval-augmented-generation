import os

import streamlit as st
import phoenix as px
from phoenix.trace.langchain import LangChainInstrumentor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.runnables import RunnableConfig

from src import CFG
from src.retrieval_qa import build_retrieval_chain
from src.vectordb import build_vectordb, load_faiss, load_chroma, load_pgvector, check_pgvector_connection
from streamlit_app.utils import perform, load_base_embeddings, load_llm, load_reranker

st.set_page_config(page_title="Conversational Retrieval QA")

LLM = load_llm()
BASE_EMBEDDINGS = load_base_embeddings()
RERANKER = load_reranker()

if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = ""

if 'phoenix_initialized' not in st.session_state:
    session = px.launch_app()
    phoenix_tracer = LangChainInstrumentor().instrument()
    st.session_state['phoenix_initialized'] = True

def init_chat_history():
    """Initialise chat history."""
    clear_button = st.sidebar.button("Clear Chat", key="clear")
    if clear_button:
        st.session_state["chat_history"] = list()
        st.session_state["display_history"] = [("", "Hello! How can I help you?", None)]
    elif "chat_history" not in st.session_state or "display_history" not in st.session_state:
        st.session_state.setdefault("chat_history", list())
        st.session_state.setdefault("display_history", [("", "Hello! How can I help you?", None)])


@st.cache_resource
def load_vectordb():
    if CFG.VECTORDB_TYPE == "faiss":
        return load_faiss(BASE_EMBEDDINGS)
    if CFG.VECTORDB_TYPE == "chroma":
        return load_chroma(BASE_EMBEDDINGS)
    if CFG.VECTORDB_TYPE == "pgvector":
        return load_pgvector(BASE_EMBEDDINGS, CFG.VECTORDB_PATH, CFG.COLLECTION_NAME)
    raise NotImplementedError

def display_source_document_info(row):
    # Check if 'page' key exists before accessing it
    if 'page' in row.metadata:
        st.write("**Page {}**".format(row.metadata["page"] + 1))
    else:
        # Handle the case where 'page' key is missing
        st.write("**Page information not available**")
    st.info(row.page_content)

def doc_conv_qa():
    init_chat_history()

    with st.sidebar:
        st.title("Conversational RAG with quantized LLM")

        with st.expander("Models used"):
            st.info(f"LLM: `{CFG.LLM_PATH}`")
            st.info(f"Embeddings: `{CFG.EMBEDDINGS_PATH}`")
            st.info(f"Reranker: `{CFG.RERANKER_PATH}`")

        uploaded_files = st.file_uploader(
            "Upload PDFs and build VectorDB", type=["pdf"], accept_multiple_files=True
        )
        if st.button("Build VectorDB"):
            if not uploaded_files:
                st.error("No PDF uploaded")
            else:
                for uploaded_file in uploaded_files:
                    with st.spinner(f"Building VectorDB for {uploaded_file.name}..."):
                        perform(build_vectordb, uploaded_file.read())
                st.session_state.uploaded_filenames = [file.name for file in uploaded_files]

        if st.session_state.get("uploaded_filename", "") != "":
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
            with st.status("Load retrieval chain", expanded=False) as status:
                st.write("Loading retrieval chain...")
                vectordb = load_vectordb()
                retrieval_chain = build_retrieval_chain(vectordb, RERANKER, LLM)
                status.update(
                    label="Loading complete!", state="complete", expanded=False
                )
            st.success("Reading from existing VectorDB")
        except Exception as e:
            print(e)
            st.error(e)
            st.stop()

    st.sidebar.write("---")

    # Display chat history
    for question, answer, source_documents in st.session_state.display_history:
        if question != "":
            with st.chat_message("user"):
                st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)

            if source_documents is not None:
                with st.expander("Sources"):
                    for row in source_documents:
                        display_source_document_info(row)

    if user_query := st.chat_input("Your query"):
        with st.chat_message("user"):
            st.markdown(user_query)

        query_embedding = BASE_EMBEDDINGS.embed_query(user_query)
        print(f"Query Embedding for '{user_query}': {query_embedding}")

    if user_query is not None:
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(
                parent_container=st.container(),
                expand_new_thoughts=True,
                collapse_completed_thoughts=True,
            )
            response = retrieval_chain.invoke(
                {
                    "question": user_query,
                    "chat_history": st.session_state.chat_history,
                },
                config=RunnableConfig(callbacks=[st_callback]),
            )
            st_callback._complete_current_thought()
            st.markdown(response["answer"])

            with st.expander("Sources"):
                print(response["source_documents"])
                for row in response["source_documents"]:
                    # Debugging: Print the keys available in row.metadata
                    # print("Available keys in metadata:", row.metadata.keys())
                    display_source_document_info(row)

            st.session_state.chat_history.append(
                (response["question"], response["answer"])
            )
            st.session_state.display_history.append(
                (response["question"], response["answer"], response["source_documents"])
            )
    
if __name__ == "__main__":
    doc_conv_qa()