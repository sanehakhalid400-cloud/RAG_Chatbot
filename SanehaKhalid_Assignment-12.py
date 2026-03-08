import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── ENV & PAGE SETUP ───────────────────────────────────────────────
def setup_env_and_page():
    load_dotenv()
    st.set_page_config(page_title="🤖 RAG Document Chatbot", layout="wide")
    st.title("🧠 RAG Document Chatbot – Ask Questions from PDFs")
    
setup_env_and_page()

# ── GET API KEY ─────────────────────────────────────────────────────
def get_api_key():
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key_input = st.text_input("🔑 Groq API Key", type="password")
        st.caption("Upload PDFs → Ask questions → Get Answers 💬")
    api_key = api_key_input or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.warning("⚠️ Please enter your Groq API Key or set GROQ_API_KEY in .env")
        st.stop()
    return api_key

api_key = get_api_key()

# ── LOAD EMBEDDINGS ───────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

embeddings = load_embeddings()

# ── LOAD LLM ──────────────────────────────────────────────────────
@st.cache_resource
def load_llm(key):
    return ChatGroq(
        groq_api_key=key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

llm = load_llm(api_key)

# ── UPLOAD PDFs ───────────────────────────────────────────────────
def upload_pdfs():
    uploaded_files = st.file_uploader(
        "📂 Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("ℹ️ Please upload one or more PDFs to begin")
        st.stop()
    all_docs = []
    tmp_paths = []
    for pdf in uploaded_files:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf.getvalue())
        tmp.close()
        tmp_paths.append(tmp.name)
        loader = PyPDFLoader(tmp.name)
        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = pdf.name
        all_docs.extend(docs)
    st.success(f"✅ Loaded {len(all_docs)} pages from {len(uploaded_files)} PDFs 📖")
    for p in tmp_paths:
        try:
            os.unlink(p)
        except Exception:
            pass
    return all_docs

all_docs = upload_pdfs()

# ── SPLIT DOCUMENTS ───────────────────────────────────────────────
def split_documents(docs, chunk_size=1200, overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    splits = text_splitter.split_documents(docs)
    if not splits:
        st.warning("⚠️ No text chunks were created from the uploaded PDFs! Please check your files.")
        st.stop()
    return splits

splits = split_documents(all_docs)

# ── CREATE VECTORSTORE ───────────────────────────────────────────
def create_vectorstore(splits, embeddings, index_dir="chroma_index"):
    vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=index_dir)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":5, "fetch_k":20})
    st.sidebar.write(f"🔍 Indexed {len(splits)} chunks for retrieval")
    return vectorstore, retriever

vectorstore, retriever = create_vectorstore(splits, embeddings)

# ── PROMPTS ──────────────────────────────────────────────────────
def create_prompts():
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a query rewriting assistant. Convert the user's latest question into a clear standalone search query using the chat history. Return ONLY the rewritten query."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a STRICT RAG assistant. Answer ONLY using the provided context.\n"
         "If the answer is not in the context, respond exactly:\n"
         "'Out of scope - not found in provided documents.'\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    return contextualize_q_prompt, qa_prompt

contextualize_q_prompt, qa_prompt = create_prompts()

# ── CHAT HISTORY MANAGEMENT ───────────────────────────────────────
def get_history(session_id):
    if "chat_history_store" not in st.session_state:
        st.session_state.chat_history_store = {}
    if session_id not in st.session_state.chat_history_store:
        st.session_state.chat_history_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_history_store[session_id]

# ── JOIN DOCUMENTS ───────────────────────────────────────────────
def join_docs(docs, max_chars=7000):
    chunks, total = [], 0
    for d in docs:
        piece = d.page_content
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(chunks)

# ── CHAT INTERFACE ──────────────────────────────────────────────
def run_chat():
    session_id = st.text_input("🆔 Session ID", value="default")
    user_q = st.chat_input("💬 Ask a question about your PDFs...")
    history = get_history(session_id)
    for msg in history.messages:
        if msg.type == "human":
            st.chat_message("user").write(msg.content)
        else:
            st.chat_message("assistant").write(msg.content)

    if user_q:
        with st.spinner("🤖 AI is generating your answer..."):
            rewrite_msgs = contextualize_q_prompt.format_messages(
                chat_history=history.messages,
                input=user_q
            )
            standalone_q = llm.invoke(rewrite_msgs).content.strip()
            docs = retriever.invoke(standalone_q)
            if not docs:
                answer = "Out of scope - not found in provided documents. ❌"
            else:
                context_str = join_docs(docs)
                qa_msgs = qa_prompt.format_messages(
                    chat_history=history.messages,
                    input=user_q,
                    context=context_str
                )
                answer = llm.invoke(qa_msgs).content + " ✅"
        st.chat_message("user").write(f"👤 {user_q}")
        st.chat_message("assistant").write(f"🤖 {answer}")
        if docs:
            sources = set(d.metadata.get("source_file", "Unknown") for d in docs)
            st.markdown(f"📚 **Sources:** {', '.join(sources)}")
        history.add_user_message(user_q)
        history.add_ai_message(answer)
        with st.expander("🧪 Debug Retrieval & Standalone Query"):
            st.write("📝 Standalone Query:")
            st.code(standalone_q)
            st.write(f"🔍 Retrieved {len(docs)} chunk(s)")
            for i, d in enumerate(docs, 1):
                st.markdown(f"{i}. 📄 {d.metadata.get('source_file', 'Unknown')} (p{d.metadata.get('page','?')})")
                st.write(d.page_content[:400] + "...")

run_chat()