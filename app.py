import os
import streamlit as st

# Core Orchestration Modules
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Arxelos RAG Engine",
    layout="wide"
)

class MedicalRAGPipeline:
    def __init__(self, vector_store):
        # Pass the dynamically compiled vector store directly
        self.vector_store = vector_store
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            st.error("Configuration Error: GOOGLE_API_KEY environment variable is missing.")
            st.stop()
            
        self.llm = self._initialize_llm()
        self.prompt = self._initialize_prompt()
        
    def _initialize_llm(self):
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=self.api_key, 
            temperature=0.1
        )
        
    def _initialize_prompt(self) -> ChatPromptTemplate:
        template = """You are an advanced biomedical AI assistant specializing in neuro-oncology and brain tumor research. 
Use the following pieces of retrieved medical literature context to accurately answer the user's question. 
If you do not know the answer based on the context, state that the provided literature does not contain the answer.

---
RETRIEVED CONTEXT:
{context}
---

QUESTION: {question}

EXPERT ANSWER:"""
        return ChatPromptTemplate.from_template(template)
        
    def _format_docs(self, docs) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def compile_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        data_preparation = RunnableParallel({
            "context": retriever | self._format_docs,
            "question": RunnablePassthrough()
        })
        return data_preparation | self.prompt | self.llm | StrOutputParser()

# --- Cached On-Demand Database Compiler ---
@st.cache_resource
def load_rag_engine():
    """Compiles the vector database directly in cloud RAM on container boot."""
    try:
        # Reuses your exact data engineering parsing functions from Task 02
        from build_vector_db import load_bioc_jsons
        
        corpus_dir = "medical_corpus"
        if not os.path.exists(corpus_dir):
            st.error(f"Data Schema Error: '{corpus_dir}' directory not found in repository.")
            return None
            
        # 1. Parse documents
        docs = load_bioc_jsons(corpus_dir)
        
        # 2. Fragment chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        # 3. Vectorize directly into memory
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
        
        # 4. Bind to pipeline
        pipeline = MedicalRAGPipeline(vector_store=vector_store)
        return pipeline.compile_chain()
        
    except Exception as e:
        st.error(f"On-Demand Compilation Error: {e}")
        return None

# --- UI Layout ---
st.title("Arxelos Engine")
st.caption("Phase 03: Distributed Medical Literature RAG Platform")

with st.sidebar:
    st.subheader("System Status")
    st.markdown("""
    - **Database:** Compiled in-memory (RAM)
    - **Model:** Gemini 2.5 Flash
    - **Parameters:** $k=3$, Temp 0.1
    
    ---
    **Architecture:** Headless on-demand document vectorization pipeline.
    """)

# Trigger compilation (Streamlit caches this so it only runs ONCE on launch)
rag_chain = load_rag_engine()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "System initialized. Vector database compiled in cloud storage. Query the corpus below."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask your Medical Corpus..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        if rag_chain:
            with st.spinner("Searching localized embeddings and synthesizing response..."):
                try:
                    response = rag_chain.invoke(user_query)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Orchestration Error: {e}")
        else:
            st.error("RAG Engine pipeline failed to compile.")