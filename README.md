##Arxelos: Distributed Medical Literature RAG Platform
An advanced, production-grade Retrieval-Augmented Generation (RAG) engine specializing in the synthesis and verification of neuro-oncology research literature. Built with modern LangChain primitives and engineered for optimized cloud resource allocation.

Live Platform: arxelos.com

##Key Architectural Innovations
LangChain Expression Language (LCEL): Implements a highly predictable, thread-safe, and traceable Directed Acyclic Graph (DAG) for data extraction and synthesis, avoiding legacy imperative control flows.

Headless Cloud Compute & On-Demand Memory Compilation: Dynamically builds and fragments the vector database directly within containerized RAM upon server initialization. This optimized design bypasses heavy binary pickle serialization, reducing repo footprint while keeping query performance sub-second.

State-of-the-Art LLM Layer: Integrated with gemini-2.5-flash for high-throughput, accurate, context-grounded response generation with a zero-token free-tier penalty.

Utility-First UI: Built using a ultra-minimalist, non-distracting Streamlit layout designed entirely for rapid data lookup and clean researcher workflow.

##Tech Stack
Orchestration: LangChain Core (LCEL), RunnableParallel, RunnablePassthrough

LLM Provider: Google GenAI (gemini-2.5-flash)

Embeddings Model: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)

Vector Database: LangChain InMemoryVectorStore

Frontend & Deployment: Streamlit Community Cloud

📁 ##Repository Structure
Plaintext
├── medical_corpus/          # Specialized neuro-oncology research papers (BioC JSON)
├── app.py                  # Production deployment engine & Streamlit UI logic
├── build_vector_db.py      # Core text-splitting and vectorization logic
├── orchestrate_rag.py      # Local LCEL pipeline testing harness
├── requirements.txt        # Production dependency manifest
└── .gitignore              # Ultra-lean environment configuration\

##Quick Start
1. Clone & Set Up Environment
PowerShell
git clone https://github.com/aryanp2107/MedicalRAG.git
cd MedicalRAG
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
2. Configure Credentials
Create a local .env file in the root directory:

Plaintext
GOOGLE_API_KEY=your_gemini_api_key_here
3. Launch Local Platform
PowerShell
streamlit run app.py
