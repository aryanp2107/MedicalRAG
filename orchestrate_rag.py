import os
import pickle
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

# Modern Core Orchestration Modules
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableSerializable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableSerializable

class MedicalRAGPipeline:
    def __init__(self, vector_store_path: str):
        self.vector_store_path = vector_store_path
        
        # 1. Try to get it from the system environment first
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        # 2. EMERGENCY FALLBACK: If PowerShell fails to pass it, hardcode it here for the demo
        if not self.api_key or self.api_key == "AIzaSyCFh6cI2ozdYJMmvp5ULHGbVQZaS6Yu49U":
            self.api_key = "AIzaSyCFh6cI2ozdYJMmvp5ULHGbVQZaS6Yu49U"  # <--- PASTE YOUR GEMINI KEY DIRECTLY HERE
            
        if not self.api_key or self.api_key.startswith("AIzaSyYourActual"):
            raise ValueError(
                "GOOGLE_API_KEY is missing. Please paste your actual Gemini API key "
                "from Google AI Studio into line 20 of orchestrate_rag.py"
            )
            
        self.retriever = self._initialize_retriever()
        self.llm = self._initialize_llm()
        self.prompt = self._initialize_prompt()
        
        
    def _initialize_retriever(self):
        with open(self.vector_store_path, "rb") as f:
            vector_store = pickle.load(f)
        # Orchestration layer expects a Runnable; as_retriever() implements the Runnable protocol
        return vector_store.as_retriever(search_kwargs={"k": 3})
        
    def _initialize_llm(self):
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=self.api_key, 
            temperature=0.1
        )
        
    def _initialize_prompt(self) -> ChatPromptTemplate:
        template = """You are an expert neuro-oncology AI engine. Synthesize an answer using the provided literature chunks.
        If the context doesn't contain the answer, state that explicitly.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        return ChatPromptTemplate.from_template(template)
        
    def _format_docs(self, docs) -> str:
        """Helper component to transform raw documents into a single text block."""
        return "\n\n".join(doc.page_content for doc in docs)

    def compile_chain(self) -> RunnableSerializable:
        """
        Orchestration Graph Execution Layer
        Using LCEL to build a deterministic, traceable DAG (Directed Acyclic Graph)
        """
        # 1. Parallel execution branch: fetch context and pass query simultaneously
        data_preparation = RunnableParallel({
            "context": self.retriever | self._format_docs,
            "question": RunnablePassthrough()
        })
        
        # 2. Linear execution pipeline
        orchestrated_chain = data_preparation | self.prompt | self.llm | StrOutputParser()
        
        return orchestrated_chain

# --- Execution Driver ---
if __name__ == "__main__":
    # Initialize the orchestrated pipeline
    pipeline = MedicalRAGPipeline(vector_store_path="vector_store.pkl")
    rag_chain = pipeline.compile_chain()
    
    # Run a test query
    query = "What specific deep learning models are mentioned for brain tumor classification?"
    print(f"\nSearching and Orchestrating Answer for: '{query}'\n")
    
    # Invoke the graph
    response = rag_chain.invoke(query)
    print(response)