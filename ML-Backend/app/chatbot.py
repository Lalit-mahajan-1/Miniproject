# app/chatbot.py

import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Generator, Any

# --- MODEL CONFIGURATION ---
# TinyLlama is a fantastic lightweight but powerful model.
# It's significantly better than Flan-T5 for Q&A.
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# BGE-Small is a top-tier embedding model for retrieval.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# --- GLOBAL VARIABLES ---
# To avoid reloading models on every request
llm = None
tokenizer = None
embeddings = None
vectorstore = None

def load_models():
    """Loads the LLM, Tokenizer, and Embedding models into memory."""
    global llm, tokenizer, embeddings

    if llm and tokenizer and embeddings:
        print("Models are already loaded.")
        return

    # --- Load Embedding Model ---
    print(f"🔄 Loading Embedding Model: {EMBED_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'}, # Embeddings are fine on CPU
        encode_kwargs={'normalize_embeddings': True} # Recommended for BGE
    )
    print("✅ Embedding model loaded successfully!")

    # --- Load LLM and Tokenizer ---
    print(f"🔄 Loading LLM: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16, # Use float16 for less memory
        device_map="auto",         # Automatically use GPU if available
    )
    print("✅ LLM and Tokenizer loaded successfully!")


def rebuild_rag_index(syllabus_rows: List[Dict[str, str]]):
    """Rebuilds the FAISS vector store from syllabus documents."""
    global vectorstore, embeddings
    
    if embeddings is None:
        raise RuntimeError("Embeddings model not loaded. Cannot build RAG index.")
        
    print("🔄 Rebuilding RAG index...")
    if not syllabus_rows:
        print("❗ No syllabus data provided. RAG index will be empty.")
        vectorstore = None
        return False

    all_docs = []
    # Split text into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for row in syllabus_rows:
        subject = row.get("name", "Unknown Subject")
        text = row.get("text", "")
        if not text.strip():
            continue
        
        # Create LangChain Document objects with metadata
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={"source": subject, "chunk": i}
            )
            all_docs.append(doc)

    if not all_docs:
        print("❗ RAG index NOT built: No text chunks found after processing.")
        vectorstore = None
        return False

    try:
        # Create the vector store from the documents and embeddings
        vectorstore = FAISS.from_documents(documents=all_docs, embedding=embeddings)
        print(f"✅ RAG Index built successfully with {len(all_docs)} chunks.")
        return True
    except Exception as e:
        print(f"❌ RAG Index build FAILED: {e}")
        vectorstore = None
        return False


def create_conversational_chain():
    """Creates the main conversational chain for Q&A."""
    global llm, tokenizer, vectorstore

    if not all([llm, tokenizer, vectorstore]):
        raise RuntimeError("Models and vectorstore must be loaded before creating a chain.")

    # A streamer is used to get the output token-by-token
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # HuggingFacePipeline integrates the local model with LangChain
    from langchain_huggingface import HuggingFacePipeline
    hf_pipeline = HuggingFacePipeline(
        pipeline=pipeline(
            "text-generation",
            model=llm,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.95,
            repetition_penalty=1.1,
            streamer=streamer,
        )
    )

    # Memory to remember past conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # This prompt template is critical. It's specifically for chat models.
    PROMPT_TEMPLATE = """
<|system|>
You are a helpful and friendly academic assistant. Use the syllabus context below to answer the user's question accurately.
- Provide answers based ONLY on the provided context.
- If the answer is not in the context, clearly state that the information is not available in the uploaded syllabuses. DO NOT use your general knowledge.
- Be concise and clear. Mention the source subject when possible.</s>
<|user|>
Context:
{context}

Question:
{question}</s>
<|assistant|>
"""
    
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

    # The final chain that connects the retriever, memory, and LLM
    chain = ConversationalRetrievalChain.from_llm(
        llm=hf_pipeline,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}), # Retrieve top 4 chunks
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    
    return chain, streamer


def process_user_query_stream(query: str) -> Generator[str, Any, None]:
    """
    Processes a user query and yields the response token by token (streaming).
    This is the main function called by the API.
    """
    if not vectorstore:
        yield "I'm sorry, no syllabus has been uploaded yet. Please upload a document first."
        return

    try:
        chain, streamer = create_conversational_chain()
        
        # Run the chain in a separate thread to allow for simultaneous generation and streaming
        def run_chain():
            chain.invoke({"question": query})

        thread = Thread(target=run_chain)
        thread.start()

        # Yield each new token as it is generated
        for token in streamer:
            yield token

        thread.join()

    except Exception as e:
        print(f"❌ Error during query processing: {e}")
        import traceback
        traceback.print_exc()
        yield "An error occurred while processing your request. Please check the logs."