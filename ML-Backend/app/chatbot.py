# app/chatbot.py

import os
import torch
import threading
from typing import List, Dict, Generator

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

# Optional 4-bit quant for GPU; fall back gracefully if not installed
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None  # type: ignore
    _HAS_BNB = False

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# -------------------- Config --------------------
DEFAULT_GPU_MODEL = os.getenv("LLM_MODEL_GPU", "Qwen/Qwen2.5-1.5B-Instruct")
DEFAULT_CPU_MODEL = os.getenv("LLM_MODEL_CPU", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# -------------------- Globals --------------------
tokenizer = None
llm = None
embeddings = None
vectorstore = None

# -------------------- Loader --------------------
def load_models():
    global tokenizer, llm, embeddings

    # Load embeddings (CPU is fine)
    if embeddings is None:
        print(f"🔄 Loading Embedding Model: {EMBED_MODEL}...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ Embeddings loaded")

    if tokenizer is not None and llm is not None:
        return

    has_cuda = torch.cuda.is_available()
    model_name = DEFAULT_GPU_MODEL if has_cuda else DEFAULT_CPU_MODEL
    print(f"🔄 Loading LLM: {model_name} (cuda={has_cuda})...")

    if has_cuda and _HAS_BNB:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer_local = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        llm_local = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
        )
    elif has_cuda and not _HAS_BNB:
        print("⚠️ bitsandbytes not installed; loading in float16 (may use more VRAM).")
        tokenizer_local = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        llm_local = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        # CPU fallback
        tokenizer_local = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # new transformers shows deprecation warning for torch_dtype; it still works.
        llm_local = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
        )

    # Assign globals after everything is loaded
    tokenizer = tokenizer_local
    llm = llm_local

    # Ensure pad/eos tokens exist
    if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ LLM loaded")

def init_chatbot():
    """Call this on startup to ensure models & embeddings are ready."""
    load_models()

# -------------------- RAG Index --------------------
def rebuild_rag_index(syllabus_rows: List[Dict[str, str]]):
    global vectorstore, embeddings
    if embeddings is None:
        raise RuntimeError("Embeddings not loaded. Call load_models() first.")

    print("🔄 Rebuilding RAG index...")
    if not syllabus_rows:
        print("❗ No syllabus data provided. RAG index will be empty.")
        vectorstore = None
        return False

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
    docs = []

    for row in syllabus_rows:
        subject = (row.get("name") or "Unknown Subject").strip()
        text = (row.get("text") or "").strip()
        if not text:
            continue
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(page_content=chunk, metadata={"source": subject, "chunk": i})
            )

    if not docs:
        print("❗ No text chunks found after processing.")
        vectorstore = None
        return False

    try:
        vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
        print(f"✅ RAG index built with {len(docs)} chunks.")
        return True
    except Exception as e:
        print(f"❌ Failed to build RAG index: {e}")
        vectorstore = None
        return False

# -------------------- Prompt helpers --------------------
RAG_SYSTEM_PROMPT = (
    "You are a helpful academic assistant. Answer ONLY from the given context. "
    "If the answer isn't present, say it isn't available in the uploaded syllabuses. "
    "Be concise and mention the source subject when possible."
)

def _build_rag_prompt(question: str, context_blocks: List[Document]) -> str:
    context_text = []
    for d in context_blocks:
        src = d.metadata.get("source", "Unknown Subject")
        context_text.append(f"[{src}] {d.page_content}")
    context_joined = "\n\n".join(context_text[:8])

    user_content = f"Context:\n{context_joined}\n\nQuestion:\n{question}\n\nAnswer:"
    try:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = f"<SYS>{RAG_SYSTEM_PROMPT}</SYS>\nUser: {user_content}\nAssistant:"
    return prompt

def _build_chat_prompt(query: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a concise, helpful assistant."},
                {"role": "user", "content": query},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"Assistant, answer concisely:\n\nUser: {query}\nAssistant:"

# -------------------- Generation --------------------
def _generate_text(prompt: str) -> str:
    # non-streaming generate
    device = next(llm.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out_ids = llm.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    # Try to strip the prompt if it's included
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    return text.strip()

def _generate_stream(prompt: str) -> Generator[str, None, None]:
    device = next(llm.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
    )
    thread = threading.Thread(target=llm.generate, kwargs=gen_kwargs)
    thread.start()
    for token in streamer:
        yield token
    thread.join()

# -------------------- Public API --------------------
def process_user_query(query: str, use_rag: bool = True) -> str:
    if not query or not query.strip():
        return "Query cannot be empty."

    if tokenizer is None or llm is None or embeddings is None:
        load_models()

    if use_rag:
        if vectorstore is None:
            return "No syllabus has been uploaded yet or the index is empty. Please upload and rebuild the index."
        try:
            docs = vectorstore.similarity_search(query, k=4)
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return "There was an error retrieving context. Please try again."
        prompt = _build_rag_prompt(query, docs)
        try:
            return _generate_text(prompt)
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return "An error occurred while generating the response."
    else:
        prompt = _build_chat_prompt(query)
        try:
            return _generate_text(prompt)
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return "An error occurred while generating the response."

def process_user_query_stream(query: str, use_rag: bool = True) -> Generator[str, None, None]:
    if tokenizer is None or llm is None or embeddings is None:
        load_models()

    if use_rag:
        if vectorstore is None:
            yield "No syllabus has been uploaded yet or the index is empty. Please upload and rebuild the index."
            return
        try:
            docs = vectorstore.similarity_search(query, k=4)
        except Exception as e:
            yield f"[Retrieval Error] {e}"
            return
        prompt = _build_rag_prompt(query, docs)
        yield from _generate_stream(prompt)
    else:
        prompt = _build_chat_prompt(query)
        yield from _generate_stream(prompt)