import os
import re
import json
import asyncio
import httpx
from typing import List, Dict, Any, Generator

# -------------------- Configuration --------------------
MARKS_API_URL = "http://localhost:5000/api/marks/me"
LOCAL_API_BASE = "http://127.0.0.1:8000"

# -------------------- Robust Dependency Loading --------------------
# We wrap imports in try/except so the server NEVER crashes with 503
# even if you haven't installed the AI libraries.
HAS_ML = False
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    HAS_ML = True
except ImportError as e:
    print(f"⚠️ [Chatbot] ML libraries not found: {e}")
    print("⚠️ [Chatbot] Running in KEYWORD SEARCH mode (No AI embeddings).")

class SyllabusChatbot:
    def __init__(self):
        self.model = None
        self.index = None
        self.chunks = []
        self.is_ready = False

    def initialize(self):
        """Load AI models if available."""
        if HAS_ML:
            try:
                print("⏳ [Chatbot] Loading embedding model...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.is_ready = True
                print("✅ [Chatbot] AI Model loaded successfully.")
            except Exception as e:
                print(f"❌ [Chatbot] Model load failed: {e}")
                self.is_ready = False
        else:
            print("ℹ️ [Chatbot] Skipping model load (ML deps missing).")

    def build_index(self, syllabus_data: List[Dict[str, str]]):
        """Ingest syllabus text into memory."""
        self.chunks = []
        
        # 1. Text Chunking
        print(f"🔄 [Chatbot] Indexing {len(syllabus_data)} subjects...")
        for item in syllabus_data:
            subject = item.get("name", "General")
            text = item.get("text", "")
            
            # Split by double newlines to get paragraphs
            raw_parts = text.split("\n\n")
            for part in raw_parts:
                clean_part = part.strip()
                if len(clean_part) > 30:  # Skip very short noise
                    # Store as: "Subject Name: The actual text content"
                    self.chunks.append(f"**{subject}**: {clean_part}")

        if not self.chunks:
            print("⚠️ [Chatbot] No text found to index.")
            return

        # 2. Vector Embedding (only if ML is enabled)
        if self.is_ready and HAS_ML:
            try:
                embeddings = self.model.encode(self.chunks)
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(np.array(embeddings).astype('float32'))
                print(f"✅ [Chatbot] Vector index built with {len(self.chunks)} chunks.")
            except Exception as e:
                print(f"❌ [Chatbot] Index build failed: {e}")
                self.index = None
        else:
            print(f"✅ [Chatbot] Keyword index ready ({len(self.chunks)} chunks).")

    def search(self, query: str, top_k=3) -> List[str]:
        """Find relevant info using AI or Keywords."""
        if not self.chunks:
            return []

        # A. AI Vector Search
        if self.is_ready and self.index is not None and HAS_ML:
            try:
                q_vec = self.model.encode([query])
                D, I = self.index.search(np.array(q_vec).astype('float32'), top_k)
                results = []
                for idx in I[0]:
                    if 0 <= idx < len(self.chunks):
                        results.append(self.chunks[idx])
                return results
            except Exception as e:
                print(f"⚠️ Search error: {e}")
        
        # B. Fallback: Keyword Search
        query_lower = query.lower()
        # Simple ranking: count how many query words appear in the chunk
        query_words = [w for w in query_lower.split() if len(w) > 3]
        
        scores = []
        for chunk in self.chunks:
            chunk_lower = chunk.lower()
            score = 0
            if query_lower in chunk_lower: 
                score += 10 # Exact phrase match
            for word in query_words:
                if word in chunk_lower:
                    score += 1
            if score > 0:
                scores.append((score, chunk))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scores[:top_k]]

# Global Instance
bot = SyllabusChatbot()

# -------------------- API Helpers --------------------

async def fetch_marks():
    """Get marks from localhost:5000"""
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(MARKS_API_URL)
            if resp.status_code == 200:
                data = resp.json()
                # Format data nicely
                msg = "📊 **Your Academic Performance:**\n"
                for subject, score in data.items():
                    msg += f"- {subject}: {score}\n"
                return msg
            return f"⚠️ Could not fetch marks (Status {resp.status_code})."
    except Exception:
        return "⚠️ Could not connect to the Student Database (Port 5000). Is it running?"

async def fetch_topics(subject_name):
    """Get topics from localhost:8000"""
    url = f"{LOCAL_API_BASE}/syllabus/topics/{subject_name}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                modules = data.get("modules", [])
                if not modules:
                    return f"I found the subject **{data.get('subject')}**, but the syllabus parsing didn't find clear modules."
                
                msg = f"📘 **Syllabus for {data.get('subject')}:**\n\n"
                for mod in modules:
                    title = mod.get('title', 'Unit')
                    num = mod.get('module_no', '')
                    msg += f"**Module {num}: {title}**\n"
                    # List first 2 units to keep it brief
                    units = mod.get('units', [])
                    for u in units[:3]:
                        msg += f"  • {u.get('content')}\n"
                    if len(units) > 3:
                        msg += "  • ...and more\n"
                    msg += "\n"
                return msg
            elif resp.status_code == 404:
                return f"❌ I couldn't find a subject named '{subject_name}' in your uploads."
            else:
                return "⚠️ Error retrieving syllabus structure."
    except Exception as e:
        return f"⚠️ Internal Error: {str(e)}"

# -------------------- Mental Health Logic --------------------

def check_mental_health(query):
    q = query.lower()
    responses = {
        "suicid": "🚨 **Important:** You are not alone. If you are in danger, please call a local emergency number or a suicide prevention hotline immediately. Your life has value.",
        "kill myself": "🚨 **Please stop.** Reach out to a friend, family member, or a professional right now. There is help available.",
        "depress": "💙 I'm sorry you're feeling this way. Depression is tough, but you don't have to fight it alone. Have you considered talking to a campus counselor?",
        "anxiety": "🌬️ **Breathe.** Try the 4-7-8 technique: Inhale for 4s, hold for 7s, exhale for 8s. Focus on right now, not the future.",
        "panic": "🛑 It's going to be okay. Look around you. Name 5 things you see, 4 you feel, 3 you hear. Ground yourself in the present.",
        "stress": "📚 Exams can be stressful. Remember to take breaks. A 10-minute walk can do wonders for your brain. You've got this!",
        "tired": "💤 Rest is part of studying. If you are exhausted, your brain won't retain info. Go get some sleep!",
        "sad": "💙 It's okay to have down days. Be kind to yourself today."
    }
    for key, resp in responses.items():
        if key in q:
            return resp
    return None

def check_small_talk(query):
    q = query.lower().strip()
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if q in greetings:
        return "👋 Hello! I'm your Study Assistant. I can help with your **Syllabus**, **Marks**, or **Mental Health**. What do you need?"
    if "who are you" in q:
        return "🤖 I am an AI bot designed to help students navigate their coursework and academic stress."
    return None

# -------------------- Main Hooks (Called by main.py) --------------------

def init_chatbot():
    """Called by main.py on startup"""
    bot.initialize()

def rebuild_rag_index(data):
    """Called by main.py when files change"""
    bot.build_index(data)

def process_user_query(query: str, use_rag: bool = True) -> str:
    """
    The main entry point.
    Because main.py calls this synchronously, we wrap async logic here.
    """
    # We run the async logic inside a synchronous wrapper
    return asyncio.run(_async_process(query, use_rag))

async def _async_process(query: str, use_rag: bool) -> str:
    q_lower = query.lower()

    # 1. Mental Health Check (Highest Priority)
    mh_msg = check_mental_health(query)
    if mh_msg:
        return mh_msg

    # 2. Small Talk Check
    st_msg = check_small_talk(query)
    if st_msg:
        return st_msg

    # 3. Check for Marks request
    if any(x in q_lower for x in ["my marks", "my grades", "my score", "result", "cgpa"]):
        return await fetch_marks()

    # 4. Check for Syllabus Structure request (Regex)
    # Matches: "modules of Java", "syllabus for Python", "topics in Chemistry"
    match = re.search(r"(?:modules|syllabus|topics|chapters)\s+(?:of|for|in)\s+(.+)", q_lower)
    if match:
        subject = match.group(1).replace("?", "").strip()
        return await fetch_topics(subject)

    # 5. RAG / Knowledge Base Search
    # If the user specifically wants syllabus info (or if general toggle is on)
    if use_rag:
        results = bot.search(query)
        if results:
            # Join the chunks into a response
            response = "Here is what I found in your documents:\n\n"
            response += "\n\n---\n".join(results)
            return response
        else:
            return "🤷 I searched your uploaded syllabus files but couldn't find a relevant answer. Try checking the exact subject name or upload more documents."

    # 6. Fallback
    return "I'm not sure about that. Try asking about your **marks**, a specific **subject syllabus**, or toggle **Syllabus Mode** to search your documents."

def process_user_query_stream(query: str, use_rag: bool = True) -> Generator[str, None, None]:
    """Simulates streaming for the frontend"""
    response = process_user_query(query, use_rag)
    words = response.split(" ")
    for word in words:
        yield word + " "
        # Tiny delay to simulate thinking/typing
        import time
        time.sleep(0.01)