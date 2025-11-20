from dotenv import load_dotenv
load_dotenv()

from fastapi import (
    FastAPI, HTTPException, UploadFile, File, Form,
    BackgroundTasks, APIRouter
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from app.gemini_client import ask_gemini
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4
import sqlite3
import io
import re
import os
import csv
import json

from app.syllabus_context import build_context



# Optional OCR deps (don’t block CSV upload if missing)
try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
except Exception:
    pytesseract = None  # type: ignore
    Image = None  # type: ignore

try:
    import fitz  # PyMuPDF  # type: ignore
except Exception:
    fitz = None  # type: ignore

app = FastAPI(title="Syllabus OCR + Chatbot API")

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Paths & Directories --------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "syllabus"
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "db.sqlite3"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Tesseract configuration --------------------
def setup_tesseract_cmd():
    if pytesseract is None:
        print("ℹ️ pytesseract not installed. OCR endpoints will be limited.")
        return

    env_path = os.getenv("TESSERACT_CMD")
    if env_path and Path(env_path).exists():
        pytesseract.pytesseract.tesseract_cmd = env_path
        return

    win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if Path(win_path).exists():
        pytesseract.pytesseract.tesseract_cmd = win_path
        return

    for p in ["/usr/bin/tesseract", "/usr/local/bin/tesseract", "/opt/homebrew/bin/tesseract"]:
        if Path(p).exists():
            pytesseract.pytesseract.tesseract_cmd = p
            return

    print("⚠️ Tesseract path not explicitly set. Ensure tesseract is in PATH or set TESSERACT_CMD env var.")

setup_tesseract_cmd()

# -------------------- DB helpers --------------------
def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS syllabus (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                file_name TEXT,
                file_path TEXT,
                outline_json TEXT
            );
            """
        )
        try:
            cols = [r[1] for r in con.execute("PRAGMA table_info('syllabus')").fetchall()]
            if "outline_json" not in cols:
                con.execute("ALTER TABLE syllabus ADD COLUMN outline_json TEXT")
        except Exception:
            pass
        con.commit()

def save_subject(name: str, text: str, file_name: Optional[str], file_bytes: Optional[bytes]) -> str:
    sid = str(uuid4())
    created_at = datetime.utcnow().isoformat()

    saved_path = None
    if file_name and file_bytes:
        ext = Path(file_name).suffix
        saved_path = UPLOADS_DIR / f"{sid}{ext}"
        saved_path.write_bytes(file_bytes)

    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO syllabus (id, name, text, created_at, file_name, file_path, outline_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sid, name, text, created_at, file_name or None, str(saved_path) if saved_path else None, None),
        )
        con.commit()

    return sid

def update_outline(subject_id: str, outline: Dict[str, Any]):
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "UPDATE syllabus SET outline_json = ? WHERE id = ?",
            (json.dumps(outline, ensure_ascii=False), subject_id),
        )
        con.commit()

def list_subjects():
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT id, name, created_at FROM syllabus ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

def get_subject(subject_id: str):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT id, name, text, created_at, outline_json FROM syllabus WHERE id = ?",
            (subject_id,)
        ).fetchone()
        return dict(row) if row else None

def get_subject_by_name(subject_name: str):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT id, name, text, created_at, outline_json FROM syllabus WHERE LOWER(name) = LOWER(?)",
            (subject_name.strip(),)
        ).fetchone()
        return dict(row) if row else None

def get_all_syllabus_data() -> List[Dict[str, str]]:
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT name, text FROM syllabus").fetchall()
        return [dict(r) for r in rows]

# -------------------- RAG Indexing (Background) --------------------
def run_rag_rebuild():
    print("Starting synchronous RAG index rebuild...")
    try:
        syllabus_data = get_all_syllabus_data()
        if not syllabus_data:
            print("No syllabus data found in DB. RAG index will be empty.")
        # Lazy import chatbot and rebuild
        try:
            from app import chatbot as cb
            cb.rebuild_rag_index(syllabus_data)
        except Exception as e:
            print(f"❌ RAG rebuild skipped (chatbot deps missing or error): {e}")
    except Exception as e:
        print(f"❌ Error during RAG index rebuild: {e}")

def rebuild_index_background(background_tasks: BackgroundTasks):
    print("Queueing RAG index rebuild in background...")
    background_tasks.add_task(run_rag_rebuild)

# -------------------- OCR helpers --------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def ocr_image_bytes(img_bytes: bytes) -> str:
    if pytesseract is None or Image is None:
        raise HTTPException(status_code=503, detail="OCR dependencies not installed on server.")
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    out = pytesseract.image_to_string(img, lang="eng")
    return clean_text(out)

def pdf_to_images_bytes(pdf_bytes: bytes, zoom: float = 2.0) -> List[bytes]:
    if fitz is None:
        raise HTTPException(status_code=503, detail="PDF OCR dependencies not installed on server.")
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            images.append(pix.tobytes("png"))
    return images

def ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    if fitz is None:
        raise HTTPException(status_code=503, detail="PDF OCR dependencies not installed on server.")
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            extracted = "\n".join(page.get_text("text") for page in doc)
            if len(extracted.strip()) > 500:
                return clean_text(extracted)
    except Exception:
        pass

    text_chunks = []
    page_images = pdf_to_images_bytes(pdf_bytes, zoom=2.0)
    for ib in page_images:
        text_chunks.append(ocr_image_bytes(ib))
    return clean_text("\n\n".join(text_chunks))

# -------------------- CSV helpers for prediction --------------------
def norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (k or "").lower())

def get_val(row: Dict[str, Any], *candidates: str):
    idx = {norm_key(k): k for k in row.keys()}
    for cand in candidates:
        nk = norm_key(cand)
        if nk in idx:
            return row[idx[nk]]
    return None

def to_float(v, default=None):
    if v is None:
        return default
    s = str(v).strip()
    if s == "":
        return default
    try:
        return float(s)
    except Exception:
        return default

def clamp(v, lo=0.0, hi=100.0):
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return lo

def scale_0_100(value, vmin, vmax):
    v = to_float(value, None)
    if v is None:
        return None
    if vmax == vmin:
        return 0.0
    return clamp(100.0 * (v - vmin) / (vmax - vmin))

def map_level(s: str, mapping: Dict[str, float], default=50.0):
    if s is None:
        return default
    key = str(s).strip().lower()
    return mapping.get(key, default)

def socio_from_education(level: Optional[str]) -> float:
    mapping = {
        "school": 40.0,
        "highschool": 50.0,
        "college": 60.0,
        "graduate": 75.0,
        "postgraduate": 85.0,
        "masters": 85.0,
        "phd": 90.0,
    }
    return map_level(level, mapping, default=60.0)

def socio_from_income(level: Optional[str]) -> float:
    s = str(level or "").strip().lower()
    if ">100k" in s or "above" in s or "100k+" in s:
        return 75.0
    if "50k-100k" in s or "50k to 100k" in s:
        return 65.0
    if "0-50k" in s or "<50k" in s or "below" in s:
        return 55.0
    return 60.0

def compute_risk(row: Dict[str, Any]) -> float:
    prev_grade = to_float(get_val(row, "Previous_Semester_Grades"), None)
    current_score = to_float(get_val(row, "Current_Internal_Score"), None)
    assign_rate = to_float(get_val(row, "Assignment_Completion_Rate"), None)
    project = to_float(get_val(row, "Project_Score"), None)
    study_hours = to_float(get_val(row, "Study_Hours_Per_Week"), None)
    attend = to_float(get_val(row, "Attendance_Percentage"), None)

    lec_part = get_val(row, "Lecture_Participation")
    extra_part = get_val(row, "Extracurricular_Participation")
    teacher_fb = to_float(get_val(row, "Teacher_Feedback_Score"), None)
    parent_edu = get_val(row, "Parental_Education_Level")
    income = get_val(row, "Family_Income")
    parent_support = to_float(get_val(row, "Parental_Support_Score"), None)
    stress = to_float(get_val(row, "Stress_Level"), None)
    motivation = to_float(get_val(row, "Motivation_Score"), None)
    health_issues = to_float(get_val(row, "Health_Issues"), 0)
    engagement = to_float(get_val(row, "Online_Engagement_Score"), None)
    delay_days = to_float(get_val(row, "Assignment_Submission_Delay"), 0)
    ai_tools = to_float(get_val(row, "Use_of_AI_or_Tutoring_Tools"), None)
    disciplinary = to_float(get_val(row, "Disciplinary_Actions"), 0)

    study_norm = clamp(scale_0_100(study_hours, 0, 25)) if study_hours is not None else None
    teacher_norm = clamp((teacher_fb or 0) * 10, 0, 100) if teacher_fb is not None else None
    parent_support_norm = clamp((parent_support or 0) * 10, 0, 100) if parent_support is not None else None
    motivation_norm = clamp((motivation or 0) * 10, 0, 100) if motivation is not None else None
    lec_norm = map_level(lec_part, {"low": 30, "medium": 60, "high": 85}, 50)
    extra_norm = map_level(extra_part, {"low": 30, "medium": 55, "high": 75}, 50)
    socio = 0.5 * socio_from_education(parent_edu) + 0.5 * socio_from_income(income)
    ai_norm = 50.0
    if ai_tools is not None:
        if ai_tools <= 0:
            ai_norm = 45.0
        elif ai_tools <= 3:
            ai_norm = 60.0
        else:
            ai_norm = 58.0

    parts = []
    def add(weight, value_or_none):
        if value_or_none is None:
            parts.append((weight, 50.0))
        else:
            parts.append((weight, clamp(value_or_none, 0, 100)))

    add(0.10, prev_grade)
    add(0.14, current_score)
    add(0.08, assign_rate)
    add(0.06, project)
    add(0.05, study_norm)
    add(0.20, attend)
    add(0.05, lec_norm)
    add(0.02, extra_norm)
    add(0.08, teacher_norm)
    add(0.04, parent_support_norm)
    add(0.08, motivation_norm)
    add(0.05, engagement)
    add(0.02, ai_norm)
    add(0.03, socio)

    protective = sum(w * v for w, v in parts)

    penalty = 0.0
    penalty += clamp((stress or 0) * 2.0, 0, 20)
    penalty += clamp(delay_days * 2.0, 0, 20)
    penalty += 10.0 if (health_issues or 0) > 0 else 0.0
    penalty += clamp((disciplinary or 0) * 8.0, 0, 24)

    risk = clamp(100.0 - protective + penalty, 0.0, 100.0)
    return round(risk, 1)

# -------------------- Chapter/Topic extraction --------------------
chapter_header = re.compile(
    r"^\s*(?:chapter|unit|module|part|section)\s*(\d+)?\s*[:.)-]*\s*(.+)$",
    flags=re.I
)
week_header = re.compile(
    r"^\s*week\s*(\d+)\s*[:.)-]*\s*(.+)?$",
    flags=re.I
)
topic_line = re.compile(
    r"^\s*(?:[-•*]+|\d+\)|\d+\.\s+|[a-zA-Z]\))\s*(.+?)\s*$"
)

def extract_outline(text: str) -> Dict[str, Any]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chapters: List[Dict[str, Any]] = []
    current = None

    for ln in lines:
        m1 = chapter_header.match(ln)
        m2 = week_header.match(ln)

        if m1:
            num, title = m1.group(1), m1.group(2) or ""
            title = title.strip()
            label = f"Chapter {num}: {title}" if num else f"Chapter: {title}"
            current = {"title": label, "topics": []}
            chapters.append(current)
            continue

        if m2:
            num, title = m2.group(1), (m2.group(2) or "").strip()
            label = f"Week {num}" + (f": {title}" if title else "")
            current = {"title": label, "topics": []}
            chapters.append(current)
            continue

        mt = topic_line.match(ln)
        if mt and current:
            topic = mt.group(1).strip()
            if topic and len(topic) > 2:
                current["topics"].append(topic)
            continue

        if current and 3 <= len(ln) <= 120 and not ln.lower().startswith(("chapter", "unit", "module", "part", "section", "week")):
            if not re.search(r"(syllabus|objective|outcome|policy|grading|assessment)", ln, re.I):
                current["topics"].append(ln)

    if not chapters:
        topics = []
        for ln in lines:
            mt = topic_line.match(ln)
            if mt:
                topics.append(mt.group(1).strip())
        if topics:
            chapters = [{"title": "Syllabus", "topics": topics}]

    chapters = [c for c in chapters if c.get("title") and isinstance(c.get("topics"), list)]
    return {"chapters": chapters}

# -------------------- Chatbot Router (lazy import) --------------------


# harshit chatgpt code
@app.post("/chat/gemini")
async def gemini_chat_smart(query: str = Form(...)):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    print(f"📩 Smart Query: {query}")

    # Build dynamic syllabus context
    context = build_context()

    # Construct final prompt for Gemini
    prompt = f"""
You are an academic assistant for students.

Here is the entire syllabus data of the system. Use it to answer:

-------------------
CONTEXT START
{context}
-------------------
CONTEXT END

User question: {query}

Rules:
- Give answers ONLY using the syllabus context.
- If the question is about modules/units/books, extract correct info.
- If not found, say 'This topic is not in the syllabus.'
"""
    # prompt = query
    print("promt",prompt)
    reply = ask_gemini(prompt)

    return {
        "success": True,
        "response": reply
    }





chat_router = APIRouter(prefix="/chat", tags=["Chatbot"])

@chat_router.post("/")
@chat_router.post("")
async def handle_chat_query(
    query: str = Form(...),
    use_rag: bool = Form(True)
):
    print(f"📩 Received chat query: '{query}' | use_rag={use_rag}")
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        try:
            from app import chatbot as cb
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Chatbot unavailable: {e}")
        response = cb.process_user_query(query.strip(), use_rag)
        print(f"✅ Response generated: {response[:120]}...")
        return {"response": response, "success": True}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat query: {str(e)}")

@chat_router.post("/stream")
async def handle_chat_stream(
    query: str = Form(...),
    use_rag: bool = Form(True)
):
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    def sse():
        try:
            from app import chatbot as cb
            for token in cb.process_user_query_stream(query.strip(), use_rag=use_rag):
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
        finally:
            yield "event: done\ndata: [END]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")

app.include_router(chat_router)

# -------------------- Routes --------------------
@app.get("/")
async def root():
    return {"message": "Syllabus OCR + Chatbot API is running"}

@app.get("/health")
async def health():
    return {"ok": True}

# Accept both /uploadfile and /uploadfile/ to avoid redirect issues
@app.post("/uploadfile")
@app.post("/uploadfile/")
async def upload_csv(file_upload: UploadFile = File(...)):
    fname = (file_upload.filename or "").lower()
    if not fname.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    raw = await file_upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except Exception:
            raise HTTPException(status_code=400, detail="Could not decode CSV. Use UTF-8 or UTF-8 with BOM.")

    reader = csv.DictReader(io.StringIO(text))
    rows: List[Dict[str, Any]] = []
    for row in reader:
        if any((v or "").strip() for v in row.values()):
            rows.append(row)

    if not rows:
        return {"data": []}

    results = []
    for r in rows:
        orig_pred = to_float(get_val(r, "predicted_risk_percentage"), None)
        new_risk = compute_risk(r)
        r["predicted_risk_percentage"] = new_risk
        if orig_pred is not None:
            r["_original_predicted_risk_percentage"] = orig_pred
        results.append(r)

    return {"data": results}

@app.post("/ocr/syllabus")
async def upload_syllabus(
    background_tasks: BackgroundTasks,
    subject_name: str = Form(...),
    file: UploadFile = File(...)
):
    if not subject_name or len(subject_name.strip()) < 2:
        raise HTTPException(status_code=400, detail="Subject name is required.")

    filename = file.filename or ""
    name_lower = filename.lower()
    if not re.search(r"\.(pdf|png|jpg|jpeg|webp)$", name_lower):
        raise HTTPException(status_code=400, detail="Invalid file format. Use PDF or image (png/jpg/jpeg/webp).")

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        if name_lower.endswith(".pdf"):
            extracted_text = ocr_pdf_bytes(file_bytes)
        else:
            extracted_text = ocr_image_bytes(file_bytes)

        if not extracted_text or len(extracted_text.strip()) == 0:
            raise HTTPException(status_code=422, detail="OCR yielded no text. Try a clearer scan.")

        subject_id = save_subject(subject_name.strip(), extracted_text, filename, file_bytes)

        outline = extract_outline(extracted_text)
        try:
            update_outline(subject_id, outline)
        except Exception as e:
            print(f"⚠️ Could not save outline JSON: {e}")

        rebuild_index_background(background_tasks)

        return {"subject_id": subject_id, "text": extracted_text, "name": subject_name.strip(), "outline": outline}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ OCR/Storage error: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing error: {str(e)}")

@app.get("/ocr/syllabus")
async def get_syllabus_list():
    try:
        items = list_subjects()
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ocr/syllabus/{subject_id}")
async def get_syllabus(subject_id: str):
    s = get_subject(subject_id)
    if not s:
        raise HTTPException(status_code=404, detail="Subject not found")
    return {"id": s["id"], "name": s["name"], "text": s["text"], "created_at": s["created_at"], "outline": json.loads(s["outline_json"] or '{"chapters": []}')}


@app.get("/syllabus/topics/{subject_name}")
async def get_topics_by_subject(subject_name: str):
    """
    Returns structured syllabus for a subject:

    {
      "subject": "...",
      "modules": [ ... ],
      "textbooks": [ ... ],
      "reference_books": [ ... ]
    }
    """
    row = get_subject_by_name(subject_name)
    if not row:
        raise HTTPException(status_code=404, detail="Subject not found")

    text = row["text"] or ""

    # --------- helpers local to this endpoint ---------
    def parse_modules_and_units(txt: str):
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        modules = []
        current_module = None
        current_unit = None
        in_theory = False

        for line in lines:
            low = line.lower()

            # Enter theory component section
            if not in_theory:
                if "theory component" in low:
                    in_theory = True
                continue

            # Stop parsing modules/units when lab or books section starts
            if re.search(r"\blaboratory component\b", low):
                break
            if low.startswith("text books") or low.startswith("textbooks"):
                break
            if low.startswith("reference books"):
                break

            # Skip table header lines like "Module | Unit No. No."
            if "module" in low and "unit" in low and "no" in low:
                continue
            if low.startswith("topics ref"):
                continue
            if low.startswith("total "):
                continue

            # Module line: "1 Title | Introduction to Distributed Systems"
            m_mod = re.match(r"^(\d+)\s+Title\b[:|]?\s*(.+)$", line, re.I)
            if m_mod:
                module_no = int(m_mod.group(1))
                title = m_mod.group(2).strip().rstrip(".")
                current_module = {
                    "module_no": module_no,
                    "title": title,
                    "units": [],
                }
                modules.append(current_module)
                current_unit = None
                continue

            # Special case: Self Study line like "6 Self | Cloud Computing..."
            m_self = re.match(r"^(\d+)\s+Self\b.*", line, re.I)
            if m_self:
                module_no = int(m_self.group(1))
                current_module = {
                    "module_no": module_no,
                    "title": "Self Study",
                    "units": [],
                }
                modules.append(current_module)
                current_unit = None

                # Treat same line content after '|' as a unit 6.1
                if "|" in line:
                    after_pipe = line.split("|", 1)[1].strip()
                    after_pipe = after_pipe.split("|", 1)[0].strip()
                    if after_pipe:
                        current_unit = {
                            "unit_no": f"{module_no}.1",
                            "content": after_pipe,
                        }
                        current_module["units"].append(current_unit)
                continue

            # Unit lines
            if current_module:
                # pattern with pipe: "1.1 | Definition ..."
                m_unit = re.match(r"^(\d+(?:\.\d+)?)\s*\|\s*(.+)$", line)
                if not m_unit:
                    # fallback without pipe: "3.2 Clock Synchronization: ..."
                    m_unit = re.match(r"^(\d+(?:\.\d+)?)\s+(.+)$", line)
                if m_unit:
                    num_str = m_unit.group(1).strip()
                    rest = m_unit.group(2).strip()

                    # ignore if this actually looks like a module header
                    if rest.lower().startswith("title"):
                        continue

                    # Fix common OCR like "41" instead of "4.1"
                    unit_no = num_str
                    if (
                        "." not in num_str
                        and len(num_str) == 2
                        and str(current_module["module_no"]) == num_str[0]
                    ):
                        unit_no = f"{num_str[0]}.{num_str[1]}"

                    # Remove everything after next '|' (reference / hours columns)
                    rest = rest.split("|", 1)[0].strip()

                    if rest:
                        current_unit = {
                            "unit_no": unit_no,
                            "content": rest,
                        }
                        current_module["units"].append(current_unit)
                    else:
                        current_unit = None
                    continue

            # Continuation lines for current unit
            if current_module and current_unit:
                # Avoid headers we've not caught
                if re.match(r"^\d+\s+Title\b", line, re.I):
                    continue
                if line.lower().startswith(
                    ("text books", "reference books", "laboratory component")
                ):
                    continue

                cont = line.split("|", 1)[0].strip()
                if not cont:
                    continue
                if not current_unit["content"].endswith((" ", "-", "/")):
                    current_unit["content"] += " "
                current_unit["content"] += cont
                continue

        # Cleanup text
        for mod in modules:
            for u in mod.get("units", []):
                if not isinstance(u.get("content"), str):
                    continue
                content = re.sub(r"\s+", " ", u["content"]).strip()
                u["content"] = content

        return modules

    def parse_book_entry(raw: str):
        """
        Parse a single textbook/reference book entry line into
        {title, authors, publisher, edition}.
        Best-effort, robust to OCR noise.
        """
        s = raw.replace("|", " ")
        s = re.sub(r"\s+", " ", s).strip()
        # remove leading index number
        s = re.sub(r"^\d+\s+", "", s)

        # Remove trailing year if any
        year_match = re.search(r"(19|20)\d{2}", s)
        if year_match:
            s_wo_year = s[:year_match.start()].strip()
        else:
            s_wo_year = s

        PUBLISHER_PAT = (
            r"(Pearson(?:\s+Education)?|PHI|Cambridge(?:\s+University\s+Press)?|"
            r"MIT\s+Press|Research\s+India|McGraw[-\s]?Hill|Wiley|Springer|"
            r"Oxford(?:\s+University\s+Press)?)"
        )

        # Edition
        ed_match = re.search(r"\b[\w\d]+(?:st|nd|rd|th)?\s+Edition\b", s_wo_year, re.I)

        title = ""
        authors = ""
        publisher = ""
        edition = ""

        if ed_match:
            edition = ed_match.group(0).strip()
            before_ed = s_wo_year[:ed_match.start()].strip()
            after_ed = s_wo_year[ed_match.end():].strip()
            title = before_ed.rstrip(".,;")

            if after_ed:
                m_pub = re.search(PUBLISHER_PAT, after_ed)
                if m_pub:
                    authors = after_ed[:m_pub.start()].strip().rstrip(",;")
                    publisher = after_ed[m_pub.start():].strip().rstrip(".,;")
                else:
                    authors = after_ed.strip().rstrip(",;")
        else:
            # No edition, try to find publisher directly
            m_pub = re.search(PUBLISHER_PAT, s_wo_year)
            if m_pub:
                before_p = s_wo_year[:m_pub.start()].strip()
                publisher = s_wo_year[m_pub.start():].strip().rstrip(".,;")
                title = before_p.rstrip(".,;")
            else:
                title = s_wo_year.rstrip(".,;")

        return {
            "title": title,
            "authors": authors,
            "publisher": publisher,
            "edition": edition,
        }

    def extract_books_from_text(txt: str, header_pattern: str, stop_patterns):
        """
        Extract list of books from a section starting with header_pattern,
        stopping before any of stop_patterns.
        """
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        n = len(lines)

        header_idx = -1
        for i, l in enumerate(lines):
            if re.search(header_pattern, l, re.I):
                header_idx = i
                break
        if header_idx == -1:
            return []

        stop_idx = n
        if stop_patterns:
            for i in range(header_idx + 1, n):
                low = lines[i].lower()
                if any(re.search(sp, low) for sp in stop_patterns):
                    stop_idx = i
                    break

        section_lines = [ln for ln in lines[header_idx + 1:stop_idx] if ln]
        # Remove header row like "Sr. No | Title Edition Authors Publisher Year"
        section_lines = [
            ln for ln in section_lines
            if not re.match(r"^sr\.", ln, re.I)
        ]

        books_raw = []
        current = ""

        for ln in section_lines:
            if re.match(r"^\d+\s", ln):
                if current:
                    books_raw.append(current.strip())
                current = ln
            else:
                if current:
                    current += " " + ln
        if current:
            books_raw.append(current.strip())

        books = []
        for raw in books_raw:
            entry = parse_book_entry(raw)
            if entry.get("title"):
                books.append(entry)

        return books

    def parse_syllabus_structured(txt: str):
        modules = parse_modules_and_units(txt)
        textbooks = extract_books_from_text(
            txt,
            header_pattern=r"^text\s*books",
            stop_patterns=[r"^reference\s*books"],
        )
        reference_books = extract_books_from_text(
            txt,
            header_pattern=r"^reference\s*books",
            stop_patterns=[],
        )
        return {
            "modules": modules,
            "textbooks": textbooks,
            "reference_books": reference_books,
        }

    # --------- actual logic ---------
    structured = parse_syllabus_structured(text)

    return {
        "subject": row["name"],
        "modules": structured.get("modules", []),
        "textbooks": structured.get("textbooks", []),
        "reference_books": structured.get("reference_books", []),
    }


@app.post("/syllabus/reparse/{subject_id}")
async def reparse_outline(subject_id: str):
    row = get_subject(subject_id)
    if not row:
        raise HTTPException(status_code=404, detail="Subject not found")

    outline = extract_outline(row["text"] or "")
    update_outline(subject_id, outline)
    return {"subject": row["name"], "chapters": outline.get("chapters", [])}

# -------------------- Startup --------------------
@app.on_event("startup")
async def on_startup():
    init_db()
    # Try to init chatbot without breaking the server if deps/models are missing
    try:
        from app import chatbot as cb
        cb.init_chatbot()
        print("✅ Chatbot models initialized (or deferred).")
        run_rag_rebuild()
    except Exception as e:
        print(f"ℹ️ Chatbot init skipped or failed: {e}")
    print("✅ Server ready!")

# Allow: python -m app.main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)