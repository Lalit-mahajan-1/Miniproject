import json
from app.database_utils import list_subjects, get_subject_by_name
from app.syllabus_parser import parse_syllabus_structured  # NEW FILE

def build_context():
    context_parts = []

    subjects = list_subjects()

    for sub in subjects:
        name = sub["name"]

        row = get_subject_by_name(name)
        if not row:
            continue

        text = row["text"]

        structured = parse_syllabus_structured(text)

        ctx = f"""
===== SUBJECT: {name} =====

Modules:
{json.dumps(structured.get("modules", []), indent=2)}

Textbooks:
{json.dumps(structured.get("textbooks", []), indent=2)}

Reference Books:
{json.dumps(structured.get("reference_books", []), indent=2)}
"""
        context_parts.append(ctx)

    return "\n\n".join(context_parts)
