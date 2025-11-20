import re

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

        # Stop when lab/books start
        if re.search(r"\blaboratory component\b", low):
            break
        if low.startswith("text books") or low.startswith("textbooks"):
            break
        if low.startswith("reference books"):
            break

        # Skip headers like: Module | Unit | No.
        if "module" in low and "unit" in low and "no" in low:
            continue
        if low.startswith("topics ref"):
            continue
        if low.startswith("total "):
            continue

        # MODULE — pattern: "1 Title | Something"
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

        # SELF-STUDY Module
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

            # parse inline unit
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

        # UNIT patterns
        if current_module:
            m_unit = re.match(r"^(\d+(?:\.\d+)?)\s*\|\s*(.+)$", line)
            if not m_unit:
                m_unit = re.match(r"^(\d+(?:\.\d+)?)\s+(.+)$", line)

            if m_unit:
                num_str = m_unit.group(1).strip()
                rest = m_unit.group(2).strip()

                # ignore false module header
                if rest.lower().startswith("title"):
                    continue

                # Fix OCR: "41" -> "4.1"
                unit_no = num_str
                if (
                    "." not in num_str
                    and len(num_str) == 2
                    and str(current_module["module_no"]) == num_str[0]
                ):
                    unit_no = f"{num_str[0]}.{num_str[1]}"

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

        # CONTINUATION lines
        if current_module and current_unit:
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

    # cleanup
    for mod in modules:
        for u in mod.get("units", []):
            if isinstance(u.get("content"), str):
                u["content"] = re.sub(r"\s+", " ", u["content"]).strip()

    return modules


def parse_book_entry(raw: str):
    import re
    s = raw.replace("|", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # remove leading index number
    s = re.sub(r"^\d+\s+", "", s)

    # Remove year
    year_match = re.search(r"(19|20)\d{2}", s)
    if year_match:
        s_wo_year = s[:year_match.start()].strip()
    else:
        s_wo_year = s

    publisher_pat = (
        r"(Pearson(?:\s+Education)?|PHI|Cambridge(?:\s+University\s+Press)?|"
        r"MIT\s+Press|Research\s+India|McGraw[-\s]?Hill|Wiley|Springer|"
        r"Oxford(?:\s+University\s+Press)?)"
    )

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
            m_pub = re.search(publisher_pat, after_ed)
            if m_pub:
                authors = after_ed[:m_pub.start()].strip().rstrip(",;")
                publisher = after_ed[m_pub.start():].strip().rstrip(".,;")
            else:
                authors = after_ed.strip().rstrip(",;")
    else:
        m_pub = re.search(publisher_pat, s_wo_year)
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

    # remove headers
    section_lines = [ln for ln in section_lines if not re.match(r"^sr\.", ln, re.I)]

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
    """
    Main function you will import in syllabus_context.py
    """
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
