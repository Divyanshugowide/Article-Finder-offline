import os
import re
import json
import fitz  # PyMuPDF
from app.normalize import normalize_text


# ----------------------------------------------------------------------
# 🔹 Role Assignment (RBAC)
# ----------------------------------------------------------------------
def assign_roles_from_filename(filename: str):
    """Assign access roles based on filename naming convention."""
    if "restricted" in filename.lower():
        return ["legal", "admin"]
    return ["staff", "legal", "admin"]


# ----------------------------------------------------------------------
# 🔹 PDF Text Extraction with Line Numbers
# ----------------------------------------------------------------------
def extract_text_with_lines(pdf_path):
    """
    Extracts text line by line from each page with line numbers.
    Returns: list of dicts [{page_num, line_num, text}]
    """
    doc = fitz.open(pdf_path)
    lines = []
    for page_idx, page in enumerate(doc):
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        sorted_blocks = sorted(blocks, key=lambda b: (round(b[1]), round(b[0])))  # sort top→bottom
        line_count = 0
        for _, _, _, _, block_text, *_ in sorted_blocks:
            block_text = block_text.strip()
            if not block_text:
                continue
            # Split block into lines
            for line in block_text.splitlines():
                line = line.strip()
                if len(line) < 5:
                    continue
                line_count += 1
                lines.append({
                    "page_num": page_idx + 1,
                    "line_num": line_count,
                    "text": line
                })
    return lines


# ----------------------------------------------------------------------
# 🔹 Smart Chunking (Article + Fallback + Line-Accurate)
# ----------------------------------------------------------------------
def chunk_text_universal(lines):
    """
    Universal chunking logic:
      1️⃣ Detects "Article"/"Section"/"Chapter"
      2️⃣ Falls back to paragraph + page-based chunking
      3️⃣ Includes page + line range per chunk
      4️⃣ Assigns article_no=0 if not found
    """
    full_text = "\n".join([l["text"] for l in lines])
    article_pattern = re.compile(
        r"(?:^|\n)\s*(Article|Section|Chapter)\s*([0-9IVXLC]+|[A-Z]|\d+(\.\d+)?)\b",
        re.IGNORECASE
    )
    matches = list(article_pattern.finditer(full_text))
    chunks = []

    # --- Case 1: Article/Section-based chunking ---
    if matches:
        print("🧠 Using Article/Section-based chunking...")
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            text_chunk = full_text[start:end].strip()
            article_no = match.group(2)

            page_start, line_start = _locate_line_position(lines, start)
            page_end, line_end = _locate_line_position(lines, end)

            chunks.append({
                "article_no": article_no,
                "page_start": page_start,
                "page_end": page_end,
                "line_start": line_start,
                "line_end": line_end,
                "text": text_chunk
            })

    # --- Case 2: Fallback mode (no articles) ---
    else:
        print("📄 No Article/Section found → Using line-level fallback.")
        article_no = "0"

        # Combine every ~300-500 words into a chunk
        buffer, chunk_start, word_count = "", None, 0
        start_page, start_line = lines[0]["page_num"], lines[0]["line_num"]

        for line in lines:
            if chunk_start is None:
                chunk_start = line
                start_page, start_line = line["page_num"], line["line_num"]

            buffer += " " + line["text"]
            word_count += len(line["text"].split())

            # Create chunk every ~350 words
            if word_count >= 350:
                chunks.append({
                    "article_no": article_no,
                    "page_start": start_page,
                    "page_end": line["page_num"],
                    "line_start": start_line,
                    "line_end": line["line_num"],
                    "text": buffer.strip()
                })
                buffer, chunk_start, word_count = "", None, 0

        # Add leftover
        if buffer.strip():
            last_line = lines[-1]
            chunks.append({
                "article_no": article_no,
                "page_start": start_page,
                "page_end": last_line["page_num"],
                "line_start": start_line,
                "line_end": last_line["line_num"],
                "text": buffer.strip()
            })

    return chunks


# ----------------------------------------------------------------------
# 🔹 Find Page + Line Range for Article/Section Chunks
# ----------------------------------------------------------------------
def _locate_line_position(lines, char_index, window=250):
    """
    Estimates which page & line number a character index belongs to.
    """
    cumulative = 0
    for line in lines:
        line_len = len(line["text"]) + 1
        if cumulative <= char_index < cumulative + line_len + window:
            return line["page_num"], line["line_num"]
        cumulative += line_len
    return lines[-1]["page_num"], lines[-1]["line_num"]


# ----------------------------------------------------------------------
# 🔹 Main Processor
# ----------------------------------------------------------------------
def process_pdfs(input_dir="data/raw_pdfs", output_path="data/processed/chunks.jsonl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for pdf_file in os.listdir(input_dir):
            if not pdf_file.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(input_dir, pdf_file)
            doc_id = os.path.splitext(pdf_file)[0]
            roles = assign_roles_from_filename(pdf_file)

            print(f"\n📘 Processing: {pdf_file} → roles={roles}")
            lines = extract_text_with_lines(pdf_path)
            if not lines:
                print(f"⚠️ No text found in {pdf_file}. Skipping...")
                continue

            chunks = chunk_text_universal(lines)
            print(f"📑 Created {len(chunks)} chunks for {pdf_file}")

            for chunk in chunks:
                norm_text = normalize_text(chunk["text"])
                record = {
                    "doc_id": doc_id,
                    "article_no": chunk["article_no"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "line_start": chunk["line_start"],
                    "line_end": chunk["line_end"],
                    "text": chunk["text"],
                    "norm_text": norm_text,
                    "roles": roles
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ All chunks saved successfully → {output_path}")


# ----------------------------------------------------------------------
# 🔹 Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    process_pdfs()








