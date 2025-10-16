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








#first version
# # import os
# # import json
# # import fitz  # PyMuPDF
# # import re
# # from app.normalize import normalize_text


# # # ----------------------------------------------------------------------
# # # 🔹 Helper: Assign roles based on filename
# # # ----------------------------------------------------------------------
# # def assign_roles_from_filename(filename: str):
# #     """
# #     Assigns RBAC roles based on filename.
# #     If the file name contains 'restricted', restrict access to legal/admin only.
# #     """
# #     if "restricted" in filename.lower():
# #         return ["legal", "admin"]
# #     return ["staff", "legal", "admin"]


# # # ----------------------------------------------------------------------
# # # 🔹 Helper: Extract text from PDF
# # # ----------------------------------------------------------------------
# # def extract_text_from_pdf(pdf_path):
# #     """
# #     Extracts text content from each page of a PDF file.
# #     Returns a list of (page_number, text) tuples.
# #     """
# #     doc = fitz.open(pdf_path)
# #     pages = []
# #     for i, page in enumerate(doc):
# #         text = page.get_text("text").strip()
# #         if text:
# #             pages.append((i + 1, text))
# #     return pages


# # # ----------------------------------------------------------------------
# # # 🔹 Helper: Split text into article/section chunks
# # # ----------------------------------------------------------------------
# # def chunk_text_by_articles(pages):
# #     """
# #     Detects article or section headings using regex and splits the text accordingly.
# #     Returns a list of chunks, each with metadata: article_no, page_start, page_end, text.
# #     """
# #     chunks = []
# #     article_pattern = re.compile(r"(?:^|\n)\s*(Article|Section)\s*([0-9IVXLC]+)\b", re.IGNORECASE)

# #     current_chunk = None
# #     current_article_no = None
# #     start_page = None
# #     for page_num, text in pages:
# #         for match in article_pattern.finditer(text):
# #             # Save previous chunk if exists
# #             if current_chunk:
# #                 chunks.append({
# #                     "article_no": current_article_no,
# #                     "page_start": start_page,
# #                     "page_end": page_num,
# #                     "text": current_chunk.strip()
# #                 })
# #             # Start a new chunk
# #             current_article_no = match.group(2)
# #             start_page = page_num
# #             current_chunk = text[match.start():]
# #             break  # take first article per page (you can adjust if needed)
# #         else:
# #             # Continue accumulating text for current article
# #             if current_chunk:
# #                 current_chunk += "\n" + text
# #             else:
# #                 current_chunk = text

# #     # Add final chunk
# #     if current_chunk:
# #         chunks.append({
# #             "article_no": current_article_no or "Unknown",
# #             "page_start": start_page or 1,
# #             "page_end": pages[-1][0],
# #             "text": current_chunk.strip()
# #         })

# #     return chunks


# # # ----------------------------------------------------------------------
# # # 🔹 Main function: Chunk PDFs
# # # ----------------------------------------------------------------------
# # def process_pdfs(input_dir="data/raw_pdfs", output_path="data/processed/chunks.jsonl"):
# #     """
# #     Processes all PDFs in the input directory and writes their chunks to chunks.jsonl.
# #     Each chunk includes RBAC roles based on filename.
# #     """
# #     os.makedirs(os.path.dirname(output_path), exist_ok=True)

# #     with open(output_path, "w", encoding="utf-8") as out_f:
# #         for pdf_file in os.listdir(input_dir):
# #             if not pdf_file.lower().endswith(".pdf"):
# #                 continue

# #             pdf_path = os.path.join(input_dir, pdf_file)
# #             doc_id = os.path.splitext(pdf_file)[0]
# #             roles = assign_roles_from_filename(pdf_file)

# #             print(f"📄 Processing: {pdf_file} → roles={roles}")

# #             pages = extract_text_from_pdf(pdf_path)
# #             chunks = chunk_text_by_articles(pages)

# #             for chunk in chunks:
# #                 norm_text = normalize_text(chunk["text"])
# #                 record = {
# #                     "doc_id": doc_id,
# #                     "article_no": chunk["article_no"],
# #                     "page_start": chunk["page_start"],
# #                     "page_end": chunk["page_end"],
# #                     "text": chunk["text"],
# #                     "norm_text": norm_text,
# #                     "roles": roles  # ✅ Add RBAC roles here
# #                 }
# #                 out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

# #     print(f"\n✅ Saved chunks with RBAC roles to {output_path}")


# # # ----------------------------------------------------------------------
# # # 🔹 Entry Point
# # # ----------------------------------------------------------------------
# # if __name__ == "__main__":
# #     process_pdfs()


# second version


# import os
# import json
# import re
# import fitz  # PyMuPDF
# from app.normalize import normalize_text


# def assign_roles_from_filename(filename: str):
#     """Assign roles based on filename (RBAC)."""
#     if "restricted" in filename.lower():
#         return ["legal", "admin"]
#     return ["staff", "legal", "admin"]


# def extract_text_from_pdf(pdf_path):
#     """Extract plain text from all pages."""
#     doc = fitz.open(pdf_path)
#     pages = []
#     for i, page in enumerate(doc):
#         text = page.get_text("text").strip()
#         if text:
#             pages.append((i + 1, text))
#     return pages


# def smart_chunk_text(pages):
#     """
#     🔹 Improved chunker for legal/technical PDFs.
#     - Detects 'Article'/'Section' headings with regex.
#     - Ensures each Article is its own chunk.
#     - Automatically merges short Articles (<300 chars) with the next one for context.
#     """
#     chunks = []
#     article_pattern = re.compile(r"(?:^|\n)\s*(Article|Section)\s*([0-9IVXLC]+)\b", re.IGNORECASE)

#     current_text = ""
#     current_article = None
#     start_page = None

#     for page_num, text in pages:
#         matches = list(article_pattern.finditer(text))
#         if matches:
#             for j, match in enumerate(matches):
#                 # Save previous chunk before new article starts
#                 if current_text.strip():
#                     chunks.append({
#                         "article_no": current_article or "Unknown",
#                         "page_start": start_page or page_num,
#                         "page_end": page_num,
#                         "text": current_text.strip(),
#                     })
#                 # Start new article
#                 current_article = match.group(2)
#                 start_page = page_num
#                 # Get text till next heading or end of page
#                 end_idx = matches[j + 1].start() if j + 1 < len(matches) else len(text)
#                 current_text = text[match.start():end_idx]
#         else:
#             # Continue current article
#             current_text += "\n" + text

#     # Add final article chunk
#     if current_text.strip():
#         chunks.append({
#             "article_no": current_article or "Unknown",
#             "page_start": start_page or pages[0][0],
#             "page_end": pages[-1][0],
#             "text": current_text.strip(),
#         })

#     # --- Merge too-small chunks for better semantics ---
#     merged = []
#     buffer = None
#     for chunk in chunks:
#         if buffer is None:
#             buffer = chunk
#         else:
#             # If previous chunk too short, merge with current one
#             if len(buffer["text"]) < 300:
#                 buffer["text"] += "\n" + chunk["text"]
#                 buffer["page_end"] = chunk["page_end"]
#             else:
#                 merged.append(buffer)
#                 buffer = chunk
#     if buffer:
#         merged.append(buffer)

#     return merged


# def process_pdfs(input_dir="data/raw_pdfs", output_path="data/processed/chunks.jsonl"):
#     """
#     Process all PDFs and save optimized chunks with roles & normalized text.
#     """
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with open(output_path, "w", encoding="utf-8") as out_f:
#         for pdf_file in os.listdir(input_dir):
#             if not pdf_file.lower().endswith(".pdf"):
#                 continue

#             pdf_path = os.path.join(input_dir, pdf_file)
#             doc_id = os.path.splitext(pdf_file)[0]
#             roles = assign_roles_from_filename(pdf_file)

#             print(f"📘 Processing {pdf_file}  →  Roles={roles}")
#             pages = extract_text_from_pdf(pdf_path)
#             chunks = smart_chunk_text(pages)

#             for ch in chunks:
#                 norm_text = normalize_text(ch["text"])
#                 record = {
#                     "doc_id": doc_id,
#                     "article_no": ch["article_no"],
#                     "page_start": ch["page_start"],
#                     "page_end": ch["page_end"],
#                     "text": ch["text"],
#                     "norm_text": norm_text,
#                     "roles": roles,
#                 }
#                 out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

#     print(f"\n✅ Saved optimized chunks with RBAC roles → {output_path}")


# if __name__ == "__main__":
#     process_pdfs()
