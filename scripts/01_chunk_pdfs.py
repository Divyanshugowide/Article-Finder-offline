#final version work on all pdf
import os
import re
import json
import fitz  # PyMuPDF
from app.normalize import normalize_text


# ----------------------------------------------------------------------
# 🔹 Helper: Assign roles from filename (RBAC)
# ----------------------------------------------------------------------
def assign_roles_from_filename(filename: str):
    """
    Assigns roles based on filename keywords.
    Example:
      - 'restricted' → only legal/admin can access
      - others → staff/legal/admin
    """
    if "restricted" in filename.lower():
        return ["legal", "admin"]
    return ["staff", "legal", "admin"]


# ----------------------------------------------------------------------
# 🔹 Extract text from PDF pages
# ----------------------------------------------------------------------
def extract_text_from_pdf(pdf_path):
    """
    Reads a PDF and extracts text from each page.
    Returns: list of (page_number, text)
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append((i + 1, text))
    return pages


# ----------------------------------------------------------------------
# 🔹 Smart Chunking Function
# ----------------------------------------------------------------------
def chunk_text_universal(pages):
    """
    Universal chunking:
      1️⃣ Tries Article/Section-based splitting (for legal PDFs)
      2️⃣ Falls back to page or paragraph chunks (for general PDFs)
    Returns: list of chunk dicts
    """
    chunks = []
    article_pattern = re.compile(r"(?:^|\n)\s*(Article|Section)\s*([0-9IVXLC]+)\b", re.IGNORECASE)

    full_text = "\n".join([t for _, t in pages])
    article_matches = list(article_pattern.finditer(full_text))

    # 🧩 Mode 1: Article/Section based
    if article_matches:
        print("🧠 Using Article/Section-based chunking...")
        for i, match in enumerate(article_matches):
            start = match.start()
            end = article_matches[i + 1].start() if i + 1 < len(article_matches) else len(full_text)
            text_chunk = full_text[start:end].strip()
            article_no = match.group(2)

            # Estimate page range
            page_start, page_end = _estimate_pages_for_chunk(start, end, pages)
            chunks.append({
                "article_no": article_no,
                "page_start": page_start,
                "page_end": page_end,
                "text": text_chunk
            })

    # 🧩 Mode 2: Fallback for unstructured PDFs
    else:
        print("📄 Using fallback chunking (page + paragraph)...")
        for page_num, text in pages:
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 100]  # long paragraphs only
            if not paragraphs:
                paragraphs = [text]  # fallback if no clear paragraphs
            for i, para in enumerate(paragraphs):
                chunks.append({
                    "article_no": f"Page{page_num}-Part{i+1}",
                    "page_start": page_num,
                    "page_end": page_num,
                    "text": para.strip()
                })

    return chunks


# ----------------------------------------------------------------------
# 🔹 Estimate which pages a chunk belongs to
# ----------------------------------------------------------------------
def _estimate_pages_for_chunk(start, end, pages):
    cumulative = 0
    page_start, page_end = None, None
    for p_idx, (_, p_txt) in enumerate(pages):
        next_cum = cumulative + len(p_txt)
        if page_start is None and start < next_cum:
            page_start = p_idx + 1
        if end <= next_cum:
            page_end = p_idx + 1
            break
        cumulative = next_cum
    if page_start is None:
        page_start = 1
    if page_end is None:
        page_end = len(pages)
    return page_start, page_end


# ----------------------------------------------------------------------
# 🔹 Main: Process all PDFs and save chunks
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
            pages = extract_text_from_pdf(pdf_path)
            if not pages:
                print(f"⚠️ No text found in {pdf_file}. Skipping...")
                continue

            chunks = chunk_text_universal(pages)

            for chunk in chunks:
                norm_text = normalize_text(chunk["text"])
                record = {
                    "doc_id": doc_id,
                    "article_no": chunk["article_no"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "text": chunk["text"],
                    "norm_text": norm_text,
                    "roles": roles
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved chunks (structured + unstructured) to {output_path}")


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
