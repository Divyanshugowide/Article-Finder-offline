import os
import json
import fitz  # PyMuPDF
import re
from app.normalize import normalize_text


# ----------------------------------------------------------------------
# 🔹 Helper: Assign roles based on filename
# ----------------------------------------------------------------------
def assign_roles_from_filename(filename: str):
    """
    Assigns RBAC roles based on filename.
    If the file name contains 'restricted', restrict access to legal/admin only.
    """
    if "restricted" in filename.lower():
        return ["legal", "admin"]
    return ["staff", "legal", "admin"]


# ----------------------------------------------------------------------
# 🔹 Helper: Extract text from PDF
# ----------------------------------------------------------------------
def extract_text_from_pdf(pdf_path):
    """
    Extracts text content from each page of a PDF file.
    Returns a list of (page_number, text) tuples.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append((i + 1, text))
    return pages


# ----------------------------------------------------------------------
# 🔹 Helper: Split text into article/section chunks
# ----------------------------------------------------------------------
def chunk_text_by_articles(pages):
    """
    Detects article or section headings using regex and splits the text accordingly.
    Returns a list of chunks, each with metadata: article_no, page_start, page_end, text.
    """
    chunks = []
    article_pattern = re.compile(r"(?:^|\n)\s*(Article|Section)\s*([0-9IVXLC]+)\b", re.IGNORECASE)

    current_chunk = None
    current_article_no = None
    start_page = None
    for page_num, text in pages:
        for match in article_pattern.finditer(text):
            # Save previous chunk if exists
            if current_chunk:
                chunks.append({
                    "article_no": current_article_no,
                    "page_start": start_page,
                    "page_end": page_num,
                    "text": current_chunk.strip()
                })
            # Start a new chunk
            current_article_no = match.group(2)
            start_page = page_num
            current_chunk = text[match.start():]
            break  # take first article per page (you can adjust if needed)
        else:
            # Continue accumulating text for current article
            if current_chunk:
                current_chunk += "\n" + text
            else:
                current_chunk = text

    # Add final chunk
    if current_chunk:
        chunks.append({
            "article_no": current_article_no or "Unknown",
            "page_start": start_page or 1,
            "page_end": pages[-1][0],
            "text": current_chunk.strip()
        })

    return chunks


# ----------------------------------------------------------------------
# 🔹 Main function: Chunk PDFs
# ----------------------------------------------------------------------
def process_pdfs(input_dir="data/raw_pdfs", output_path="data/processed/chunks.jsonl"):
    """
    Processes all PDFs in the input directory and writes their chunks to chunks.jsonl.
    Each chunk includes RBAC roles based on filename.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for pdf_file in os.listdir(input_dir):
            if not pdf_file.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(input_dir, pdf_file)
            doc_id = os.path.splitext(pdf_file)[0]
            roles = assign_roles_from_filename(pdf_file)

            print(f"📄 Processing: {pdf_file} → roles={roles}")

            pages = extract_text_from_pdf(pdf_path)
            chunks = chunk_text_by_articles(pages)

            for chunk in chunks:
                norm_text = normalize_text(chunk["text"])
                record = {
                    "doc_id": doc_id,
                    "article_no": chunk["article_no"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "text": chunk["text"],
                    "norm_text": norm_text,
                    "roles": roles  # ✅ Add RBAC roles here
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved chunks with RBAC roles to {output_path}")


# ----------------------------------------------------------------------
# 🔹 Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    process_pdfs()
