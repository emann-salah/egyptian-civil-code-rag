import json
import re
import sys

import pdfplumber

from civilcode_rag.schemas import ArticleRecord
from civilcode_rag.ingest.arabic_utils import fix_pdfplumber_arabic_cell, extract_article_number

ARTICLE_EN_RE = re.compile(r"^A?rticle\s+(\d+)")
REPEALED_RANGE_RE = re.compile(r"Articles?\s+(\d+)\s*-\s*(\d+)\s+(?:have been )?repealed", re.IGNORECASE)


def extract_all_rows(pdf_path, max_pages=None):
    """Return every table row across the PDF as (page_number, row)."""
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
        for page_num, page in enumerate(pages, start=1):
            for table in page.extract_tables():
                for row in table:
                    rows.append((page_num, row))
    return rows


def build_records(rows):
    """Group rows into ArticleRecord objects."""
    records = []
    current_en_lines = []
    current_ar_lines = []
    current_number = None
    current_page = None

    def flush():
        """Save the article we've been accumulating, if any."""
        if current_number is None:
            return
        text_en = "\n".join(current_en_lines).strip()
        text_ar = "\n".join(current_ar_lines).strip() or "[EXTRACTION_FAILED]"
        records.append(ArticleRecord(
            article_number=current_number,
            text_en=text_en,
            text_ar=text_ar,
            source_page=current_page,
            citation=f"Egyptian Civil Code, Article {current_number}",
        ))

    for page_num, row in rows:
        en_cell = (row[0] or "").strip() if len(row) > 0 else ""
        ar_cell_raw = (row[1] or "").strip() if len(row) > 1 else ""
        ar_cell_fixed = fix_pdfplumber_arabic_cell(ar_cell_raw) if ar_cell_raw else ""

        match = ARTICLE_EN_RE.match(en_cell)
        if match:
            flush()  
            current_number = int(match.group(1))
            current_page = page_num
            current_en_lines = [ARTICLE_EN_RE.sub("", en_cell, count=1).strip()]
            current_ar_lines = [ar_cell_fixed.split("\n", 1)[-1] if "\n" in ar_cell_fixed else ""]
        else:
            if current_number is not None:
                if en_cell:
                    current_en_lines.append(en_cell)
                if ar_cell_fixed:
                    current_ar_lines.append(ar_cell_fixed)

    flush()  
    return records


def expand_repealed_ranges(records):
    """Turn a single 'Articles X-Y repealed' record into one flagged
    record per article number in that range, so no number is silently
    missing from the corpus."""
    expanded = []
    existing_numbers = {r.article_number for r in records}
    for r in records:
        expanded.append(r)
        match = REPEALED_RANGE_RE.search(r.text_en)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            for n in range(start, end + 1):
                if n in existing_numbers:
                    continue
                expanded.append(ArticleRecord(
                    article_number=n,
                    text_ar=r.text_ar,
                    text_en=r.text_en,
                    is_repealed=True,
                    source_page=r.source_page,
                    citation=f"Egyptian Civil Code, Article {n}",
                ))
                existing_numbers.add(n)
    return expanded





if __name__ == "__main__":
    pdf_path = "data/raw/egyptian_civil_code.pdf"
    max_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    rows = extract_all_rows(pdf_path, max_pages=max_pages)
    print(f"[info] extracted {len(rows)} rows from {max_pages} page(s)", file=sys.stderr)

    records = build_records(rows)
    records = expand_repealed_ranges(records)
    print(f"[info] built {len(records)} article records", file=sys.stderr)

    articles_data = [r.model_dump() for r in records]
    with open("data/processed/articles.json", "w", encoding="utf-8") as f:
        json.dump(articles_data, f, ensure_ascii=False, indent=2)
    print(f"[info] wrote {len(articles_data)} articles to data/processed/articles.json", file=sys.stderr)