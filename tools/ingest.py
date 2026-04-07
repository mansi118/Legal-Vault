"""
Document ingestion pipeline for the Legal Knowledge Base.
Supports URL (HTML→MD), PDF, plain text, batch directory, and status reporting.
"""

import hashlib
import re
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import click
import markdownify
import requests
from bs4 import BeautifulSoup
from rich.table import Table

from tools.utils import (
    ROOT_DIR,
    RAW_DIR,
    add_manifest_entry,
    console,
    file_hash,
    load_manifest,
    print_error,
    print_status,
    print_success,
    print_warning,
    slugify,
    word_count,
    write_with_frontmatter,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES = ["papers", "articles", "statutes", "case-law", "contracts"]
REMOVE_TAGS = ["nav", "footer", "header", "script", "style", "aside", "noscript", "iframe"]
USER_AGENT = "Mozilla/5.0 (Legal-KB Ingest Bot)"
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_id() -> str:
    """Generate an 8-char hex document ID."""
    import uuid
    return uuid.uuid4().hex[:8]


def _ext_from_content_type(content_type: str) -> str:
    """Map Content-Type to file extension."""
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/svg+xml": ".svg",
        "image/webp": ".webp",
    }
    for mime, ext in mapping.items():
        if mime in content_type:
            return ext
    return ""


def _download_images(html: str, doc_id: str, base_url: str = "") -> str:
    """Download <img> sources to raw/assets/, rewrite paths to relative."""
    soup = BeautifulSoup(html, "html.parser")
    seen = {}  # url -> local filename (avoid re-downloading same image)

    for idx, img in enumerate(soup.find_all("img", src=True)):
        src = img["src"]
        if src.startswith("data:"):
            continue

        # Resolve relative URLs
        abs_url = src
        if not src.startswith(("http://", "https://")):
            if base_url:
                abs_url = urljoin(base_url, src)
            else:
                continue

        # Reuse if already downloaded
        if abs_url in seen:
            img["src"] = f"../assets/{seen[abs_url]}"
            continue

        try:
            resp = requests.get(abs_url, timeout=15, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()

            if len(resp.content) > MAX_IMAGE_SIZE:
                print_warning(f"Image too large, skipping: {abs_url}")
                continue

            content_type = resp.headers.get("Content-Type", "")
            ext = _ext_from_content_type(content_type)
            if not ext:
                ext = Path(urlparse(abs_url).path).suffix or ".png"

            filename = f"{doc_id}_{idx:03d}{ext}"
            save_path = RAW_DIR / "assets" / filename
            save_path.write_bytes(resp.content)
            seen[abs_url] = filename
            img["src"] = f"../assets/{filename}"
        except Exception as e:
            print_warning(f"Could not download image {abs_url}: {e}")

    return str(soup)


def _html_to_markdown(url: str) -> tuple[str, dict]:
    """Fetch URL, clean HTML, convert to markdown. Returns (body, metadata)."""
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
    resp.raise_for_status()

    # If URL returns PDF, delegate
    content_type = resp.headers.get("Content-Type", "")
    if "application/pdf" in content_type:
        tmp = Path(tempfile.mktemp(suffix=".pdf"))
        tmp.write_bytes(resp.content)
        try:
            body, meta = _pdf_to_markdown(str(tmp))
            meta["source_url"] = url
            return body, meta
        finally:
            tmp.unlink(missing_ok=True)

    # Handle encoding
    resp.encoding = resp.apparent_encoding or "utf-8"
    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    # Extract metadata
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else ""
    if not title:
        title = urlparse(url).path.strip("/").split("/")[-1] or "Untitled"

    meta_desc = soup.find("meta", attrs={"name": "description"})
    description = ""
    if meta_desc and meta_desc.get("content"):
        description = meta_desc["content"]

    # Remove junk tags
    for tag_name in REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Find main content
    content_el = None
    for selector in ["article", "main"]:
        content_el = soup.find(selector)
        if content_el:
            break
    if not content_el:
        content_el = soup.select_one("[role='main']")
    if not content_el:
        content_el = soup.find("body") or soup

    # Download images
    img_id = _generate_id()
    content_html = _download_images(str(content_el), img_id, base_url=url)

    # Convert to markdown
    md = markdownify.markdownify(
        content_html,
        heading_style="ATX",
        bullets="-",
    )

    # Post-clean: collapse excessive blank lines
    md = re.sub(r"\n{3,}", "\n\n", md).strip()

    if len(md) < 100:
        print_warning(f"Very little text extracted ({len(md)} chars) — page may require JavaScript")

    metadata = {
        "title": title,
        "source_url": url,
        "description": description,
    }
    return md, metadata


def _pdf_to_markdown(filepath: str) -> tuple[str, dict]:
    """Extract text from PDF using pdfplumber (fallback PyPDF2). Returns (body, metadata)."""
    path = Path(filepath)
    pages = []
    page_count = 0

    # Try pdfplumber first
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
                # Extract tables as markdown
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        md_table = _table_to_markdown(table)
                        if md_table:
                            pages.append(md_table)
    except Exception as e:
        print_warning(f"pdfplumber failed, trying PyPDF2: {e}")
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(path)
            page_count = len(reader.pages)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        except Exception as e2:
            print_error(f"Both PDF extractors failed: {e2}")
            pages = []
            page_count = 0

    body = "\n\n---\n\n".join(pages)

    if not body.strip():
        print_warning(f"No text extracted from {filepath} — may be a scanned document")

    # Extract title from first reasonable line or filename
    title = ""
    for line in body.strip().split("\n")[:5]:
        line = line.strip()
        if 5 < len(line) < 200 and not line.startswith("|"):
            title = line
            break
    if not title:
        title = path.stem.replace("-", " ").replace("_", " ").title()

    metadata = {
        "title": title,
        "source_path": str(path.resolve()),
        "page_count": page_count,
    }
    return body, metadata


def _table_to_markdown(table: list[list]) -> str:
    """Convert a pdfplumber table to markdown table format."""
    if not table or not table[0]:
        return ""
    rows = []
    for row in table:
        rows.append([str(cell).strip() if cell else "" for cell in row])

    header = "| " + " | ".join(rows[0]) + " |"
    separator = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, separator] + body_rows)


def _text_to_markdown(filepath: str) -> tuple[str, dict]:
    """Read a plain text file and wrap as markdown. Returns (body, metadata)."""
    path = Path(filepath)
    try:
        body = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        body = path.read_text(encoding="latin-1")

    lines = body.strip().split("\n")
    title = lines[0].strip() if lines else path.stem
    # Add heading if first line looks like a title
    if lines and not lines[0].strip().startswith("#"):
        body = f"# {title}\n\n" + "\n".join(lines[1:])

    metadata = {
        "title": title,
        "source_path": str(path.resolve()),
    }
    return body, metadata


def _detect_category(text: str, source: str, title: str) -> str:
    """Heuristic classification of document category."""
    combined = (title + " " + source + " " + text[:2000]).lower()

    # Case law
    if re.search(r"\bv\.?\s", title, re.IGNORECASE) or "versus" in title.lower():
        return "case-law"
    if any(kw in combined for kw in [
        "court", "judgment", "appellant", "respondent", "hon'ble", "petitioner",
        "plaintiff", "defendant", "tribunal",
    ]):
        return "case-law"

    # Statutes
    if any(kw in combined for kw in [
        " act,", " act ", "code of", "regulation", "statute", "section ", "§",
        "ordinance", "amendment",
    ]):
        return "statutes"

    # Contracts
    if any(kw in combined for kw in [
        "agreement", "hereby", "whereas", "party of", "terms and conditions",
        "non-disclosure", "indemnif", "governing law",
    ]):
        return "contracts"

    # Papers
    if any(kw in combined for kw in [
        "abstract", "keywords:", "references", "methodology", "doi:", "journal",
        "literature review",
    ]):
        return "papers"

    return "articles"


def _body_hash(body: str) -> str:
    """Compute a hash of just the body text (no frontmatter) for duplicate detection."""
    return "sha256:" + hashlib.sha256(body.encode("utf-8", errors="replace")).hexdigest()


def _check_duplicate(body: str) -> dict | None:
    """Check if content already exists in manifest by body hash. Returns existing entry or None."""
    h = _body_hash(body)
    for entry in load_manifest():
        if entry.get("body_hash") == h:
            return entry
    return None


def _save_raw_document(
    body: str, metadata: dict, category: str, doc_id: str, title_override: str | None = None
) -> Path:
    """Save a raw document with frontmatter and update manifest. Returns file path."""
    title = title_override or metadata.get("title", "untitled")
    slug = slugify(title) or doc_id

    filepath = RAW_DIR / category / f"{slug}.md"

    # Handle slug collision
    if filepath.exists():
        filepath = RAW_DIR / category / f"{slug}-{doc_id[:4]}.md"

    wc = word_count(body)

    # Map category to type
    type_map = {
        "papers": "paper",
        "articles": "article",
        "statutes": "statute",
        "case-law": "case-law",
        "contracts": "contract",
    }

    frontmatter_data = {
        "id": doc_id,
        "title": title,
        "source_url": metadata.get("source_url", ""),
        "source_path": metadata.get("source_path", ""),
        "type": type_map.get(category, "article"),
        "category": category,
        "ingested_at": datetime.now().isoformat(timespec="seconds"),
        "compiled": False,
        "tags": [],
        "word_count": wc,
    }

    page_count = metadata.get("page_count")
    if page_count is not None:
        frontmatter_data["page_count"] = page_count

    write_with_frontmatter(filepath, frontmatter_data, body)

    # Compute hashes
    content_hash = file_hash(filepath)       # full file (for compile change detection)
    b_hash = _body_hash(body)                # body only (for duplicate detection)

    # Add to manifest
    manifest_entry = {
        "id": doc_id,
        "title": title,
        "filepath": str(filepath.relative_to(ROOT_DIR)),
        "type": frontmatter_data["type"],
        "category": category,
        "source_url": metadata.get("source_url", ""),
        "ingested_at": frontmatter_data["ingested_at"],
        "compiled": False,
        "word_count": wc,
        "content_hash": content_hash,
        "body_hash": b_hash,
    }
    add_manifest_entry(manifest_entry)

    return filepath


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------


@click.group()
def ingest():
    """Ingest legal documents into the knowledge base."""
    pass


@ingest.command("url")
@click.argument("url")
@click.option("--category", "-c", type=click.Choice(CATEGORIES), default=None,
              help="Override auto-detected category")
@click.option("--title", "-t", default=None, help="Override document title")
def ingest_url(url, category, title):
    """Ingest a web article or document from a URL."""
    print_status(f"Fetching {url}")

    try:
        body, metadata = _html_to_markdown(url)
    except requests.RequestException as e:
        print_error(f"Failed to fetch URL: {e}")
        return
    except Exception as e:
        print_error(f"Failed to process URL: {e}")
        return

    # Duplicate check
    existing = _check_duplicate(body)
    if existing:
        print_warning(
            f"Duplicate content detected — matches '{existing['title']}' "
            f"({existing['filepath']}). Skipping."
        )
        return

    doc_id = _generate_id()
    doc_title = title or metadata.get("title", "Untitled")

    if not category:
        category = _detect_category(body, url, doc_title)
        print_status(f"Auto-detected category: {category}")

    filepath = _save_raw_document(body, metadata, category, doc_id, title_override=title)
    print_success(f"Ingested: {filepath.relative_to(ROOT_DIR)} ({word_count(body)} words)")


@ingest.command("pdf")
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--category", "-c", type=click.Choice(CATEGORIES), default=None,
              help="Override auto-detected category")
@click.option("--title", "-t", default=None, help="Override document title")
def ingest_pdf(filepath, category, title):
    """Ingest a PDF document."""
    print_status(f"Processing {filepath}")

    try:
        body, metadata = _pdf_to_markdown(filepath)
    except Exception as e:
        print_error(f"Failed to process PDF: {e}")
        return

    existing = _check_duplicate(body)
    if existing:
        print_warning(
            f"Duplicate content detected — matches '{existing['title']}' "
            f"({existing['filepath']}). Skipping."
        )
        return

    doc_id = _generate_id()
    doc_title = title or metadata.get("title", "Untitled")

    if not category:
        category = _detect_category(body, filepath, doc_title)
        print_status(f"Auto-detected category: {category}")

    result_path = _save_raw_document(body, metadata, category, doc_id, title_override=title)
    print_success(f"Ingested: {result_path.relative_to(ROOT_DIR)} ({word_count(body)} words)")


@ingest.command("text")
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--category", "-c", type=click.Choice(CATEGORIES), default=None,
              help="Override auto-detected category")
@click.option("--title", "-t", default=None, help="Override document title")
def ingest_text(filepath, category, title):
    """Ingest a plain text file."""
    print_status(f"Processing {filepath}")

    try:
        body, metadata = _text_to_markdown(filepath)
    except Exception as e:
        print_error(f"Failed to process text file: {e}")
        return

    existing = _check_duplicate(body)
    if existing:
        print_warning(
            f"Duplicate content detected — matches '{existing['title']}' "
            f"({existing['filepath']}). Skipping."
        )
        return

    doc_id = _generate_id()
    doc_title = title or metadata.get("title", "Untitled")

    if not category:
        category = _detect_category(body, filepath, doc_title)
        print_status(f"Auto-detected category: {category}")

    result_path = _save_raw_document(body, metadata, category, doc_id, title_override=title)
    print_success(f"Ingested: {result_path.relative_to(ROOT_DIR)} ({word_count(body)} words)")


@ingest.command("dir")
@click.argument("directory", type=click.Path(exists=True))
@click.option("--category", "-c", type=click.Choice(CATEGORIES), default=None,
              help="Override auto-detected category for all files")
def ingest_dir(directory, category):
    """Batch ingest all PDFs, HTML, and TXT files in a directory."""
    dir_path = Path(directory)
    extensions = {".pdf", ".html", ".htm", ".txt", ".md"}

    files = sorted(
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )

    if not files:
        print_warning(f"No ingestable files found in {directory}")
        return

    print_status(f"Found {len(files)} files to ingest")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for f in files:
        try:
            suffix = f.suffix.lower()
            if suffix == ".pdf":
                body, metadata = _pdf_to_markdown(str(f))
            elif suffix in {".html", ".htm"}:
                html_content = f.read_text(encoding="utf-8", errors="replace")
                soup = BeautifulSoup(html_content, "html.parser")
                title_tag = soup.find("title")
                doc_title = title_tag.get_text(strip=True) if title_tag else f.stem
                md = markdownify.markdownify(html_content, heading_style="ATX")
                body = re.sub(r"\n{3,}", "\n\n", md).strip()
                metadata = {"title": doc_title, "source_path": str(f.resolve())}
            else:
                body, metadata = _text_to_markdown(str(f))

            existing = _check_duplicate(body)
            if existing:
                print_warning(f"Skipping duplicate: {f.name}")
                skip_count += 1
                continue

            doc_id = _generate_id()
            doc_title = metadata.get("title", f.stem)
            cat = category or _detect_category(body, str(f), doc_title)
            _save_raw_document(body, metadata, cat, doc_id)
            print_success(f"  {f.name} -> {cat}")
            success_count += 1

        except Exception as e:
            print_error(f"  Failed: {f.name} — {e}")
            fail_count += 1

    console.print()
    print_success(f"Batch complete: {success_count} ingested, {skip_count} skipped, {fail_count} failed")


@ingest.command("status")
def ingest_status():
    """Show ingestion status summary."""
    manifest = load_manifest()

    if not manifest:
        print_status("No documents ingested yet.")
        return

    # Aggregate by category
    categories = {}
    for entry in manifest:
        cat = entry.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"count": 0, "compiled": 0, "words": 0}
        categories[cat]["count"] += 1
        if entry.get("compiled", False):
            categories[cat]["compiled"] += 1
        categories[cat]["words"] += entry.get("word_count", 0)

    # Build table
    table = Table(title="Ingestion Status Summary")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Compiled", justify="right")
    table.add_column("Words", justify="right")

    total_count = 0
    total_compiled = 0
    total_words = 0

    for cat in sorted(categories.keys()):
        data = categories[cat]
        total_count += data["count"]
        total_compiled += data["compiled"]
        total_words += data["words"]
        table.add_row(
            cat,
            str(data["count"]),
            f"{data['compiled']}/{data['count']}",
            f"{data['words']:,}",
        )

    table.add_section()
    table.add_row(
        "Total",
        str(total_count),
        f"{total_compiled}/{total_count}",
        f"{total_words:,}",
        style="bold",
    )

    console.print(table)


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ingest()
