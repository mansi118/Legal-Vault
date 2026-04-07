"""
Shared utilities for the Legal Knowledge Base.
Manifest I/O, frontmatter, text helpers, Claude API wrapper, Rich console.
"""

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

import frontmatter
from rich.console import Console

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "raw"
WIKI_DIR = ROOT_DIR / "wiki"
OUTPUT_DIR = ROOT_DIR / "output"
INDEX_DIR = ROOT_DIR / "index"
MANIFEST_PATH = RAW_DIR / "_manifest.json"
LAST_COMPILED_PATH = ROOT_DIR / ".last_compiled"

# ---------------------------------------------------------------------------
# Rich console
# ---------------------------------------------------------------------------

console = Console()


def print_status(msg: str) -> None:
    console.print(f"[blue]i[/blue] {msg}")


def print_success(msg: str) -> None:
    console.print(f"[green]✓[/green] {msg}")


def print_error(msg: str) -> None:
    console.print(f"[red]✗[/red] {msg}", style="red")


def print_warning(msg: str) -> None:
    console.print(f"[yellow]![/yellow] {msg}")


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def load_manifest() -> list[dict]:
    """Read _manifest.json. Returns [] if missing or corrupt."""
    if not MANIFEST_PATH.exists():
        return []
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        print_warning("_manifest.json is not a JSON array, returning []")
        return []
    except (json.JSONDecodeError, OSError) as e:
        print_warning(f"Could not read _manifest.json: {e}")
        return []


def save_manifest(entries: list[dict]) -> None:
    """Write manifest atomically (write to .tmp, then os.replace)."""
    tmp_path = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(entries, indent=2, default=str, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, MANIFEST_PATH)


def add_manifest_entry(entry: dict) -> None:
    """Validate and append a single entry to the manifest."""
    required = {"id", "title", "filepath", "type", "ingested_at"}
    missing = required - set(entry.keys())
    if missing:
        raise ValueError(f"Manifest entry missing required keys: {missing}")

    entries = load_manifest()

    # Duplicate check by id
    if any(e["id"] == entry["id"] for e in entries):
        print_warning(f"Manifest entry with id={entry['id']} already exists, skipping")
        return

    entries.append(entry)
    save_manifest(entries)


def find_manifest_entry(doc_id: str) -> dict | None:
    """Find a manifest entry by document ID."""
    return next((e for e in load_manifest() if e["id"] == doc_id), None)


# ---------------------------------------------------------------------------
# Frontmatter I/O
# ---------------------------------------------------------------------------

def parse_frontmatter(filepath: Path) -> tuple[dict, str]:
    """Parse YAML frontmatter from a markdown file.

    Returns (metadata_dict, body_text). If no frontmatter, returns ({}, full_text).
    """
    post = frontmatter.load(filepath)
    return dict(post.metadata), post.content


def write_with_frontmatter(filepath: Path, metadata: dict, body: str) -> None:
    """Write a markdown file with YAML frontmatter. Creates parent dirs if needed."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    post = frontmatter.Post(body, **metadata)
    content = frontmatter.dumps(post) + "\n"
    filepath.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def slugify(title: str) -> str:
    """Convert a title to a URL/filename-safe slug.

    'Force Majeure (Post-COVID)' -> 'force-majeure-post-covid'
    """
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)      # remove special chars
    slug = re.sub(r"[\s_]+", "-", slug)        # spaces/underscores -> hyphens
    slug = re.sub(r"-+", "-", slug)            # collapse multiple hyphens
    slug = slug.strip("-")
    return slug


def word_count(text: str) -> int:
    """Simple word count."""
    return len(text.split())


def file_hash(filepath: Path) -> str:
    """SHA-256 hash of file contents, prefixed with 'sha256:'."""
    content = filepath.read_bytes()
    return "sha256:" + hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# Wiki stats
# ---------------------------------------------------------------------------

def wiki_stats() -> dict:
    """Compute wiki statistics: article count, concept count, word count, last compiled."""
    articles = list(WIKI_DIR.rglob("*.md"))
    concepts_dir = WIKI_DIR / "concepts"
    concepts = list(concepts_dir.glob("*.md")) if concepts_dir.exists() else []

    total_words = 0
    for f in articles:
        try:
            total_words += word_count(f.read_text(encoding="utf-8"))
        except OSError:
            pass

    last = get_last_compiled()
    return {
        "articles": len(articles),
        "concepts": len(concepts),
        "total_words": total_words,
        "last_compiled": last.isoformat() if last else "never",
    }


# ---------------------------------------------------------------------------
# Compilation timestamps
# ---------------------------------------------------------------------------

def get_last_compiled() -> datetime | None:
    """Read the last compilation timestamp. Returns None if never compiled."""
    if not LAST_COMPILED_PATH.exists():
        return None
    try:
        text = LAST_COMPILED_PATH.read_text(encoding="utf-8").strip()
        return datetime.fromisoformat(text)
    except (ValueError, OSError):
        return None


def set_last_compiled() -> None:
    """Write the current timestamp to .last_compiled."""
    LAST_COMPILED_PATH.write_text(
        datetime.now().isoformat() + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Claude API wrapper
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    """Lazy singleton for the Anthropic client."""
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Export it or add to a .env file.\n"
                "  export ANTHROPIC_API_KEY=sk-ant-..."
            )
        import anthropic
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def call_claude(
    system: str,
    user: str,
    max_tokens: int = 4096,
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """Send a single request to the Anthropic API and return the text response."""
    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def call_claude_json(
    system: str,
    user: str,
    max_tokens: int = 4096,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Call Claude and parse the response as JSON.

    Strips markdown code fences, retries once on parse failure.
    """
    system_with_json = (
        system + "\n\nRespond ONLY with valid JSON. No markdown code fences, no explanation."
    )

    raw = call_claude(system_with_json, user, max_tokens, model)
    parsed = _try_parse_json(raw)
    if parsed is not None:
        return parsed

    # Retry once with explicit correction
    retry_prompt = (
        f"Your previous response was not valid JSON. "
        f"Here is what you returned:\n\n{raw}\n\n"
        f"Please return ONLY valid JSON, no other text."
    )
    raw2 = call_claude(system_with_json, retry_prompt, max_tokens, model)
    parsed2 = _try_parse_json(raw2)
    if parsed2 is not None:
        return parsed2

    raise ValueError(
        f"Claude returned invalid JSON after retry.\n"
        f"Response: {raw2[:500]}\n"
    )


def _try_parse_json(text: str) -> dict | None:
    """Attempt to parse JSON from Claude's response, stripping code fences."""
    cleaned = text.strip()
    # Strip ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
        return None
    except json.JSONDecodeError:
        return None
