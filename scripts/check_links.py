"""Check relative Markdown file links without probing remote services."""
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\(([^\n)]+)\)")


def main():
    broken = []
    checked = 0
    files = sorted(ROOT.rglob("*.md"))
    for source in files:
        if any(part in {".git", ".venv", "data", "runs", ".superpowers"} for part in source.relative_to(ROOT).parts):
            continue
        content = source.read_text(encoding="utf-8")
        # Inline examples inside code fences are not navigational links.
        content = re.sub(r"```.*?```", "", content, flags=re.S)
        for match in LINK.finditer(content):
            target = match.group(1).strip()
            target = target[1:target.find(">")] if target.startswith("<") else target.split()[0]
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            checked += 1
            resolved = (source.parent / unquote(url.path)).resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.exists():
                line = content.count("\n", 0, match.start()) + 1
                broken.append(f"{source.relative_to(ROOT)}:{line}: {target}")
    if broken:
        print("Broken local Markdown links:")
        print("\n".join(broken))
        return 1
    print(f"Checked {checked} local file links across Markdown files: all targets exist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
