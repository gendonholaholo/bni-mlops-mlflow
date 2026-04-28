"""CLI: validate every *.yaml in given directory(ies) against PromptYAML schema.

Usage: python scripts/validate_prompts.py prompts/
Exit codes: 0 = all valid; 1 = one or more invalid.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path so `prompts._schema` is importable when the
# script is invoked as `python scripts/validate_prompts.py` (Python adds the
# script's own directory, not the CWD, to sys.path[0]).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

from prompts._schema import PromptYAML  # noqa: E402


def validate(paths: list[Path]) -> int:
    failed = 0
    for root in paths:
        for p in sorted(root.glob("*.yaml")):
            if p.name.startswith("_"):
                continue
            try:
                data = yaml.safe_load(p.read_text())
                name_hint = data.get("name", p.stem) if isinstance(data, dict) else p.stem
                PromptYAML(**data)
                if data.get("name") != p.stem:
                    raise ValueError(f"filename stem {p.stem!r} != name field {data.get('name')!r}")
                print(f"[ok]   {p.name}")
            except Exception as e:  # noqa: BLE001
                print(f"[fail] {p.name} (name={name_hint!r}): {e}", file=sys.stderr)
                failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]] or [Path("prompts")]
    sys.exit(validate(args))
