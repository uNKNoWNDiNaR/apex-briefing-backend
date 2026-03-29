from __future__ import annotations

from pathlib import Path


def discover_sd_msgs(sd_msgs_root: str | Path) -> dict[str, str]:
    root = Path(sd_msgs_root)
    discovered: dict[str, str] = {}
    for msg_path in root.rglob('*.msg'):
        rel = msg_path.relative_to(root)
        pkg = rel.parts[0]
        msg_name = msg_path.stem
        key = f'{pkg}/msg/{msg_name}'
        discovered[key] = msg_path.read_text()
    return discovered
