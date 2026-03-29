from __future__ import annotations

import pandas as pd


def merge_topics(base_df: pd.DataFrame, extra_frames: dict[str, pd.DataFrame], tolerance_ms: int) -> pd.DataFrame:
    merged = base_df.sort_values('timestamp_ns').reset_index(drop=True)
    tolerance_ns = int(tolerance_ms * 1_000_000)
    for alias, frame in extra_frames.items():
        if frame.empty:
            continue
        other = frame.sort_values('timestamp_ns').drop_duplicates('timestamp_ns', keep='last').reset_index(drop=True)
        rename = {col: f'{alias}__{col}' for col in other.columns if col != 'timestamp_ns'}
        other = other.rename(columns=rename)
        merged = pd.merge_asof(
            merged,
            other,
            on='timestamp_ns',
            direction='nearest',
            tolerance=tolerance_ns,
        )
    return merged
