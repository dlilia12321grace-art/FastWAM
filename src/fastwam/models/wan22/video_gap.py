from __future__ import annotations


def should_refresh_video_cache(
    *,
    chunk_index: int,
    video_gap: int,
    cache_available: bool,
) -> bool:
    """Decide whether the current action chunk must refresh Video DiT K/V."""
    if chunk_index < 0:
        raise ValueError(f"`chunk_index` must be non-negative, got {chunk_index}.")
    if video_gap <= 0:
        raise ValueError(f"`video_gap` must be positive, got {video_gap}.")
    return not cache_available or chunk_index % video_gap == 0


def select_dynamic_video_cache_refresh(
    *,
    cache_available: bool,
    image_mse: float | None,
    threshold: float,
    cache_age: int | None,
    max_cache_age: int,
) -> tuple[bool, str]:
    """Route a chunk using current-vs-cached image change."""
    if threshold < 0:
        raise ValueError(f"`threshold` must be non-negative, got {threshold}.")
    if max_cache_age <= 0:
        raise ValueError(
            f"`max_cache_age` must be positive, got {max_cache_age}."
        )
    if not cache_available:
        return True, "first_chunk"
    if image_mse is None or cache_age is None:
        raise ValueError("Available cache requires both image_mse and cache_age.")
    if cache_age <= 0:
        raise ValueError(f"`cache_age` must be positive, got {cache_age}.")
    if cache_age > max_cache_age:
        return True, "max_cache_age"
    if image_mse <= threshold:
        return False, "image_stable"
    return True, "image_changed"
