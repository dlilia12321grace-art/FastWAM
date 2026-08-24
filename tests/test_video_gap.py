import pytest

from fastwam.models.wan22.video_gap import (
    select_dynamic_video_cache_refresh,
    should_refresh_video_cache,
)


def test_video_gap_one_refreshes_every_chunk():
    assert all(
        should_refresh_video_cache(
            chunk_index=chunk,
            video_gap=1,
            cache_available=True,
        )
        for chunk in range(5)
    )


def test_video_gap_four_refreshes_periodically():
    decisions = [
        should_refresh_video_cache(
            chunk_index=chunk,
            video_gap=4,
            cache_available=True,
        )
        for chunk in range(9)
    ]
    assert decisions == [True, False, False, False, True, False, False, False, True]


def test_missing_cache_forces_refresh():
    assert should_refresh_video_cache(
        chunk_index=3,
        video_gap=4,
        cache_available=False,
    )


@pytest.mark.parametrize(
    ("chunk_index", "video_gap"),
    [(-1, 2), (0, 0), (0, -1)],
)
def test_video_gap_rejects_invalid_values(chunk_index, video_gap):
    with pytest.raises(ValueError):
        should_refresh_video_cache(
            chunk_index=chunk_index,
            video_gap=video_gap,
            cache_available=True,
        )


def test_dynamic_video_gap_routes_stable_images_to_cache():
    refresh, reason = select_dynamic_video_cache_refresh(
        cache_available=True,
        image_mse=0.01,
        threshold=0.02,
        cache_age=1,
        max_cache_age=1,
    )
    assert not refresh
    assert reason == "image_stable"


def test_dynamic_video_gap_refreshes_changed_or_old_cache():
    assert select_dynamic_video_cache_refresh(
        cache_available=True,
        image_mse=0.03,
        threshold=0.02,
        cache_age=1,
        max_cache_age=1,
    ) == (True, "image_changed")
    assert select_dynamic_video_cache_refresh(
        cache_available=True,
        image_mse=0.0,
        threshold=0.02,
        cache_age=2,
        max_cache_age=1,
    ) == (True, "max_cache_age")


def test_dynamic_video_gap_forces_first_refresh():
    assert select_dynamic_video_cache_refresh(
        cache_available=False,
        image_mse=None,
        threshold=0.02,
        cache_age=None,
        max_cache_age=1,
    ) == (True, "first_chunk")


@pytest.mark.parametrize(
    ("threshold", "max_cache_age"),
    [(-0.1, 1), (0.1, 0)],
)
def test_dynamic_video_gap_rejects_invalid_policy(threshold, max_cache_age):
    with pytest.raises(ValueError):
        select_dynamic_video_cache_refresh(
            cache_available=False,
            image_mse=None,
            threshold=threshold,
            cache_age=None,
            max_cache_age=max_cache_age,
        )
