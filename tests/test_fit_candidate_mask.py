import numpy as np
import pytest

from src.fit_candidate_mask import build_exclusion_mask, metric_row


def test_fit_mask_excludes_train_and_validation_responders_without_duplicates():
    candidates = ("a", "b", "c", "d")
    mask = build_exclusion_mask(
        candidates,
        train_responders=("a", "b"),
        validation_responders=("b", "c"),
        test_responders=("d",),
    )
    assert mask.tolist() == [True, True, True, False]


def test_fit_mask_rejects_test_positive_exclusion():
    with pytest.raises(ValueError, match="test responder"):
        build_exclusion_mask(
            ("a", "b"),
            train_responders=("a",),
            validation_responders=("b",),
            test_responders=("b",),
        )


def test_removing_high_ranked_validation_responder_improves_test_rank():
    candidates = ("val", "test", "other")
    scores = np.asarray([3.0, 2.0, 1.0])
    old = metric_row(
        scores,
        candidates,
        ("test",),
        np.asarray([False, False, False]),
    )
    fit = metric_row(
        scores,
        candidates,
        ("test",),
        np.asarray([True, False, False]),
    )
    assert old["best_positive_rank"] == 2
    assert fit["best_positive_rank"] == 1
    assert old["hit@1"] == 0.0
    assert fit["hit@1"] == 1.0
