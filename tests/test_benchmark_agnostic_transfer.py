import inspect

import numpy as np
import pytest


def test_public_apis_do_not_accept_split_or_dataset_metadata():
    from src import benchmark_agnostic_transfer as transfer

    for function in (
        transfer.witness_availability,
        transfer.transfer_first_auto,
        transfer.availability_state,
    ):
        parameters = inspect.signature(function).parameters
        assert "split" not in parameters
        assert "dataset" not in parameters


def test_witness_availability_uses_only_boolean_witness_and_finite_score():
    from src.benchmark_agnostic_transfer import witness_availability

    s_witness = np.array([[True, True, False, False]])
    c_witness = np.array([[True, False, True, False]])
    s_scores = np.array([[1.0, np.nan, 3.0, np.nan]])
    c_scores = np.array([[np.nan, 2.0, 3.0, 4.0]])

    s_available, c_available = witness_availability(
        s_witness, c_witness, s_scores, c_scores
    )

    np.testing.assert_array_equal(s_available, [[True, False, False, False]])
    np.testing.assert_array_equal(c_available, [[False, False, True, False]])
    assert s_available.dtype == np.bool_
    assert c_available.dtype == np.bool_


@pytest.mark.parametrize(
    "arguments",
    [
        (
            np.ones((1, 2), dtype=np.int8),
            np.ones((1, 2), dtype=bool),
            np.ones((1, 2)),
            np.ones((1, 2)),
        ),
        (
            np.ones((1, 2), dtype=bool),
            np.ones((1, 2), dtype=bool),
            np.array([["1", "2"]]),
            np.ones((1, 2)),
        ),
        (
            np.ones((2,), dtype=bool),
            np.ones((2,), dtype=bool),
            np.ones((2,)),
            np.ones((2,)),
        ),
        (
            np.ones((1, 2), dtype=bool),
            np.ones((1, 3), dtype=bool),
            np.ones((1, 2)),
            np.ones((1, 2)),
        ),
        (
            np.empty((0, 2), dtype=bool),
            np.empty((0, 2), dtype=bool),
            np.empty((0, 2)),
            np.empty((0, 2)),
        ),
        (
            np.ones((1, 2), dtype=bool),
            np.ones((1, 2), dtype=bool),
            np.array([[1.0, np.inf]]),
            np.ones((1, 2)),
        ),
    ],
)
def test_witness_availability_rejects_invalid_inputs(arguments):
    from src.benchmark_agnostic_transfer import witness_availability

    with pytest.raises(ValueError):
        witness_availability(*arguments)


def test_transfer_first_auto_routes_all_four_availability_states_exactly():
    from src.benchmark_agnostic_transfer import transfer_first_auto

    s = np.array([[10.0, 20.0, 30.0, 40.0]])
    c = np.array([[1.0, 2.0, 3.0, 4.0]])
    dstar = np.array([[0.1, 0.2, 0.3, 0.4]])
    s_available = np.array([[False, False, True, True]])
    c_available = np.array([[False, True, False, True]])

    scores, available = transfer_first_auto(
        s, c, dstar, s_available, c_available, 0.25
    )

    np.testing.assert_allclose(scores, [[0.1, 2.0, 30.0, 13.0]])
    np.testing.assert_array_equal(available, [[False, True, True, True]])


@pytest.mark.parametrize(
    "replacement,index",
    [
        (np.ones((2, 2)), 0),
        (np.array([[np.inf, 2.0]]), 1),
        (np.ones((1, 2), dtype=np.int8), 3),
    ],
)
def test_transfer_first_auto_preserves_existing_validation(replacement, index):
    from src.benchmark_agnostic_transfer import transfer_first_auto

    arguments = [
        np.ones((1, 2)),
        np.ones((1, 2)),
        np.ones((1, 2)),
        np.ones((1, 2), dtype=bool),
        np.ones((1, 2), dtype=bool),
        0.25,
    ]
    arguments[index] = replacement

    with pytest.raises(ValueError):
        transfer_first_auto(*arguments)


def test_availability_state_encodes_all_four_states_as_int8():
    from src.benchmark_agnostic_transfer import availability_state

    s_available = np.array([[False, False, True, True]])
    c_available = np.array([[False, True, False, True]])

    state = availability_state(s_available, c_available)

    np.testing.assert_array_equal(state, [[0, 1, 2, 3]])
    assert state.dtype == np.int8


@pytest.mark.parametrize(
    "s_available,c_available",
    [
        (np.ones((2,), dtype=bool), np.ones((2,), dtype=bool)),
        (np.ones((1, 2), dtype=bool), np.ones((1, 3), dtype=bool)),
        (np.ones((1, 2), dtype=np.int8), np.ones((1, 2), dtype=bool)),
    ],
)
def test_availability_state_rejects_invalid_masks(s_available, c_available):
    from src.benchmark_agnostic_transfer import availability_state

    with pytest.raises(ValueError):
        availability_state(s_available, c_available)
