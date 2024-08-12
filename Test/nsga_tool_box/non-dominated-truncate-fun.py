import pytest
import numpy as np
import sys
sys.path.append('../../Source')
import nsga_tool_box

def test_non_dominated_truncate_basic():
    fronts = [np.array([0, 1, 2], dtype=np.uint16), np.array([3, 4], dtype=np.uint16)]
    distances = np.array([0.5, 0.2, 0.1, 0.4, 0.3], dtype=np.float32)
    N = np.uint16(5)
    expected_output = np.array([0, 1, 2, 3, 4], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_crowding_distance():
    fronts = [np.array([0, 1], dtype=np.uint16), np.array([2, 3, 4], dtype=np.uint16)]
    distances = np.array([0.5, 0.2, 0.1, 0.4, 0.3], dtype=np.float32)
    N = np.uint16(4)
    expected_output = np.array([0, 1, 3, 4], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_not_enough_fronts():
    fronts = [np.array([0, 1], dtype=np.uint16), np.array([2, 3], dtype=np.uint16)]
    distances = np.array([0.5, 0.2, 0.1, 0.4], dtype=np.float32)
    N = np.uint16(3)
    expected_output = np.array([0, 1, 3], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_large_n():
    fronts = [np.array([0, 1, 2], dtype=np.uint16), np.array([3, 4], dtype=np.uint16)]
    distances = np.array([0.5, 0.2, 0.1, 0.4, 0.3], dtype=np.float32)
    N = np.uint16(6)  # This is more than the number of elements in fronts
    with pytest.raises(AssertionError):
        nsga_tool_box.NonDominatedTruncate(fronts, distances, N)

def test_non_dominated_truncate_incorrect_distances_length():
    fronts = [np.array([0, 1, 2], dtype=np.uint16), np.array([3, 4], dtype=np.uint16)]
    distances = np.array([0.5, 0.2, 0.1, 0.4], dtype=np.float32)  # Incorrect length
    N = np.uint16(5)
    with pytest.raises(AssertionError):
        nsga_tool_box.NonDominatedTruncate(fronts, distances, N)

def test_non_dominated_truncate_single_front():
    fronts = [np.array([0, 1, 2, 3, 4], dtype=np.uint16)]
    distances = np.array([0.5, 0.2, 0.1, 0.4, 0.3], dtype=np.float32)
    N = np.uint16(5)
    expected_output = np.array([0, 1, 2, 3, 4], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_exact_fit():
    fronts = [np.array([0, 1, 2], dtype=np.uint16), np.array([3, 4], dtype=np.uint16)]
    distances = np.array([0.5, 0.2, 0.1, 0.4, 0.3], dtype=np.float32)
    N = np.uint16(5)
    expected_output = np.array([0, 1, 2, 3, 4], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_partial_fill():
    fronts = [np.array([0, 1, 2], dtype=np.uint16), np.array([3, 4, 5], dtype=np.uint16)]
    distances = np.array([0.5, 0.2, 0.1, 0.4, 0.3, 0.6], dtype=np.float32)
    N = np.uint16(4)
    expected_output = np.array([0, 1, 2, 5], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_tiebreaker():
    fronts = [np.array([0, 1, 2], dtype=np.uint16), np.array([3, 4, 5], dtype=np.uint16)]
    distances = np.array([0.5, 0.5, 0.5, 0.4, 0.4, 0.6], dtype=np.float32)
    N = np.uint16(4)
    expected_output = np.array([0, 1, 2, 5], dtype=np.uint16)  # Tie resolved by ordering in input
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_all_same_distance():
    fronts = [np.array([0, 1, 2, 3, 4], dtype=np.uint16)]
    distances = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    N = np.uint16(5)
    expected_output = np.array([0, 1, 2, 3, 4], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_all_distances_zero():
    fronts = [np.array([0, 1, 2, 3, 4], dtype=np.uint16)]
    distances = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    N = np.uint16(5)
    expected_output = np.array([0, 1, 2, 3, 4], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_distances_increasing():
    fronts = [np.array([0, 1, 2], dtype=np.uint16), np.array([3, 4, 5], dtype=np.uint16)]
    distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    N = np.uint16(4)
    expected_output = np.array([0, 1, 2, 5], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_distances_decreasing():
    fronts = [np.array([0, 1, 2], dtype=np.uint16), np.array([3, 4, 5], dtype=np.uint16)]
    distances = np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    N = np.uint16(4)
    expected_output = np.array([0, 1, 2, 3], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_single_element_fronts():
    fronts = [np.array([0], dtype=np.uint16), np.array([1], dtype=np.uint16), np.array([2], dtype=np.uint16)]
    distances = np.array([0.5, 0.4, 0.3], dtype=np.float32)
    N = np.uint16(3)
    expected_output = np.array([0, 1, 2], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_single_element_different_distances():
    fronts = [np.array([0], dtype=np.uint16), np.array([1], dtype=np.uint16), np.array([2], dtype=np.uint16)]
    distances = np.array([0.3, 0.1, 0.2], dtype=np.float32)
    N = np.uint16(2)
    expected_output = np.array([0, 1], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_more_elements_in_front():
    fronts = [np.array([0, 1, 2, 3, 4, 5], dtype=np.uint16)]
    distances = np.array([0.5, 0.6, 0.1, 0.3, 0.4, 0.2], dtype=np.float32)
    N = np.uint16(4)
    expected_output = np.array([1, 0, 4, 3], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_equal_size_fronts():
    fronts = [np.array([0, 1], dtype=np.uint16), np.array([2, 3], dtype=np.uint16)]
    distances = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    N = np.uint16(3)
    expected_output = np.array([0, 1, 2], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_large_distances():
    fronts = [np.array([0, 1], dtype=np.uint16), np.array([2, 3], dtype=np.uint16)]
    distances = np.array([1000.0, 2000.0, 3000.0, 4000.0], dtype=np.float32)
    N = np.uint16(3)
    expected_output = np.array([0, 1, 3], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_negative_distances():
    fronts = [np.array([0, 1], dtype=np.uint16), np.array([2, 3], dtype=np.uint16)]
    distances = np.array([-1.0, -0.5, -2.0, -1.5], dtype=np.float32)
    N = np.uint16(3)
    expected_output = np.array([0, 1, 3], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_mixed_distances():
    fronts = [np.array([0, 1], dtype=np.uint16), np.array([2, 3], dtype=np.uint16)]
    distances = np.array([1.0, -0.5, 2.0, -1.5], dtype=np.float32)
    N = np.uint16(3)
    expected_output = np.array([0, 1, 2], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_small_n():
    fronts = [np.array([0, 1, 2], dtype=np.uint16), np.array([3, 4, 5], dtype=np.uint16)]
    distances = np.array([0.5, 0.2, 0.1, 0.4, 0.3, 0.6], dtype=np.float32)
    N = np.uint16(1)
    expected_output = np.array([0], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_random_order():
    fronts = [np.array([3, 1, 4, 0, 2], dtype=np.uint16)]
    distances = np.array([0.3, 0.2, 0.4, 0.1, 0.5], dtype=np.float32)
    N = np.uint16(3)
    expected_output = np.array([4, 2, 0], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_large_values():
    fronts = [np.array([0, 1, 2], dtype=np.uint16), np.array([3, 4], dtype=np.uint16)]
    distances = np.array([0.5, 0.4, 0.3, 0.2, np.inf], dtype=np.float32)
    N = np.uint16(4)
    expected_output = np.array([0, 1, 2, 4], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_single_element_front():
    fronts = [np.array([0], dtype=np.uint16)]
    distances = np.array([0.5], dtype=np.float32)
    N = np.uint16(1)
    expected_output = np.array([0], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_with_large_N():
    fronts = [np.array([0, 1, 2], dtype=np.uint16), np.array([3, 4, 5], dtype=np.uint16)]
    distances = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)
    N = np.uint16(6)
    expected_output = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_tie_and_fill():
    fronts = [np.array([0, 1, 2], dtype=np.uint16), np.array([3, 4], dtype=np.uint16)]
    distances = np.array([0.5, 0.5, 0.5, 0.4, 0.3], dtype=np.float32)
    N = np.uint16(5)
    expected_output = np.array([0, 1, 2, 3, 4], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_with_empty_front():
    fronts = [np.array([], dtype=np.uint16), np.array([0, 1, 2], dtype=np.uint16)]
    distances = np.array([0.5, 0.4, 0.3], dtype=np.float32)
    N = np.uint16(3)
    expected_output = np.array([0, 1, 2], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_single_element_large_distance():
    fronts = [np.array([0], dtype=np.uint16)]
    distances = np.array([1e6], dtype=np.float32)
    N = np.uint16(1)
    expected_output = np.array([0], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_single_element_negative_distance():
    fronts = [np.array([0], dtype=np.uint16)]
    distances = np.array([-1e6], dtype=np.float32)
    N = np.uint16(1)
    expected_output = np.array([0], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)

def test_non_dominated_truncate_all_elements_same_front():
    fronts = [np.array([0, 1, 2, 3, 4, 5], dtype=np.uint16)]
    distances = np.array([0.5, 0.4, 0.3, 0.6, 0.2, 0.1], dtype=np.float32)
    N = np.uint16(4)
    expected_output = np.array([3, 0, 1, 2], dtype=np.uint16)
    result = nsga_tool_box.NonDominatedTruncate(fronts, distances, N)
    assert np.array_equal(result, expected_output)