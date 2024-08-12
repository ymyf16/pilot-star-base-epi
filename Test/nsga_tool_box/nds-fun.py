# Python 3.12.4: conda activate star-epi-pre

import pytest
import numpy as np
import sys

sys.path.append('../../Source')
import nsga_tool_box

def test_non_dominated_sorting():
    # Test case 1
    objectives = np.array([
        [1, 2],  # Solution 0
        [2, 1],  # Solution 1
        [1.5, 1.5],  # Solution 2
        [3, 3],  # Solution 3
        [0, 0]   # Solution 4
    ], dtype=np.float32)
    fronts, ranks = nsga_tool_box.NonDominatedSorting(objectives)
    true_rank = np.array([1, 1, 1, 0, 2], dtype=np.uint16)
    true_fronts = [np.array([3], dtype=np.uint16), np.array([0, 1, 2], dtype=np.uint16), np.array([4], dtype=np.uint16)]

    assert ranks.shape == true_rank.shape
    assert np.all(true_rank == ranks)

    assert len(fronts) == len(true_fronts)
    assert all(np.all(x == y) for x, y in zip(fronts, true_fronts))

def test_non_dominated_sorting_basic():
    population = np.array([
        [1, 4],
        [2, 3],
        [3, 2],
        [4, 1]
    ], dtype=np.float32)
    true_fronts = [np.array([0,1,2,3], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, true_fronts))

    # make sure ranks are correct
    assert np.all(ranks == 0)

def test_non_dominated_sorting_two_elements():
    population = np.array([
        [1, 2],
        [2, 1]
    ], dtype=np.float32)
    true_fronts = [np.array([0, 1], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, true_fronts))

    # make sure ranks are correct
    assert np.all(ranks == 0)

def test_non_dominated_sorting_tied_elements():
    population = np.array([
        [1, 2],
        [1, 2],
        [2, 1]
    ], dtype=np.float32)
    expected_fronts = [np.array([0, 1, 2], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == 0)

def test_non_dominated_sorting_dominated_elements():
    population = np.array([
        [1, 2],
        [2, 3],
        [3, 1],
        [4, 4]
    ], dtype=np.float32)
    expected_fronts = [np.array([3], dtype=np.uint16), np.array([1,2], dtype=np.uint16), np.array([0], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == np.array([2,1,1,0], dtype=np.uint16))

def test_non_dominated_sorting_large_population():
    population = np.array([
        [1, 10],
        [2, 9],
        [3, 8],
        [4, 7],
        [5, 6],
        [6, 5],
        [7, 4],
        [8, 3],
        [9, 2],
        [10, 1]
    ], dtype=np.float32)
    expected_fronts = [np.array([x for x in range(10)], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == 0)

def test_non_dominated_sorting_all_equal():
    population = np.array([
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1]
    ], dtype=np.float32)
    expected_fronts = [np.array([0, 1, 2, 3], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == 0)

def test_non_dominated_sorting_complex():
    population = np.array([
        [1, 2],
        [2, 1],
        [2, 2],
        [1, 3],
        [3, 1],
        [3, 3],
        [4, 4],
        [5, 0]
    ], dtype=np.float32)
    expected_fronts = [np.array([6, 7], dtype=np.uint16), np.array([5], dtype=np.uint16), np.array([2, 3, 4], dtype=np.uint16), np.array([0, 1], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == np.array([3,3,2,2,2,1,0,0], dtype=np.uint16))

def test_non_dominated_sorting_all_same_distance():
    population = np.array([
        [1, 2],
        [2, 1],
        [3, 3],
        [4, 4]
    ], dtype=np.float32)
    expected_fronts = [np.array([3], dtype=np.uint16), np.array([2], dtype=np.uint16), np.array([0,1], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == np.array([2,2,1,0], dtype=np.uint16))

def test_non_dominated_sorting_all_distances_zero():
    population = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0]
    ], dtype=np.float32)
    expected_fronts = [np.array([0, 1, 2, 3], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == 0)

def test_non_dominated_sorting_distances_increasing():
    population = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4]
    ], dtype=np.float32)
    expected_fronts = [np.array([3], dtype=np.uint16), np.array([2], dtype=np.uint16), np.array([1], dtype=np.uint16), np.array([0], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == np.array([3,2,1,0], dtype=np.uint16))

def test_non_dominated_sorting_distances_decreasing():
    population = np.array([
        [4, 4],
        [3, 3],
        [2, 2],
        [1, 1]
    ], dtype=np.float32)
    expected_fronts = [np.array([0], dtype=np.uint16), np.array([1], dtype=np.uint16), np.array([2], dtype=np.uint16), np.array([3], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == np.array([0,1,2,3], dtype=np.uint16))

def test_non_dominated_sorting_single_element_fronts():
    population = np.array([
        [1, 2],
        [2, 1],
        [3, 3],
        [4, 4]
    ], dtype=np.float32)
    expected_fronts = [np.array([3], dtype=np.uint16), np.array([2], dtype=np.uint16), np.array([0,1], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == np.array([2,2,1,0], dtype=np.uint16))

def test_non_dominated_sorting_single_element_different_distances():
    population = np.array([
        [1, 3],
        [2, 2],
        [3, 1]
    ], dtype=np.float32)
    expected_fronts = [np.array([0, 1, 2], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == np.array([0,0,0], dtype=np.uint16))

def test_non_dominated_sorting_more_elements_in_front():
    population = np.array([
        [1, 4],
        [2, 3],
        [3, 2],
        [4, 1],
        [5, 5]
    ], dtype=np.float32)
    expected_fronts = [np.array([4], dtype=np.uint16), np.array([0,1,2,3], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == np.array([1,1,1,1,0], dtype=np.uint16))

def test_non_dominated_sorting_large_distances():
    population = np.array([
        [1000, 2000],
        [2000, 1000],
        [3000, 3000],
        [4000, 4000]
    ], dtype=np.float32)
    expected_fronts = [np.array([0, 1], dtype=np.uint16), np.array([2], dtype=np.uint16), np.array([3], dtype=np.uint16)][::-1]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == np.array([2,2,1,0], dtype=np.uint16))

def test_non_dominated_sorting_negative_distances():
    population = np.array([
        [-1, -2],
        [-2, -1],
        [-3, -3],
        [-4, -4]
        ], dtype=np.float32)
    expected_fronts = [np.array([0, 1], dtype=np.uint16), np.array([2], dtype=np.uint16), np.array([3], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == np.array([0,0,1,2], dtype=np.uint16))

def test_non_dominated_sorting_mixed_distances():
    population = np.array([
        [1, -2],
        [-2, 1],
        [3, 3],
        [-4, -4]
        ], dtype=np.float32)
    expected_fronts = [np.array([2], dtype=np.uint16), np.array([0,1], dtype=np.uint16), np.array([3], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == np.array([1,1,0,2], dtype=np.uint16))

def test_non_dominated_sorting_small_n():
    population = np.array([
        [1, 2],
        [2, 1],
        [3, 3],
        [4, 4]
        ], dtype=np.float32)
    expected_fronts = [np.array([0, 1], dtype=np.uint16), np.array([2], dtype=np.uint16), np.array([3], dtype=np.uint16)][::-1]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == np.array([2,2,1,0], dtype=np.uint16))

def test_non_dominated_sorting_random_order():
    population = np.array([
        [3, 1],
        [1, 2],
        [2, 3],
        [4, 0],
        [0, 4]
        ], dtype=np.float32)
    expected_fronts = [np.array([0,2,3,4], dtype=np.uint16), np.array([1], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == np.array([0,1,0,0,0], dtype=np.uint16))

def test_non_dominated_sorting_large_population():
    population = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5],
        [6, 6],
        [7, 7],
        [8, 8],
        [9, 9],
        [10, 10]
        ], dtype=np.float32)
    expected_fronts = [np.array([0], dtype=np.uint16), np.array([1], dtype=np.uint16), np.array([2], dtype=np.uint16), np.array([3], dtype=np.uint16), np.array([4], dtype=np.uint16), np.array([5], dtype=np.uint16), np.array([6], dtype=np.uint16), np.array([7], dtype=np.uint16), np.array([8], dtype=np.uint16), np.array([9], dtype=np.uint16)][::-1]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == np.array(list(reversed([x for x in range(len(population))])), dtype=np.uint16))

def test_non_dominated_sorting_all_elements_same_front():
    population = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5]
        ], dtype=np.float32)
    expected_fronts = reversed([np.array([0], dtype=np.uint16), np.array([1], dtype=np.uint16), np.array([2], dtype=np.uint16), np.array([3], dtype=np.uint16), np.array([4], dtype=np.uint16)])
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == np.array(list(reversed([0,1,2,3,4])), dtype=np.uint16))

def test_non_dominated_sorting_large_values():
    population = np.array([
        [1000, 2000],
        [2000, 1000],
        [3000, 3000],
        [4000, 4000]
        ], dtype=np.float32)
    expected_fronts = [np.array([3], dtype=np.uint16), np.array([2], dtype=np.uint16), np.array([0, 1], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == np.array([2,2,1,0], dtype=np.uint16))

def test_non_dominated_sorting_single_element_large_distance():
    population = np.array([
        [1e6, 2e6]
        ], dtype=np.float32)
    expected_fronts = [np.array([0], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == 0)

def test_non_dominated_sorting_single_element_negative_distance():
    population = np.array([
        [-1e6, -2e6]
        ], dtype=np.float32)
    expected_fronts = [np.array([0], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == 0)

def test_non_dominated_sorting_tied_elements_with_different_values():
    population = np.array([
        [1, 2],
        [2, 1],
        [1, 2],
        [2, 1]
        ], dtype=np.float32)
    expected_fronts = [np.array([0, 1, 2, 3], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == 0)

def test_non_dominated_sorting_elements_with_equal_distances():
    population = np.array([
        [1, 1],
        [1, 1],
        [2, 2],
        [2, 2]
        ], dtype=np.float32)
    expected_fronts = [np.array([2, 3], dtype=np.uint16), np.array([0,1], dtype=np.uint16)]
    fronts, ranks = nsga_tool_box.NonDominatedSorting(population)

    # make sure fronts are correct
    assert all(np.all(x == y) for x, y in zip(fronts, expected_fronts))

    # make sure ranks are correct
    assert np.all(ranks == np.array([1,1,0,0], dtype=np.uint16))