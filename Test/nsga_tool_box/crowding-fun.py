import pytest
import numpy as np
import sys
sys.path.append('../../Source')
import nsga_tool_box

# https://medium.com/@rossleecooloh/optimization-algorithm-nsga-ii-and-python-package-deap-fca0be6b2ffc
# will use to compare the results to our implementation
def CrowdingDist(fitness=None):
    """
    :param fitness: A list of fitness values
    :return: A list of crowding distances of chrmosomes

    The crowding-distance computation requires sorting the population according to each objective function value
    in ascending order of magnitude. Thereafter, for each objective function, the boundary solutions (solutions with smallest and largest function values)
    are assigned an infinite distance value. All other intermediate solutions are assigned a distance value equal to
    the absolute normalized difference in the function values of two adjacent solutions.
    """

    # initialize list: [0.0, 0.0, 0.0, ...]
    distances = [0.0] * len(fitness)
    crowd = [(f_value, i) for i, f_value in enumerate(fitness)]  # create keys for fitness values

    n_obj = len(fitness[0])

    for i in range(n_obj):  # calculate for each objective
        crowd.sort(key=lambda element: element[0][i])
        # After sorting,  boundary solutions are assigned Inf
        # crowd: [([obj_1, obj_2, ...], i_0), ([obj_1, obj_2, ...], i_1), ...]
        distances[crowd[0][1]] = float("Inf")
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:  # If objective values are same, skip this loop
            continue
        # normalization (max - min) as Denominator
        norm = float(crowd[-1][0][i] - crowd[0][0][i])
        # crowd: [([obj_1, obj_2, ...], i_0), ([obj_1, obj_2, ...], i_1), ...]
        # calculate each individual's Crowding Distance of i th objective
        # technique: shift the list and zip
        for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (next[0][i] - prev[0][i]) / norm  # sum up the distance of ith individual along each of the objectives

    return distances

def test_basic_case():
    population = np.array([
        [3.0, 5.0],
        [4.0, 3.0],
        [2.0, 6.0],
        [5.0, 2.0],
    ], dtype=np.float32)
    objectives = np.int32(2)
    expected_distances = np.array([17/12, 17/12, np.inf, np.inf], dtype=np.float32)
    result = nsga_tool_box.CrowdingDistance(population, objectives)
    assert len(result) == len(expected_distances)
    for res, exp in zip(result, expected_distances):
        if np.isinf(exp):
            assert np.isinf(res)
        else:
            assert pytest.approx(res, 0.00001) == exp

def test_single_objective():
    population = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0]
    ], dtype=np.float32)
    objectives = np.int32(1)
    expected_distances = np.array([np.inf, 0.5, 0.5, 0.5, np.inf], dtype=np.float32)
    result = nsga_tool_box.CrowdingDistance(population, objectives)
    assert len(result) == len(expected_distances)
    for res, exp in zip(result, expected_distances):
        if np.isinf(exp):
            assert np.isinf(res)
        else:
            assert pytest.approx(res, 0.00001) == exp

def test_identical_objectives():
    population = np.array([
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0]
    ], dtype=np.float32)
    objectives = np.int32(2)
    expected_distances = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    result = nsga_tool_box.CrowdingDistance(population, objectives)
    assert len(result) == len(expected_distances)
    for res, exp in zip(result, expected_distances):
        if np.isinf(exp):
            assert np.isinf(res)
        else:
            assert pytest.approx(res, 0.00001) == exp

# large loop to test accuracy of our implementation
def test_loop_basic_case():
    rng = np.random.default_rng(1)

    for _ in range(10000):
        pop_size = rng.integers(2, 300)
        scores_l = [list(rng.random(size=pop_size, dtype=np.float32))]
        expected_distances = CrowdingDist(scores_l)
        results = nsga_tool_box.CrowdingDistance(np.array([np.array(score, dtype=np.float32) for score in scores_l]), np.int32(pop_size))

        assert len(results) == len(expected_distances)