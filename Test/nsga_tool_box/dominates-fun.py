import pytest
import numpy as np
import sys

sys.path.append('../../Source')
import nsga_tool_box

def test_dominates_true():
    # Simple cases
    assert nsga_tool_box.Dominates(np.array([3, 5], dtype=np.float32), np.array([2, 4], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([1, 2], dtype=np.float32), np.array([0, 1], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([4, 4], dtype=np.float32), np.array([4, 3], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([6, 7, 8], dtype=np.float32), np.array([6, 6, 7], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([5, 5], dtype=np.float32), np.array([4, 4], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([10, 20, 30], dtype=np.float32), np.array([5, 15, 25], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([7, 7, 7], dtype=np.float32), np.array([6, 6, 6], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([8, 9, 10], dtype=np.float32), np.array([7, 8, 9], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([15, 15, 15], dtype=np.float32), np.array([10, 10, 10], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([12, 13], dtype=np.float32), np.array([11, 12], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([9, 14], dtype=np.float32), np.array([8, 13], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([20, 25, 30], dtype=np.float32), np.array([15, 20, 25], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([2, 8], dtype=np.float32), np.array([1, 7], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([3, 6, 9], dtype=np.float32), np.array([2, 5, 8], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([14, 18, 22], dtype=np.float32), np.array([10, 17, 21], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([30, 40, 50], dtype=np.float32), np.array([20, 30, 40], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([33, 44, 55], dtype=np.float32), np.array([22, 33, 44], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([100, 200], dtype=np.float32), np.array([50, 150], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([45, 55, 65], dtype=np.float32), np.array([35, 45, 55], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([150, 250, 350], dtype=np.float32), np.array([100, 200, 300], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([80, 85, 90], dtype=np.float32), np.array([70, 80, 85], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([99, 101, 105], dtype=np.float32), np.array([98, 100, 104], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([70, 90, 110], dtype=np.float32), np.array([60, 80, 100], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([120, 140, 160], dtype=np.float32), np.array([110, 130, 150], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([200, 300, 400], dtype=np.float32), np.array([150, 250, 350], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([500, 600, 700], dtype=np.float32), np.array([400, 500, 600], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([250, 350, 450], dtype=np.float32), np.array([200, 300, 400], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([55, 65, 75], dtype=np.float32), np.array([45, 55, 65], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([99, 199, 299], dtype=np.float32), np.array([89, 189, 289], dtype=np.float32)) == True
    assert nsga_tool_box.Dominates(np.array([85, 95, 105], dtype=np.float32), np.array([75, 85, 95], dtype=np.float32)) == True

def test_nsga_dominates_false():
    # Simple cases
    assert nsga_tool_box.Dominates(np.array([2, 4], dtype=np.float32), np.array([3, 5], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([0, 1], dtype=np.float32), np.array([1, 2], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([4, 3], dtype=np.float32), np.array([4, 4], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([6, 6, 7], dtype=np.float32), np.array([6, 7, 8], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([5, 4], dtype=np.float32), np.array([6, 4], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([7, 6], dtype=np.float32), np.array([7, 7], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([1, 2, 3], dtype=np.float32), np.array([2, 2, 2], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([5, 5, 5], dtype=np.float32), np.array([6, 4, 5], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([8, 6, 7], dtype=np.float32), np.array([8, 7, 7], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([2, 3], dtype=np.float32), np.array([3, 3], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([10, 9], dtype=np.float32), np.array([10, 10], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([5, 7, 9], dtype=np.float32), np.array([6, 7, 8], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([12, 15, 18], dtype=np.float32), np.array([13, 14, 17], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([7, 8], dtype=np.float32), np.array([8, 8], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([19, 20, 21], dtype=np.float32), np.array([20, 20, 21], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([30, 35, 40], dtype=np.float32), np.array([31, 34, 41], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([25, 25, 25], dtype=np.float32), np.array([26, 24, 26], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([2, 2, 3], dtype=np.float32), np.array([3, 2, 2], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([15, 10], dtype=np.float32), np.array([20, 5], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([14, 13], dtype=np.float32), np.array([15, 12], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([1, 1, 1], dtype=np.float32), np.array([2, 1, 1], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([3, 3, 2], dtype=np.float32), np.array([4, 2, 2], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([7, 9], dtype=np.float32), np.array([9, 7], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([18, 17, 16], dtype=np.float32), np.array([19, 16, 17], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([20, 10, 5], dtype=np.float32), np.array([21, 9, 6], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([30, 25], dtype=np.float32), np.array([31, 24], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([12, 13, 11], dtype=np.float32), np.array([13, 12, 10], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([15, 16, 14], dtype=np.float32), np.array([16, 15, 13], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([5, 6, 5], dtype=np.float32), np.array([6, 5, 6], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([50, 40, 30], dtype=np.float32), np.array([51, 39, 31], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([70, 60], dtype=np.float32), np.array([71, 59], dtype=np.float32)) == False

def test_nsga_dominates_equal():
    # Simple cases
    assert nsga_tool_box.Dominates(np.array([2, 4], dtype=np.float32), np.array([2, 4], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([1, 2, 3], dtype=np.float32), np.array([1, 2, 3], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([5, 5], dtype=np.float32), np.array([5, 5], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([0, 0], dtype=np.float32), np.array([0, 0], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([10, 20, 30], dtype=np.float32), np.array([10, 20, 30], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([7, 7, 7], dtype=np.float32), np.array([7, 7, 7], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([3, 3, 3], dtype=np.float32), np.array([3, 3, 3], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([100, 200, 300], dtype=np.float32), np.array([100, 200, 300], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([50, 50, 50], dtype=np.float32), np.array([50, 50, 50], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([20, 30, 40], dtype=np.float32), np.array([20, 30, 40], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([8, 8, 8], dtype=np.float32), np.array([8, 8, 8], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([6, 6], dtype=np.float32), np.array([6, 6], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([15, 25, 35], dtype=np.float32), np.array([15, 25, 35], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([12, 12, 12], dtype=np.float32), np.array([12, 12, 12], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([7, 14, 21], dtype=np.float32), np.array([7, 14, 21], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([10, 10, 10], dtype=np.float32), np.array([10, 10, 10], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([9, 9, 9], dtype=np.float32), np.array([9, 9, 9], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([4, 4, 4], dtype=np.float32), np.array([4, 4, 4], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([14, 28, 42], dtype=np.float32), np.array([14, 28, 42], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([6, 12, 18], dtype=np.float32), np.array([6, 12, 18], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([11, 11, 11], dtype=np.float32), np.array([11, 11, 11], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([2, 2, 2], dtype=np.float32), np.array([2, 2, 2], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([25, 50, 75], dtype=np.float32), np.array([25, 50, 75], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([100, 100], dtype=np.float32), np.array([100, 100], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([33, 33, 33], dtype=np.float32), np.array([33, 33, 33], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([8, 16, 24], dtype=np.float32), np.array([8, 16, 24], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([20, 20, 20], dtype=np.float32), np.array([20, 20, 20], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([18, 18, 18], dtype=np.float32), np.array([18, 18, 18], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([5, 10, 15], dtype=np.float32), np.array([5, 10, 15], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([9, 18, 27], dtype=np.float32), np.array([9, 18, 27], dtype=np.float32)) == False

def test_nsga_dominates_mixed():
    # Simple cases
    assert nsga_tool_box.Dominates(np.array([1, 3], dtype=np.float32), np.array([2, 2], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([3, 1], dtype=np.float32), np.array([2, 2], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([5, 2], dtype=np.float32), np.array([3, 4], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([2, 5], dtype=np.float32), np.array([4, 3], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([1, 5, 3], dtype=np.float32), np.array([2, 3, 4], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([6, 2, 4], dtype=np.float32), np.array([5, 3, 4], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([4, 4, 5], dtype=np.float32), np.array([4, 5, 4], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([9, 5, 6], dtype=np.float32), np.array([10, 4, 7], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([8, 3, 9], dtype=np.float32), np.array([7, 4, 8], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([1, 2, 3], dtype=np.float32), np.array([3, 2, 1], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([3, 4, 5], dtype=np.float32), np.array([5, 4, 3], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([10, 20, 15], dtype=np.float32), np.array([15, 20, 10], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([6, 7, 8], dtype=np.float32), np.array([8, 7, 6], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([5, 10, 5], dtype=np.float32), np.array([10, 5, 10], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([2, 4, 6], dtype=np.float32), np.array([6, 4, 2], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([3, 3, 3], dtype=np.float32), np.array([2, 4, 2], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([7, 8, 9], dtype=np.float32), np.array([9, 8, 7], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([10, 15, 10], dtype=np.float32), np.array([15, 10, 15], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([5, 5, 5], dtype=np.float32), np.array([6, 4, 6], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([20, 10, 20], dtype=np.float32), np.array([10, 20, 10], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([12, 6, 12], dtype=np.float32), np.array([6, 12, 6], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([8, 9, 7], dtype=np.float32), np.array([9, 7, 9], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([3, 6, 9], dtype=np.float32), np.array([9, 6, 3], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([11, 12, 13], dtype=np.float32), np.array([13, 12, 11], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([2, 8, 2], dtype=np.float32), np.array([8, 2, 8], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([7, 6, 5], dtype=np.float32), np.array([5, 6, 7], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([4, 2, 6], dtype=np.float32), np.array([6, 2, 4], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([15, 10, 15], dtype=np.float32), np.array([10, 15, 10], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([3, 7, 3], dtype=np.float32), np.array([7, 3, 7], dtype=np.float32)) == False
    assert nsga_tool_box.Dominates(np.array([9, 8, 7], dtype=np.float32), np.array([7, 8, 9], dtype=np.float32)) == False