# Python 3.12.4: conda activate star-epi-pre; python -m pytest

import pytest
import numpy as np
import sys

sys.path.append('../../Source')
import snp_hub

test_file_dir = '~/Desktop/Repositories/pilot-star-base-epi/pruned_ratdata_bmitail_onSNPnum.csv'

def test_intialize_class():
    hub = snp_hub.SNPhub(test_file_dir, np.uint16(100))
    assert True

def test_generate_hub():
    hub = snp_hub.SNPhub(test_file_dir, np.uint16(10))
    hub.GenerateBins()
    hub.PrintBins()

    assert False