#####################################################################################################
#
# Class dedicated to keeping track of the SNP's and their importance in regard to the metric used to
# measure their importance. For example, if using r^2, we want a larger average score.
#
# Python 3.12.4: conda activate star-epi-pre
#####################################################################################################

import numpy as np
from typeguard import typechecked
from typing import List
import numpy.typing as npt
from typing import List, Tuple
import pandas as pd

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

# data types for this work

# chromosome number
chrom_num_t = np.uint8
# chromosome snp position
chrom_pos_t = np.uint32
# hub bin number
hub_bin_t = np.uint16
# data type for hub
hub_data_t = np.float32

# expected keys for hub
hub_sum_key = 'sum'
hub_cnt_key = 'cnt'
hub_bin_key = 'bin'

@typechecked # for debugging purposes
class SNPhub:
    def __init__(self, file_path: str, bin_size: np.uint16):
        # get pandas dataframe snp names without loading all data
        self.snp_names = pd.read_csv(file_path, nrows=0, index_col=0).columns.tolist()
        # we don't care about the response variable 'y'
        self.snp_names.remove('y')
        # how big are bins?
        self.bin_size = bin_size

    # generate bins for the SNPs
    # bins will help recommend potential SNPs to be included in the model
    def GenerateBins(self, ) -> None:
        # quick checks
        assert len(self.snp_names) > 0
        assert self.bin_size > 0

        # collect all chromosomes and sort snp postions
        sorted_snps = {}
        for snp in self.snp_names:
            # split header into chromosome and position
            chrom, pos = self.GenerateChromPos(snp)

            # if we get a new chromosome, intialize a new bin for it
            if chrom not in sorted_snps:
                sorted_snps[chrom] = [pos]
            else:
                sorted_snps[chrom].append(pos)


        # sort the positions and assigen them to a bin
        self.bins = {}
        for chrom, pos in sorted_snps.items():
            sorted_pos = np.sort(np.array(pos, dtype=chrom_pos_t), kind='mergesort')

            for p in sorted_pos:
                # if we get a new chromosome, intialize a new bin for it
                if chrom not in self.bins:
                    self.bins[chrom] = [[p]]
                else:
                    # check if the current bin is full
                    if len(self.bins[chrom][-1]) == self.bin_size:
                        self.bins[chrom].append([p])
                    else:
                        self.bins[chrom][-1].append(p)

        # cast all bins to numpy arrays for efficiency
        for chrom, bins in self.bins.items():
            self.bins[chrom] = [np.array(b, dtype=chrom_pos_t) for b in bins]

        # make sure all SNPs are accounted for
        assert len(self.snp_names) == self.CountBinObjects()

    def GenerateHub(self) -> None:
        # create container of bins
        self.hub = {}

        # sort bins
        for chrom, bin in self.bins.items():
            cur_bin = 0
            sorted_bin = sorted(bin)

            for i, pos in enumerate(sorted_bin):
                # update bin number?
                if i % self.bin_size == 0:
                    cur_bin += 1

                # add to hub
                if chrom not in self.hub:
                    self.hub[chrom] = {pos: {hub_sum_key: hub_data_t(0.0), hub_cnt_key: hub_data_t(0.0), hub_bin_key:  hub_bin_t(cur_bin)}}
                else:
                    self.hub[chrom][pos] = {hub_sum_key: hub_data_t(0.0), hub_cnt_key: hub_data_t(0.0), hub_bin_key:  hub_bin_t(cur_bin)}


        # make sure hub contains all headers
        assert len(self.snp_names) == sum([len(posi) for posi in self.hub.values()])

    # get the single SNP's value for some given key
    def GetDataSNP(self, snp_name: str, key: str) -> hub_data_t | hub_bin_t:
        # split header into chromosome and position
        chrom, pos = self.GenerateChromPos(snp_name)

        # quick checks
        assert chrom in self.hub
        assert pos in self.hub[chrom]
        assert key in self.hub[chrom][pos]

        # return requested data
        return self.hub[chrom][pos][key]

    # update the SNP's value for some given metric value
    def UpdateSNP(self, snp_name: str, value: hub_data_t) -> None:
        # split header into chromosome and position
        chrom, pos = self.GenerateChromPos(snp_name)

        # quick checks
        assert chrom in self.hub
        assert pos in self.hub[chrom]

        # update values
        self.hub[chrom][pos][hub_sum_key] += value
        self.hub[chrom][pos][hub_cnt_key] += np.float32(1.0)

    # helper to generate chromosome number and snp position
    def GenerateChromPos(self, snp_name: str) -> Tuple[chrom_num_t, chrom_pos_t]:
        chrom, pos = snp_name.split('.')
        chrom, pos = chrom_num_t(chrom), chrom_pos_t(pos)
        return chrom, pos

    def PrintBins(self) -> None:
        # print chromosome
        for chrom, bins in self.bins.items():
            print('chrom:',chrom)
            # print position and values
            for i,bin in enumerate(bins):
                print('bin',i,':',bin)
                print('size:',len(bin))
                print()

    def PrintHeaders(self) -> None:
        print('self.snp_names:',self.snp_names)

    def PrintChromosomes(self) -> None:
        print('self.chromosomes:',self.chromosomes)

    # deugging purposes

    def CountBinObjects(self) -> int:
        sum = 0
        for chrom, bins in self.bins.items():
            for bin in bins:
                sum += len(bin)

        return sum