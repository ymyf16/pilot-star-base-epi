#####################################################################################################
#
# Interface for communicating with both EPI and SNP hubs.
#
#####################################################################################################

import numpy as np
from typing import List, Tuple

from typeguard import typechecked
import numpy.typing as npt


# chromosome number
chrom_num_t = np.uint8
# chromosome snp position
chrom_pos_t = np.uint32
# hub bin number
hub_bin_t = np.uint16
# data type for hub
hub_data_t = np.float32

@typechecked
class GenoHub:
    """
    Interface for communicating with both EPI and SNP hubs.
    We do not allow for the creation of new hubs, only the updating of existing ones.
    Treat as a private class.
    """

    class SNP:
        """
        Hub to keep track of snp weights.
        """
        def __init__(self):
            """
            self.hub: dictionary to hold all snp and values
            assuming that all snps are already in the hub
            if we get a snp that is not in the hub, we throw an error in debug mode
            """

            self.hub = {} # {snp: [sum(np.float32), cnt(np.uint32)], ...}
            self.sum_pos = 0 # position for summation variable in hub value list
            self.cnt_pos = 1 # position for count variable in hub value list
            self.bin_pos = 2 # position for bin number in hub value list

        # print: min, 25% quantile, avg, median, 75% quantile, max
        def print_stats(self) -> None:
            # collect & sort averages
            # TODO: make sure that scores have a count greater than 0
            avgs = np.sort(np.array([self.get_snp_avg(interaction) for interaction in self.hub.keys()], dtype=np.float32))
            print(f"min={avgs[0]} | 25%={avgs[int(len(avgs) * .25)]} | avg={np.mean(avgs):.2f} | med={np.median(avgs):.2f} | 75%={int(len(avgs) * .75)} | max={avgs[-1]:.2f}")
            return

        # print snp, avg, sum, cnt
        def print_hub(self) -> None:
            for k,v in self.hub.items():
                print(f"{k}: avg={self.get_snp_avg(k):.2f} sum={v[self.sum_pos]:.2f} cnt={v[self.cnt_pos]:.2f}")
            return

        # initialize hub with starting value
        def initialize_hub(self, bins, snps: npt.NDArray[np.str_], starting_value: np.float32 = np.float32(0.01)) -> None:
            # make sure bins is the correct type
            assert isinstance(bins, GenoHub.Bin)

            for snp in snps:
                # sum, cnt
                # TODO: set the positional value of the snp relative to the header of the csv file
                self.hub[snp] = [starting_value, np.uint32(0), bins.bin_binary_search_range(snp)]
            return

        # get snp sum
        def get_snp_sum(self, snp: np.str_) -> np.float32:
            # assert that snp is in hub
            assert snp in self.hub
            # return data
            return np.float32(self.hub[snp][self.sum_pos])

        # get snp count
        def get_snp_cnt(self, snp: np.str_) -> np.uint32:
            # assert that snp is in hub
            assert snp in self.hub
            # return data
            return np.uint32(self.hub[snp][self.cnt_pos])

        # get snp average. If count is zero, return sum
        def get_snp_avg(self, snp: np.str_)-> np.float32:
            # assert that snp is in hub
            assert snp in self.hub

            # make sure we are not dividing by zero
            if self.hub[snp][self.cnt_pos] == np.uint32(0):
                return self.get_snp_sum(snp)

            # return data
            return np.float32(self.hub[snp][self.sum_pos] / np.float32(self.hub[snp][self.cnt_pos]))

        # get snp bin number
        def get_snp_bin(self, snp: np.str_) -> hub_bin_t:
            # assert that snp is in hub
            assert snp in self.hub
            # return data
            return hub_bin_t(self.hub[snp][self.bin_pos])

        # update snp sum and count. If count is zero, set sum to value and increment count
        def update_hub(self, snp: np.str_, value: np.float32) -> None:
            # assert that snp is in hub
            assert snp in self.hub

            # check if we are updating a snp for the first time
            if self.hub[snp][self.mapping['cnt']] == np.uint32(0):
                # set the starting value if first time
                self.hub[snp][self.mapping['sum']] = value
                self.hub[snp][self.mapping['cnt']] += np.uint32(1)
                return
            else:
                # aggregate current values if not the first time
                self.hub[snp][self.mapping['sum']] += value
                self.hub[snp][self.mapping['cnt']] += np.uint32(1)
                return

        # get all averages for given snps
        def get_all_avgs(self, snps: np.str_) -> npt.NDArray[np.float32]:
            # return all averages for given snps
            return np.array([self.get_snp_avg(snp) for snp in snps], dtype=np.float32)

    class Bin:
        """
        Hub to all snps and their apporpriate bin.
        """
        def __init__(self, bin_size: np.uint16):
            self.bins = {} # {chrom: [np.array([pos1, pos2, ...]), np.array([pos1, pos2, ...]), ...]}
            self.bin_size = bin_size
            pass

        # create bins for snps
        # O(|snps| * log(|snps|)) time complexity
        def generate_bins(self, snps: npt.NDArray[np.str_]) -> None:
            # quick checks
            assert len(snps) > 0
            assert self.bin_size > 0

            # collect all chromosomes and sort snp postions
            sorted_snps = {}
            for snp in snps:
                # split header into chromosome and position
                chrom, pos = self.snp_chrm_pos(snp)

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
            assert len(snps) == self.count_bin_objs()

        # helper to generate chromosome number and snp position
        def snp_chrm_pos(self, snp_name: np.str_) -> Tuple[chrom_num_t, chrom_pos_t]:
            chrom, pos = snp_name.split('.')
            chrom, pos = chrom_num_t(chrom), chrom_pos_t(pos)
            return chrom, pos

        # count all objects in bins
        def count_bin_objs(self) -> np.uint16:
            sum = 0
            for _, bins in self.bins.items():
                for bin in bins:
                    sum += len(bin)

            return np.uint16(sum)

        # find bin number given list of bins
        # O(log(bin_count)) time complexity
        def bin_binary_search_range(self, snp: np.str_) -> Tuple[np.uint32, np.uint32]:
            """
            Will go through all bins and find the bin number for a given snp.
            The first part of the snp name is the chromosome number and the second part is the position.
            We can assume that each bin is sorted in ascending order.
            As such, we can just make sure that the snp is in between the bin's first and last element. (inclusive)

            Parameters:
            - snp: SNP name in the format 'chromosome.position'

            Returns:
            - chrom: Chromosome number for the given snp
            - bin: Bin number for the given snp
            """

            # get chromosome and position
            chrom, pos = self.snp_chrm_pos(snp)
            # debug check
            assert self.is_snp_in_a_bin(snp)

            # get the list of bins for the binary search
            left, right = 0, len(self.bins[chrom]) - 1

            while left <= right:
                # middle of the list of bins
                mid = (left + right) // 2
                # get the low and high values of the bin
                low, high = self.bins[chrom][mid][0], self.bins[chrom][mid][-1]

                if low <= pos <= high:
                    return np.uint32(chrom), np.uint32(mid)
                elif pos < low:
                    right = mid - 1
                else:
                    left = mid + 1

            exit(-1, "bin_binary_search_range Error: SNP not in bin")

        # check if snp is actually in a bin
        # O(log(bin_cnt) + log(bin_size)) time complexity
        def is_snp_in_a_bin(self, snp: np.str_) -> bool:
            """
            Will go through all bins and find the bin number for a given snp.
            The first part of the snp name is the chromosome number and the second part is the position.
            We can assume that each bin is sorted in ascending order.
            As such, we will do a binary search to find the actual snn position in the bin.

            Parameters:
            - snp: SNP name in the format 'chromosome.position'

            Returns:
            - chrom: Chromosome number for the given snp
            - bin: Bin number for the given snp
            """

            # get chromosome and position
            chrom, pos = self.snp_chrm_pos(snp)
            bin_id = None

            # binary search for the correct bin
            left, right = 0, len(self.bins[chrom]) - 1
            while left <= right:
                # middle of the list of bins
                mid = (left + right) // 2
                # get the low and high values of the bin
                low, high = self.bins[chrom][mid][0], self.bins[chrom][mid][-1]

                # check that position is in the bin range
                if low <= pos <= high:
                    bin_id = mid
                    break
                elif pos < low:
                    right = mid - 1
                else:
                    left = mid + 1

            # make sure we found a bin
            if bin_id is None:
                return False

            # binary search to find the snp position in the bin
            bin = self.bins[chrom][bin_id]
            low = 0
            high = len(bin) - 1

            while low <= high:
                mid = (low + high) // 2
                mid_value = bin[mid]

                if mid_value == pos:
                    return True  # Target found, return index
                elif mid_value < pos:
                    low = mid + 1  # Search the right half
                else:
                    high = mid - 1  # Search the left half
            return False  # Target not found

        # get all snps in a given bin with r2 > 0.0                       SNPS              weighted r2 scores > 0
        def get_snps_r2_in_bin(self, snp: np.str_, snp_hub) -> Tuple[npt.NDArray[np.str_], npt.NDArray[np.float32]]:
            # make sure that snp_hub is the correct type
            assert isinstance(snp_hub, GenoHub.SNP)

            # get chromosome and position from snp
            chrom, _ = self.snp_chrm_pos(snp)
            bin = snp_hub.get_snp_bin(snp)

            # go thorugh all snps in the bin and collect the ones with r2 > 0.0
            snps = []
            r2 = []
            bin_snps = np.array([f"{chrom}.{pos}" for pos in self.bins[chrom][bin]], dtype=np.str_)

            for snp in bin_snps:
                # make sure this snp is in the hub
                assert snp in snp_hub.hub

                # check if r2 is greater than 0.0
                if snp_hub.get_snp_avg(snp) > np.float32(0.0):
                    snps.append(snp)
                    r2.append(snp_hub.get_snp_avg(snp))

            # make sure snps and r2 are the same size
            assert len(snps) == len(r2)

            # get the snps in the bin
            return np.array(snps, dtype=np.str_), np.array(r2, dtype=np.float32) / np.sum(r2, dtype=np.float32)

        # return a random snp from the same chromosome and bin
        def get_ran_snp_in_bin(self, snp: np.str_, rng: np.random.Generator, snp_hub) -> np.str_:
            # make sure there is a '.' inside the snp string
            assert '.' in snp
            # make sure snp_hub is the correct type
            assert isinstance(snp_hub, GenoHub.SNP)

            # get chromosome and position from snp
            chrom, _ = self.snp_chrm_pos(snp)
            bin = snp_hub.get_snp_bin(snp)

            # get ran pos from bin
            pos = self.bins[chrom][bin][rng.integers(0, len(self.bins[chrom][bin]), dtype=np.uint16)]

            # return a random snp
            return np.str_(f"{chrom}.{pos}")

        # get all snps in the same chromosome but different bin
        def get_snps_r2_in_chrom(self, snp: np.str_, snp_hub) -> Tuple[npt.NDArray[np.str_], npt.NDArray[np.float32]]:
            # make sure there is a '.' inside the snp string
            assert '.' in snp
            # make sure snp_hub is the correct type
            assert isinstance(snp_hub, GenoHub.SNP)

            # get chromosome and position from snp
            chrom, _ = self.snp_chrm_pos(snp)
            bin = snp_hub.get_snp_bin(snp)

            # return all snps and r2 > 0.0 in the chromosome
            snps = []
            r2 = []

            # go through all bins in the chromosome
            for i, b in enumerate(self.bins[chrom]):
                if i == bin:
                    continue

                # go through each chromose position in the bin
                for pos in b:
                    s = np.str_(f"{chrom}.{pos}")

                    # make sure this snp is in the hub
                    if snp_hub.get_snp_avg(s) > np.float32(0.0):
                        snps.append(s)
                        r2.append(snp_hub.get_snp_avg(s))

            # make sure snps and r2 are the same size
            assert len(snps) == len(r2)

            # return all snps in the chromosome
            return np.array(snps, dtype=np.str_) , np.array(r2, dtype=np.float32)/ np.sum(r2, dtype=np.float32)

        # get random snp from the same chromosome but different bin
        def get_ran_snp_in_chrom(self, snp: np.str_, rng: np.random.Generator, snp_hub) -> np.str_:
            # make sure there is a '.' inside the snp string
            assert '.' in snp
            # make sure snp_hub is the correct type
            assert isinstance(snp_hub, GenoHub.SNP)

            # get chromosome and position from snp
            chrom, _ = self.snp_chrm_pos(snp)
            bin = snp_hub.get_snp_bin(snp)

            # get random bin index from the chromosome
            i = bin
            while i == bin:
                i = rng.integers(0, len(self.bins[chrom]), dtype=np.uint16)

            # get random snp from the bin
            pos = self.bins[chrom][i][rng.integers(0, len(self.bins[chrom][i]), dtype=np.uint16)]

            # return a random snp
            return np.str_(f"{chrom}.{pos}")

        # get all snps outside the chromosome with r2 > 0.0
        def get_snps_r2_out_chrom(self, snp: np.str_, snp_hub) -> Tuple[npt.NDArray[np.str_], npt.NDArray[np.float32]]:
            # make sure there is a '.' inside the snp string
            assert '.' in snp
            # make sure snp_hub is the correct type
            assert isinstance(snp_hub, GenoHub.SNP)

            # get chromosome and position from snp
            chrom, _ = self.snp_chrm_pos(snp)

            # get all keys in the hub
            chrom_keys = list(self.hub.keys())

            # remove chrom from chorom_keys
            chrom_keys.remove(chrom)

            # go through all remaining keys and get all snps with r2 > 0.0
            snps = []
            r2 = []
            for c in chrom_keys:
                # go through all bins in the chromosome
                for bin in self.bins[c]:
                    # go through each chromosome position in the bin
                    for pos in bin:
                        s = np.str_(f"{c}.{pos}")

                        # make sure this snp is in the hub
                        if snp_hub.get_snp_avg(s) > np.float32(0.0):
                            snps.append(s)
                            r2.append(snp_hub.get_snp_avg(s))

            # make sure snps and r2 are the same size
            assert len(snps) == len(r2)

            # return all snps outside the chromosome
            return np.array(snps, dtype=np.str_), np.array(r2, dtype=np.float32) / np.sum(r2, dtype=np.float32)

        # get random snp outside the chromosome
        def get_ran_snp_out_chrom(self, snp: np.str_, rng: np.random.Generator, snp_hub) -> np.str_:
            # make sure there is a '.' inside the snp string
            assert '.' in snp
            # make sure snp_hub is the correct type
            assert isinstance(snp_hub, GenoHub.SNP)

            # get chromosome and position from snp
            chrom, _ = self.snp_chrm_pos(snp)

            # get all keys in the hub
            chrom_keys = list(self.hub.keys())

            # remove chrom from chorom_keys
            chrom_keys.remove(chrom)

            # get random chromosome
            c = chrom_keys[rng.integers(0, len(chrom_keys), dtype=np.uint16)]

            # get random bin index from the chromosome
            i = rng.integers(0, len(self.bins[c]), dtype=np.uint16)

            # get random snp from the bin
            pos = self.bins[c][i][rng.integers(0, len(self.bins[c][i]), dtype=np.uint16)]

            # return a random snp
            return np.str_(f"{c}.{pos}")

        # get a random snp from the hub
        def get_ran_snp(self, rng: np.random.Generator) -> np.str_:
            # get all chromosome keys in the hub
            chrom_keys = list(self.bins.keys())

            # get a random chromosome key
            c = chrom_keys[rng.integers(0, len(chrom_keys), dtype=np.uint16)]

            # get a random bin index from the chromosome
            i = rng.integers(0, len(self.bins[c]), dtype=np.uint16)

            # get a random snp from the bin
            pos = self.bins[c][i][rng.integers(0, len(self.bins[c][i]), dtype=np.uint16)]

            # return a random snp
            return np.str_(f"{c}.{pos}")


    class EPI:
        """
        Hub to keep track of epistatic performances encountered.
        We assume data in the hub is for the best performing configuration of the interaction and the logical operator.
        """

        def __init__(self) -> None:
            """
            self.hub: dictionary to hold all interactions (snp1,snp2) and values [result(np.float32), LO(str)]
            assuming that hub will grow from an empty start
            we ALWAYS sort keys such that snp1 < snp2
            if we try to update an existing interaction, we throw an error in debug mode
            """

            self.hub = {} # {('snp1', 'snp2'): [result(np.float32), LO(str)], ...}
            self.res_pos = 0 # position for result variable in hub value list
            self.lo_pos = 1 # position for logical operator variable in hub value list
            return

        def print_stats(self) -> None:
            print(f"EPI hub: {self.hub}")
            return

        def is_interaction_in_hub(self, snp1: np.str_, snp2: np.str_) -> bool:
            # make sure that snp1 < snp2
            assert snp1 < snp2

            # check if interaction is in hub
            return (snp1, snp2) in self.hub

        # get interaction result r^2
        def get_interaction_res(self, snp1: np.str_, snp2: np.str_) -> np.float32:
            # assert that snp is in hub
            assert (snp1, snp2) in self.hub or (snp2, snp1) in self.hub

            # create snp key
            snp_key = (snp1, snp2) if snp1 < snp2 else (snp2, snp1)

            # return data
            return self.hub[snp_key][self.res_pos]

        # get interaction logical operator
        def get_interaction_lo(self, snp1: np.str_, snp2: np.str_) -> np.str_:
            # assert that snp is in hub
            assert (snp1, snp2) in self.hub or (snp2, snp1) in self.hub

            # create snp key
            snp_key = (snp1, snp2) if snp1 < snp2 else (snp2, snp1)

            # return data
            return self.hub[snp_key][self.lo_pos]

        # return unseen interactions in the hub from a given list of interactions
        def unseen_interactions(self, snps: npt.NDArray[np.str_]) -> npt.NDArray[np.str_]:
            """
            Receives a list of interactions (snp1, snp2) and returns interactions not in the hub

            Parameters:
            - snps: List of snp interactions (snp1, snp2) to check for in the hub

            Returns:
            - unseen: List of snp interactions (snp1, snp2) not in the hub
            """

            # collect all unseen interactions
            unseen = []
            # iterate through all snps
            for snp in snps:
                # make sure they are the right size & type
                assert len(snp) == 2
                assert isinstance(snp[0], np.str_) and isinstance(snp[1], np.str_)
                assert '.' in snp

                # form a lookup key snp1 and snp2
                snp1 = snp[0]
                snp2 = snp[1]
                snp_key = (snp1, snp2) if snp1 < snp2 else (snp2, snp1)

                # check if interaction is in hub
                if snp_key not in self.hub:
                    unseen.append(snp_key)

            return np.array(unseen, dtype=np.str_)

        # update hub with new interaction and results (r^2, lo)
        def update_hub(self, snp1: np.str_, snp2: np.str_, result: np.float32, lo: np.str_) -> None:
            # assert that interaction is NOT in hub
            assert (snp1, snp2) not in self.hub and (snp2, snp1) not in self.hub

            # create snp key
            snp_key = (snp1, snp2) if snp1 < snp2 else (snp2, snp1)

            # update hub with new interaction
            self.hub[snp_key] = [result, lo]

            return

    # initialize all hubs
    def __init__(self, snps: npt.NDArray[np.str_], bin_size: np.uint16) -> None:
        # bin hub stuff
        print('Initializing GenoHub')
        self.bin = self.Bin(bin_size)
        self.bin.generate_bins(snps)
        print('Bin Hub Initialized')
        # snp hub stuff
        self.snp = self.SNP()
        self.snp.initialize_hub(bins=self.bin, snps=snps)
        print('SNP Hub Initialized')
        # epi hub stuff
        self.epi = self.EPI()
        print('EPI Hub Initialized')

        return

    # print epi and snp hub stats
    def hub_stats(self) -> None:
        print("SNP Hub Summary Stats:")
        self.snp.print_stats()
        print("EPI Hub Summary Stats:")
        self.epi.print_stats()
        return

    def update_hubs(self, interactions) -> None:
        """
        Takes in a list of interactions and results to update the EPI and SNP hubs.
        Calls each hub to update itself appropriately.

        Parameters:
        interactions ([(snp1,snp2,r^2,lo), ...]): List of interactions and results.
        snp1: str
        snp2: str
        r^2: np.float32
        lo: str
        """
        # do nothing if interactions is empty
        if len(interactions) == 0:
            return None

        # make sure each entry is the right type
        assert len(interactions[0]) == 4
        assert all(isinstance(x[0], str) for x in interactions) # snp1
        assert all(isinstance(x[1], str) for x in interactions) # snp2
        assert all(isinstance(x[2], np.float32) for x in interactions) # r^2
        assert all(isinstance(x[3], str) for x in interactions) # lo

        # update epi hub
        for interaction in interactions:
            self.epi.update_hub(interaction[0], interaction[1], interaction[2], interaction[3])
        # update snp hub
        for interaction in interactions:
            self.snp.update_hub(interaction[0], interaction[2])
            self.snp.update_hub(interaction[1], interaction[2])

        return

    # call the epi class to unseen interactions
    def unseen_interactions(self, snps) -> npt.NDArray[np.str_]:
        return self.epi.unseen_interactions(snps)

    # get a snp from the same chromosome and bin with r2 > 0.0 based on r2 weight
    # TODO: make sure that the snp is not the same as the input snp
    def get_smt_snp_in_bin(self, snp: np.str_, rng: np.random.Generator) -> np.str_:
        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # get all snps and r2 scores for a given snp within the same chorosome and bin
        snps, r2 = self.bin.get_snps_r2_in_bin(snp, self.snp)
        assert(len(snps) == len(r2))

        # if no snps were returned, return a random one
        if len(snps) == 0:
            return self.bin.get_ran_snp_in_bin(snp)

        # get a random snp based on r2 scores as weights
        return rng.choice(snps, p=r2)

    # get a random snp from the same chromosome and bin
    # TODO: make sure that the snp is not the same as the input snp
    def get_ran_snp_in_bin(self, snp: np.str_, rng: np.random.Generator) -> List[np.str_]:
        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # call snp hub for a random snp in-in
        return self.bin.get_ran_snp_in_bin(snp, rng, self.snp)

    # geta snp from the same chromosome but different bin
    def get_smt_snp_in_chrm(self, snp: np.str_, rng: np.random.Generator) -> np.str_:
        # get all snps and r2 scores for a given snp within the same chorosome and bin
        snps, r2 = self.bin.get_snps_r2_in_chrom(snp, self.snp)
        assert(len(snps) == len(r2))

        # if no snps were returned, return a random one
        if len(snps) == 0:
            return self.bin.get_ran_snp_in_bin(snp, rng, self.snp)

        # get a random snp based on r2 scores as weights
        return rng.choice(snps, p=r2)

    # get a random snp from the same chromosome but different bin
    def get_ran_snp_in_chrm(self, snp: np.str_, rng: np.random.Generator) -> np.str_:
        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # call snp hub for a random snp in-in
        return self.bin.get_ran_snp_in_chrom(snp, rng, self.snp)

    # get a snp from outside the chromosome with r2 > 0.0 based on r2 weight
    def get_smt_snp_out_chrm(self, snp: np.str_, rng: np.random.Generator) -> np.str_:
        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # get all snps and r2 scores for a given snp within the same chorosome and bin
        snps, r2 = self.bin.get_snps_r2_out_chrom(snp, self.snp)
        assert(len(snps) == len(r2))

        # if no snps were returned, return a random one
        if len(snps) == 0:
            return self.bin.get_ran_snp_out_chrom(snp, rng, self.snp)

        # get a random snp based on r2 scores as weights
        return rng.choice(snps, p=r2)

    # get random snp from outside the chromosome
    def get_ran_snp_out_chrm(self, snp: np.str_, rng: np.random.Generator) -> np.str_:
        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # call snp hub for a random snp in-in
        return self.bin.get_ran_snp_out_chrom(snp, rng, self.snp)

    # get a random snp from the same chromosome and bin
    def get_ran_snp(self, rng: np.random.Generator) -> np.str_:
        # get random snp
        return self.bin.get_ran_snp(rng)