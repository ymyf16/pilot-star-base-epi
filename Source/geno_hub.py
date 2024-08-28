#####################################################################################################
#
# Interface for communicating with both EPI and SNP hubs.
#
#####################################################################################################

import numpy as np
from typing import List, Tuple

from typeguard import typechecked
import numpy.typing as npt


### General Types

# chromosome number
gen_chrom_num_t = np.uint8
# chromosome snp position
gen_chrom_pos_t = np.uint32
# header snps type
gen_header_snps_t = npt.NDArray[np.str_]
# individual snp type
gen_snp_t = np.str_
# numpy random number generator type
gen_rng_t = np.random.Generator

### SNP Hub Types

# sum type
snp_hub_sum_t = np.float32
# bin id position type
snp_hub_bin_t = np.uint16
# count type
snp_hub_cnt_t = np.uint32
# header position
snp_hub_pos_t = np.uint32

### Bin Hub Types

# type of object inside bins
bin_hub_arr_t = np.uint32
# type for bin size
bin_hub_size_t = np.uint16

### EPI Hub Types

# type of results being stored
epi_hub_res_t = np.float32
# type of logical operator being stored
epi_hub_lo_t = np.str_


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

            sum_pos = 0 # position for summation variable in hub value list
            cnt_pos = 1 # position for count variable in hub value list
            bin_pos = 2 # position for bin number in hub value list
            pos_pos = 3 # position for header position in hub value list
            """

            self.hub = {} # {snp: [sum(np.float32), cnt(np.uint32), bin_pos()], ...}

        # return two random snps from the hub
        def get_random_interaction(self, rng: gen_rng_t) -> Tuple[gen_snp_t, gen_snp_t]:
            # get keys from the hub
            snp_keys = list(self.hub.keys())

            # two random snps without replacement
            snp1, snp2 = np.array(rng.choice(snp_keys, 2, replace=False), dtype=gen_snp_t)

            # return the correct ordering of snps
            if snp1 > snp2:
                return snp2, snp1

            return snp1, snp2

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
                print(f"{k}: avg={self.get_snp_avg(k):.2f} sum={v[0]:.2f} cnt={v[1]:.2f} bin={v[2]} pos={v[3]}")
            return

        # will add snp, sum, cnt, bin, and pos to the hub
        def add_to_hub(self,
                       snp: gen_snp_t,
                       sum: snp_hub_sum_t,
                       cnt: snp_hub_cnt_t,
                       bin: snp_hub_bin_t,
                       pos: snp_hub_pos_t) -> None:
            """
            will take in a snp, sum, cnt, bin, and pos and add it to the hub

            Args:
                snp (gen_snp_t): chrm.pos string
                sum (snp_hub_sum_t): starting sum value
                cnt (snp_hub_cnt_t): count of how many times the snp has been updated
                bin (snp_hub_bin_t): bin number of the snp relative to the bin hub
                pos (snp_hub_pos_t): position of the snp in the csv header
            """


            # add to hub
            self.hub[snp] = [sum, cnt, bin, pos]
            return

        # get snp sum
        def get_snp_sum(self, snp: gen_snp_t) -> snp_hub_sum_t:
            # assert that snp is in hub
            assert snp in self.hub
            # return data
            return np.float32(self.hub[snp][0])

        # get snp count
        def get_snp_cnt(self, snp: gen_snp_t) -> snp_hub_pos_t:
            # assert that snp is in hub
            assert snp in self.hub
            # return data
            return np.uint32(self.hub[snp][1])

        # get snp average. If count is zero, return sum
        def get_snp_avg(self, snp: gen_snp_t)-> snp_hub_sum_t:
            # assert that snp is in hub
            assert snp in self.hub

            # make sure we are not dividing by zero
            if self.hub[snp][1] ==snp_hub_cnt_t(0):
                return self.get_snp_sum(snp)

            # return data
            return np.float32(self.hub[snp][0] / np.float32(self.hub[snp][1]))

        # get snp bin number
        def get_snp_bin(self, snp: gen_snp_t) -> bin_hub_size_t:
            # assert that snp is in hub
            assert snp in self.hub
            # return data
            return bin_hub_size_t(self.hub[snp][2])

        # get snp header position
        def get_snp_pos(self, snp: gen_snp_t) -> snp_hub_pos_t:
            # assert that snp is in hub
            assert snp in self.hub
            # return data
            return snp_hub_pos_t(self.hub[snp][3])

        # update snp sum and count. If count is zero, set sum to value and increment count
        def update_hub(self, snp: gen_snp_t, value: np.float32) -> None:
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
        def get_all_avgs(self, snps: List[gen_snp_t]) -> npt.NDArray[snp_hub_sum_t]:
            # return all averages for given snps
            return np.array([self.get_snp_avg(snp) for snp in snps], dtype=np.float32)

    class Bin:
        """
        Hub to all snps and their apporpriate bin.
        """
        def __init__(self) -> None:
            self.bins = {} # {chrom: [np.array([pos1, pos2, ...], dtype=bin_hub_arr_t), ...]}
            return

        # create bins for snps
        # O(|snps| * log(|snps|)) time complexity
        def generate_bins(self, snps: gen_header_snps_t, bin_size: bin_hub_size_t) -> List:
            # quick checks
            assert len(snps) > 0
            assert bin_size > 0

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
                sorted_pos = np.sort(np.array(pos, dtype=gen_chrom_pos_t), kind='mergesort')

                for p in sorted_pos:
                    # if we get a new chromosome, intialize a new bin for it
                    if chrom not in self.bins:
                        self.bins[chrom] = [[p]]
                    else:
                        # check if the current bin is full
                        if len(self.bins[chrom][-1]) == bin_size:
                            self.bins[chrom].append([p])
                        else:
                            self.bins[chrom][-1].append(p)

            # collect each snps bin number
            snp_bins = []
            # go through self.bins and collect all snps, bin_num
            for chrom, bins in self.bins.items():
                for i in range(len(bins)):
                    for pos in bins[i]:
                        snp = np.str_(f"{chrom}.{pos}")
                        snp_bins.append((snp, i))


            # cast all bins to numpy arrays for efficiency
            for chrom, bins in self.bins.items():
                self.bins[chrom] = [np.array(b, dtype=gen_chrom_pos_t) for b in bins]

            # make sure all SNPs are accounted for
            assert len(snps) == self.count_bin_objs()
            # make sure snp_bins is the correct size
            assert len(snp_bins) == len(snps)

            return snp_bins

        # helper to generate chromosome number and snp position
        def snp_chrm_pos(self, snp: np.str_) -> Tuple[gen_chrom_num_t, gen_chrom_pos_t]:
            chrom, pos = snp.split('.')
            chrom, pos = gen_chrom_num_t(chrom), gen_chrom_pos_t(pos)
            return chrom, pos

        # count all objects in bins
        def count_bin_objs(self) -> np.uint16:
            sum = 0
            for _, bins in self.bins.items():
                for bin in bins:
                    sum += len(bin)

            return np.uint16(sum)

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

            res_pos = 0 # position for result variable in hub value list
            lo_pos = 1 # position for logical operator variable in hub value list
            """

            self.hub = {} # {('snp1', 'snp2'): [result(np.float32), LO(str)], ...}

            return

        def print_stats(self) -> None:
            print(f"EPI hub:")
            for k,v in self.hub.items():
                print(f"{k}: res={v[0]:.2f} lo={v[1]}")
            return

        def is_interaction_in_hub(self, snp1: gen_snp_t, snp2: gen_snp_t) -> bool:
            # make sure that snp1 < snp2
            assert snp1 < snp2

            # check if interaction is in hub
            return (snp1, snp2) in self.hub

        # get interaction result r^2
        def get_interaction_res(self, snp1: gen_snp_t, snp2: gen_snp_t) -> epi_hub_res_t:
            # check for snp < snp2
            assert snp1 < snp2
            # assert that interaction is NOT in hub
            assert (snp1, snp2) in self.hub

            # return data
            return self.hub[(snp1, snp2)][0]

        # get interaction logical operator
        def get_interaction_lo(self, snp1: gen_snp_t, snp2: gen_snp_t) -> epi_hub_lo_t:
            # check for snp < snp2
            assert snp1 < snp2
            # assert that interaction is NOT in hub
            assert (snp1, snp2) in self.hub

            # return data
            return self.hub[(snp1, snp2)][1]

        # return unseen interactions in the hub from a given list of interactions
        def unseen_interactions(self, snps: npt.NDArray[gen_snp_t]) -> npt.NDArray[gen_snp_t]:
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
                assert isinstance(snp[0], gen_snp_t) and isinstance(snp[1], gen_snp_t)
                assert '.' in snp

                # form a lookup key snp1 and snp2
                snp1 = snp[0]
                snp2 = snp[1]
                snp_key = (snp1, snp2) if snp1 < snp2 else (snp2, snp1)

                # check if interaction is in hub
                if snp_key not in self.hub:
                    unseen.append(snp_key)

            return np.array(unseen, dtype=gen_snp_t)

        # update hub with new interaction and results (r^2, lo)
        def update_hub(self, snp1: gen_snp_t, snp2: gen_snp_t, result: epi_hub_res_t, lo: epi_hub_lo_t) -> None:
            # check for snp < snp2
            assert snp1 < snp2
            # assert that interaction is NOT in hub
            assert (snp1, snp2) not in self.hub

            # update hub with new interaction
            self.hub[(snp1, snp2)] = [result, lo]
            return

    # initialize all hubs
    def __init__(self, snps: gen_header_snps_t, bin_size: bin_hub_size_t) -> None:
        # bin hub stuff
        print('Initializing GenoHub')
        self.bin_hub = self.Bin()
        # get snps and their bin id
        snp_bin = self.bin_hub.generate_bins(snps, bin_size)
        print('Bin Hub Initialized')

        # snp hub stuff
        self.snp_hub = self.SNP()
        # update snp_hub with snp_bin and snp header positions
        for s in snp_bin:
            # find where snp is located in csv header (snps)
            h_pos = snp_hub_pos_t(np.where(snps == s[0])[0][0])
            assert s[0] == snps[h_pos]
            # add snp to hub with all its data
            self.snp_hub.add_to_hub(s[0], snp_hub_sum_t(0.0001), snp_hub_cnt_t(0), snp_hub_bin_t(s[1]), h_pos)
        print('SNP Hub Initialized')

        # epi hub stuff
        self.epi = self.EPI()
        print('EPI Hub Initialized')
        print()
        return

    # get best lo for a given interaction
    def get_interaction_lo(self, snp1: gen_snp_t, snp2: gen_snp_t) -> epi_hub_lo_t:
        return self.epi.get_interaction_lo(snp1, snp2)

    # get r2 for a given interaction from the epi hub
    def get_interaction_res(self, snp1: gen_snp_t, snp2: gen_snp_t) -> epi_hub_res_t:
        return self.epi.get_interaction_res(snp1, snp2)

    # update epistatic hub and snp hub with new interaction and results
    def update_epi_n_snp_hub(self, snp1: gen_snp_t, snp2: gen_snp_t, result: epi_hub_res_t, lo: epi_hub_lo_t) -> None:
        self.epi.update_hub(snp1, snp2, result, lo)
        return

    # check if interaction is in the epi hub
    def is_interaction_in_hub(self, snp1: gen_snp_t, snp2: gen_snp_t) -> bool:
        return self.epi.is_interaction_in_hub(snp1, snp2)

    # get snp position from the snp hub
    def get_snp_pos(self, snp: gen_snp_t) -> snp_hub_pos_t:
        return self.snp_hub.get_snp_pos(snp)

    # get random interaction from snp_hub
    def get_ran_interaction(self, rng: gen_rng_t) -> Tuple[gen_snp_t, gen_snp_t]:
        return self.snp_hub.get_random_interaction(rng)