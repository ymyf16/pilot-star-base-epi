# Will contain the definition of Pipeline class which will be used by both the Epistatic branch and the univariate branch.

from sklearn.base import BaseEstimator
from typing import List, Dict, Tuple, Set
from .scikit_node import ScikitNode
import numpy as np
from typeguard import typechecked
import numpy.typing as npt
from typing_extensions import Self
import copy as cp
from .geno_hub import GenoHub #YF for diversity calc

# numpy random number generator type
gen_rng_t = np.random.Generator
gen_snp_arr_t = npt.NDArray[np.str_]
epi_pairs_t = Set[Tuple]
uni_snps_t = Set #YF
uniq_chrombin_t = Set
traits_t = List
r2_t = np.float32
feature_cnt_t = np.int16
div_score_t = np.float32

@typechecked # for debugging purposes
class Pipeline:
    def __init__(self,
                 epi_pairs: epi_pairs_t,
                 uni_snps: uni_snps_t, #YF
                 selector_node: ScikitNode,
                 root_node: ScikitNode,
                 traits: traits_t,
                 hub: GenoHub) -> None:

        # need to make a deep copy to keep independent
        self.epi_pairs = cp.deepcopy(epi_pairs) # will contain the interacting_features
        self.uni_snps = cp.deepcopy(uni_snps) # will contain individual snps
        self.traits = cp.deepcopy(traits) # will contain the r2 (traits[0]) and feature_cnt (traits[1])
        self.selector_node = cp.deepcopy(selector_node) # will contain the selector node
        self.root_node = cp.deepcopy(root_node) # will contain the regressor node

        #YF trying to implement the diversity stat
        self.uniq_chrombins = set()
        self.hub = hub


    # method to set the epi_pairs
    def set_epi_pairs(self, epi_pairs: epi_pairs_t) -> None:
        # check that internal epi_pairs is empty
        assert len(self.epi_pairs) == 0
        # make sure that the epi_pairs are not empty
        assert len(epi_pairs) > 0

        # update the epi_pairs
        self.epi_pairs = cp.deepcopy(epi_pairs)
        return
    
    #YF
    def set_uni_snps(self, uni_snps: uni_snps_t) -> None:
        # check that internal uni_snps is empty
        assert len(self.uni_snps) == 0
        # make sure that the uni_snps are not empty
        assert len(uni_snps) > 0
        # update 
        self.uni_snps = cp.deepcopy(uni_snps)
        return

    # set traits
    def set_traits(self, traits: traits_t) -> None:
        # check that internal traits is empty
        assert len(self.traits) == 0
        # make sure that the traits are not empty
        assert len(traits) == 2
        # make sure correct types
        assert isinstance(traits[0], r2_t)
        assert isinstance(traits[1], feature_cnt_t)
        # make sure we have a non-negative number of features
        assert traits[1] >= 0

        # update the traits
        self.traits = cp.deepcopy(traits)
        return

    def get_trait_r2(self) -> r2_t:
        assert len(self.traits) == 2
        return self.traits[0]

    def get_trait_feature_cnt(self) -> feature_cnt_t:
        assert len(self.traits) == 2
        assert self.traits[1] >= 0 # make sure we have a non-negative number of features
        return self.traits[1]

    def get_traits(self) -> traits_t:
        return self.traits

    def get_epi_pairs(self) -> epi_pairs_t:
        return self.epi_pairs
    #YF
    def get_uni_snps(self) -> uni_snps_t:
        return self.uni_snps

    # method to get the number of nodes in the pipeline
    def get_branch_count(self):
        return len(self.epi_pairs)

    # method to get the number of features/SNPs in the pipeline
    def get_feature_count(self):
        return self.selector_node.get_feature_count()

    # method to get the selector node
    def get_selector_node(self):
        return self.selector_node

    # method to get the root node
    def get_root_node(self):
        return self.root_node
    
    #YF
    # method to get the number of unique bins in the current pipeline
    def get_uniq_chrom_bins(self) -> uniq_chrombin_t:
        if len(self.uniq_chrombins) == 0: #YF? not sure if this is the most efficient way to go
            self.calc_uniq_chrom_bins(self.hub)
            return uniq_chrombin_t(self.uniq_chrombins)
        else:
            return uniq_chrombin_t(self.uniq_chrombins)
        
    def calc_uniq_chrom_bins(self) -> None:
        # collect unique SNPs from uni_snps and epi_pairs
        uniq_snps = set()
        uniq_snps.update(self.uni_snps)
        for snp_tuple in self.epi_pairs:
            uniq_snps.add(snp_tuple[0])
            uniq_snps.add(snp_tuple[1])
        # get chrom.bin info from SNPs
        assert len(uniq_snps) > 0
        for snp in uniq_snps:
            assert snp in self.hub
            # get chrom and bin number
            chrom, _ = self.hub.bin_hub.snp_chrm_pos(snp)
            bin = self.hub.snp_hub.get_snp_bin(snp)
            # create tuple to store
            self.uniq_chrombins.add((chrom,bin))
        assert len(self.uniq_chrombins) > 0
        print("unique bins of current pipeline calculated")
        return

    ##YFupdate
    def get_diversity(self) -> div_score_t:
        total_combo = self.hub.total_combo
        pipeline_combo = len(self.uniq_chrombins)
        return div_score_t(pipeline_combo/total_combo)

    # print the pipeline
    def print_pipeline(self) -> None:
        print("Pipeline:")
        print("UniNodes:")
        print('len(self.uni_snps):', len(self.uni_snps))
        for uni_node in self.uni_snps:
            print(uni_node)
        print("EpiNodes:")
        print('len(self.epi_pairs):', len(self.epi_pairs))
        for epi_node in self.epi_pairs:
            print(epi_node)
        print("Selector Node:")
        print(self.selector_node.selector)
        print("Root Node:")
        print(self.root_node.regressor)
        print("Unique chrom,bin in pipeline:")
        print(len(self.uniq_chrombins))
        return

    def mutate_selector_node(self, rng: gen_rng_t) -> None:
        self.selector_node.mutate(rng)

    def mutate_root_node(self, rng: gen_rng_t) -> None:
        self.root_node.mutate(rng)