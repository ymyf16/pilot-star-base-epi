#####################################################################################################
#
# NSGA-II tool box for the selection and evolutionary process.
#
#####################################################################################################

import numpy as np
from typeguard import typechecked
from typing import List, Tuple, Set
import numpy.typing as npt
from pipeline import Pipeline # our custom pipeline
from geno_hub import GenoHub
from epi_nodes import EpiNode

@typechecked
def variation_order(rng: np.random.Generator,
                    offpring_cnt: np.uint16,
                    mut_prob: np.float32 = np.float32(.5),
                    cross_prob: np.float32 = np.float32(.5)) -> npt.NDArray[np.str_]:
    """
    Function to generate the order of variation operators to be applied to generate offspring.
    The order is determined by the probabilities of mutation and crossover.
    We return a list with the names of the operators in the order they should be applied.
    E.g.: ['m', 'c', 'm', 'c', ...]

    Crossover means two parents are required
    Mutation means one parent is required

    Parameters:
    mut_prob (np.float32): Probability of mutation occuring
    cross_prob (np.float32): Probability of crossover occuring

    Returns:
    npt.NDArray[np.str_]: A list of strings representing the order of variation operators to be applied
    """

    # keep count of how many offspring each operator generates
    parent_count = {'m': 1, 'c': 2}

    # generate half of the operators
    # we do this so that in the worse case (all crossover) we don't need to generate more than needed
    order = rng.choice(['m', 'c'], offpring_cnt // 2, p=[mut_prob, cross_prob]).tolist()

    # how many more offspring do we need
    left = offpring_cnt -  np.uint16(sum(parent_count[op] for op in order))

    # get that many more operators
    while left > 0:
        op = rng.choice(['m', 'c'], 1, p=[mut_prob, cross_prob])[0]
        if parent_count[op] <= left:
            order.append(op)
            left -= parent_count[op]

    # make sure we have the correct number of offspring
    assert sum(parent_count[op] for op in order) == offpring_cnt
    return np.array(order, dtype='<U1')

@typechecked
def produce_offspring(parents: List[Pipeline],
                      order: npt.NDArray[np.str_],
                      offspring_cnt: np.uint16,
                      rng: np.random.Generator,
                      mut_selector_p: np.float32 = np.float32(.5),
                      mut_regressor_p: np.float32 = np.float32(.5),
                      mut_ran_p: np.float32 = np.float32(.45),
                      mut_non_p: np.float32 = np.float32(.1),
                      mut_smt_p: np.float32 = np.float32(.45),
                      smt_in_in_p: np.float32 = np.float32(.1),
                      smt_in_out_p: np.float32 = np.float32(.45),
                      smt_out_out_p: np.float32 = np.float32(.45)) -> List[Pipeline]:
    # list to hold the offspring
    offspring = []

    # iterate over the order of operators
    i = 0
    for op in order:
        # mutation
        if op == 'm':
            # get the parent
            new_off = mutate(rng,parents[i])
            # clone the parent
            offspring.append(new_off)
            i += 1
        # crossover
        elif op == 'c':
            # get the parents
            parent1, parent2 = parents[i], parents[i + 1]
            # clone the parents
            offspring.append(parent1)
            offspring.append(parent2)
            i += 2
        else:
            raise ValueError(f"Unknown operator {op}")

    return offspring

def mutate(rng: np.random.Generator,
           parent: Pipeline,
           hub: GenoHub,
           mut_selector_p: np.float32 = np.float32(.5),
           mut_regressor_p: np.float32 = np.float32(.5),
           mut_ran_p: np.float32 = np.float32(.45),
           mut_non_p: np.float32 = np.float32(.1),
           mut_smt_p: np.float32 = np.float32(.45),
           smt_in_in_p: np.float32 = np.float32(.1),
           smt_in_out_p: np.float32 = np.float32(.45),
           smt_out_out_p: np.float32 = np.float32(.45)) -> Pipeline:

    # clone the pipeline
    offspring = Pipeline(epi_branches=parent.get_epi_branches(),
                         epi_pairs=parent.get_epi_pairs(),
                         selector_node=parent.get_selector_node(),
                         root_node=parent.get_root_node(),
                         traits=parent.get_traits())

    # print offspring pipeline
    print('*'*100)
    print("Offspring pipeline before:")
    offspring.print_pipeline()
    print('*'*100)

    # mutate the offspring

    # go through the epi branches and mutate if needed
    for node in offspring.get_epi_branches():
        # coin flip to determine if we mutate
        if rng.choice([True, False], p=[1.0-mut_non_p, mut_non_p]):
            # coin flip to determine the type of mutation
            if rng.choice([True, False], p=[mut_smt_p / (mut_smt_p + mut_ran_p), mut_ran_p / (mut_smt_p + mut_ran_p)]):
                # random mutation
                print('repo::mutate_epi_node_smrt')
                mutate_epi_node_smrt(rng, hub, node, smt_in_in_p, smt_in_out_p, smt_out_out_p)
            else:
                # smt mutation
                print('repo::mutate_epi_node_smrt')
                mutate_epi_node_smrt(rng, hub, smt_in_in_p, smt_in_out_p, smt_out_out_p)

    exit()

    # mutate the selector node
    if rng.choice([True, False], p=[mut_selector_p, 1.0-mut_selector_p]):
        print('repo::mutate_selector_node')
        offspring.mutate_selector_node(rng)

    # mutate the regressor node
    if rng.choice([True, False], p=[mut_regressor_p, 1.0-mut_regressor_p]):
        offspring.mutate_root_node(rng)

    # print offspring pipeline
    print('*'*100)
    print("Offspring pipeline after:")
    offspring.print_pipeline()
    print('*'*100)

    print('*'*100)
    print("Parent pipeline after:")
    parent.print_pipeline()
    print('*'*100)

    return offspring

# execute a smart mutation on the epi_node
def mutate_epi_node_smrt(rng: np.random.Generator,
                         hub: GenoHub,
                         node: EpiNode,
                         smt_in_in_p: np.float32 = np.float32(.1),
                         smt_in_out_p: np.float32 = np.float32(.45),
                         smt_out_out_p: np.float32 = np.float32(.45)) -> None:

    # pull snp1 and snp2 from node
    snp1_name, snp2_name = node.get_snp1_name(), node.get_snp2_name()

    # will hold new interactions
    new_snp1_name, new_snp2_name = None, None

    # randomly select one of the SNPs
    new_snp1_name = rng.choice([snp1_name, snp2_name])

    # randomly select one of the mutations to perform
    mut_fun = rng.choice([0,1,2], p=[smt_in_in_p, smt_in_out_p, smt_out_out_p])

    if mut_fun == 0:
        # in chromosome and in bin
        new_snp2_name = hub.get_smt_snp_in_bin(snp=new_snp1_name, rng=rng)
    elif mut_fun == 1:
        # in chromosome and out of bin
        new_snp2_name = hub.get_smt_snp_in_chrm(snp=new_snp1_name, rng=rng)
    elif mut_fun == 2:
        # out of chromosome
        new_snp2_name = hub.get_smt_snp_out_chrm(snp=new_snp1_name, rng=rng)
    else:
        exit("Unknown mutation function", -1)

    # order the new snps by size
    if new_snp1_name > new_snp2_name:
        new_snp1_name, new_snp2_name = new_snp2_name, new_snp1_name

    # make sure the new snps are set
    assert new_snp1_name != None and new_snp2_name != None
    return new_snp1_name, new_snp2_name

# execute a random mutation on the epi_node
def mutate_epi_node_rand(rng: np.random.Generator, node: EpiNode, hub: GenoHub,) -> Tuple[np.str_, np.str_]:
    # pull snp1 and snp2 from node
    snp1_name, snp2_name = node.get_snp1_name(), node.get_snp2_name()

    # will hold new interactions
    new_snp1_name, new_snp2_name = None, None

    # randomly select one of the SNPs
    new_snp1_name = rng.choice([snp1_name, snp2_name])

    # randomly select another SNP
    new_snp2_name = hub.get_ran_snp(rng)

    # make new snps are different
    while new_snp2_name == new_snp1_name:
        new_snp2_name = hub.get_ran_snp(rng)

    # order the new snps
    if new_snp1_name > new_snp2_name:
        new_snp1_name, new_snp2_name = new_snp2_name, new_snp1_name

    # make sure the new snps are set
    assert new_snp1_name != None and new_snp2_name != None
    return new_snp1_name, new_snp2_name

# execute a crossover between two pipelines
def crossover(rng: np.random.Generator, parent1: Pipeline, parent2: Pipeline) -> Pipeline:
    # get the epi branches from the parents
    p1_epi_pairs = list(parent1.epi_pairs())
    p2_epi_pairs = list(parent2.epi_pairs())

    # get smallest half length from both
    half_len = min(len(p1_epi_pairs), len(p2_epi_pairs)) // 2

    # randomly select indecies from both parents epi branches
    p1_idx = rng.choice(len(p1_epi_pairs), half_len, replace=False)
    p2_idx = rng.choice(len(p2_epi_pairs), half_len, replace=False)

    # swap elements between parents
    for i1, i2 in zip(p1_idx, p2_idx):
        p1_epi_pairs[i1], p2_epi_pairs[i2] = p2_epi_pairs[i2], p1_epi_pairs[i1]

    # create one offspring per parent
    offspring_1 = Pipeline(epi_branches=parent1.get_epi_branches(),
                           epi_pairs=set(p1_epi_pairs),
                           selector_node=parent1.get_selector_node(),
                           root_node=parent1.get_root_node(),
                           traits=parent1.get_traits())

    return None

def generate_epi_node(epi_pairs: Set[np.str_], rng: np.random.Generator, hub: GenoHub) -> EpiNode:


    return None

def generate_random_interaction(rng: np.random.Generator, hub: GenoHub) -> Tuple[np.str_, np.str_]:
    # randomly select two SNPs
    snp1_name, snp2_name = None, None
    snp1_name = hub.get_ran_snp(rng)
    snp2_name = hub.get_ran_snp(rng)

    # make new snps are different
    while snp2_name == snp1_name:
        snp2_name = hub.get_ran_snp(rng)

    if snp1_name < snp2_name:
        return snp1_name, snp2_name
    else:
        return snp2_name, snp1_name
