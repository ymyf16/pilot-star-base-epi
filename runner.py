import argparse
import numpy as np
import ray
from Source.evolver import EA

def main(args):
    # set experiment configurations using arguments from SLURM array job
    ea_config = {
        'seed': np.uint16(args.seed),
        'pop_size': np.uint16(args.pop_size),
        'epi_cnt_max': np.uint16(args.epi_cnt_max),
        'epi_cnt_min': np.uint16(args.epi_cnt_min),
        'uni_cnt_max': np.uint16(args.uni_cnt_max),
        'uni_cnt_min': np.uint16(args.uni_cnt_min),
        'cores': args.cores,
        'mut_selector_p': np.float64(1.0),  # This remains constant
        'mut_regressor_p': np.float64(.5),  # This remains constant
        'mut_ran_p': np.float64(args.mut_ran_p),
        'mut_smt_p': np.float64(args.mut_smt_p),
        'smt_in_in_p': np.float64(args.smt_in_in_p),
        'smt_in_out_p': np.float64(args.smt_in_out_p),
        'smt_out_out_p': np.float64(args.smt_out_out_p),
        'mut_prob': np.float64(.5),         # Assuming constant
        'cross_prob': np.float64(.5),       # Assuming constant
        'num_add_interactions': np.uint16(args.num_add_interactions),
        'num_del_interactions': np.uint16(args.num_del_interactions),
        'save_directory': args.save_directory
    }

    ea = EA(**ea_config)

    # Update to use the data_dir passed as argument
    ea.data_loader(args.data_dir)
    ea.initialize_hubs(args.bin_size)
    ea.evolve(args.gens)
    ea.post_analysis()

    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EA with SLURM parameters")

    # Match arguments from SLURM job
    parser.add_argument('--seed', type=int, required=True, help="Random seed")
    parser.add_argument('--pop_size', type=int, default=100, help="Population size")
    parser.add_argument('--epi_cnt_max', type=int, default=200, help="Max episode count")
    parser.add_argument('--epi_cnt_min', type=int, default=10, help="Min episode count")
    parser.add_argument('--uni_cnt_max', type=int, default=200, help="Max uni episode count")
    parser.add_argument('--uni_cnt_min', type=int, default=10, help="Min uni episode count")
    parser.add_argument('--cores', type=int, default=10, help="Number of cores")
    parser.add_argument('--mut_ran_p', type=float, default=0.45, help="Mutation random probability")
    parser.add_argument('--mut_smt_p', type=float, default=0.45, help="Mutation smooth probability")
    parser.add_argument('--smt_in_in_p', type=float, default=0.10, help="Smooth in-in probability")
    parser.add_argument('--smt_in_out_p', type=float, default=0.450, help="Smooth in-out probability")
    parser.add_argument('--smt_out_out_p', type=float, default=0.450, help="Smooth out-out probability")
    parser.add_argument('--num_add_interactions', type=int, default=10, help="Number of interactions to add")
    parser.add_argument('--num_del_interactions', type=int, default=10, help="Number of interactions to delete")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the data file")
    parser.add_argument('--save_directory', type=str, required=True, help="Path to the results director")
    parser.add_argument('--bin_size', type=int, required=True, help="Bin size")
    parser.add_argument('--gens', type=int, required=True, help="Number of generations")

    args = parser.parse_args()
    main(args)