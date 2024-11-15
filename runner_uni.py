# runner.py
import numpy as np
import ray
from Source.evolver import EA

def main():
    # Set experiment configurations directly
    ea_config = {
        'seed': np.uint16(1),
        'pop_size': np.uint16(50),
        'epi_cnt_max': np.uint16(0),
        'epi_cnt_min': np.uint16(0),
        'uni_cnt_max': np.uint16(50),
        'uni_cnt_min': np.uint16(20),
        'cores': 3,
        'mut_selector_p': np.float64(1.0),
        'mut_regressor_p': np.float64(0.5),
        'mut_ran_p': np.float64(0.5),
        'mut_smt_p': np.float64(0.5),
        'smt_in_in_p': np.float64(0.33),
        'smt_in_out_p': np.float64(0.33),
        'smt_out_out_p': np.float64(0.33),
        'mut_prob': np.float64(0.5),
        'cross_prob': np.float64(0.5),
        'num_add_interactions': np.uint16(10),
        'num_del_interactions': np.uint16(10),
        'save_directory': '/Users/yufeimeng/Desktop/pilot-star-base-epi_uni/uni_results'
    }

    ea = EA(**ea_config)

    # Provide data directory and bin size directly
    data_dir = '/Users/yufeimeng/Desktop/pilot-star-base-epi_uni/BMIwTail_qtl_10.csv'
    bin_size = 10
    gens = 20

    ea.data_loader(data_dir)
    ea.initialize_hubs(bin_size)
    ea.evolve(gens)
    ea.post_analysis()

    ray.shutdown()

if __name__ == "__main__":
    main()