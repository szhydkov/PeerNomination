import sys, os

from sqlalchemy import false
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

### Standard Magic and startup initializers.

import math
import numpy as np
import random
import itertools
import pandas as pd
import multiprocessing

from peerselect import impartial
from peerselect import profile_generator
from peerselect.estimate_eps import estimate_eps

class Impartial:
    VANILLA = "Vanilla"
    EXACT = "ExactDollarPartition"
    PARTITION = "Partition"
    DPR = "DollarPartitionRaffle"
    CREDIABLE = "CredibleSubset"
    RAFFLE = "DollarRaffle"
    NOMINATION = "PeerNomination"
    ALL = (VANILLA, EXACT, PARTITION, RAFFLE, CREDIABLE, DPR, NOMINATION)

# Run the experiment for one parameter set s times.

def run_experiment(n, k, m, l, p, s):
    gt_results = {}
    pn_sizes = {}
    is_m_regular = {}
    is_m_regular_run = np.ones(s, dtype = bool)
    agents = np.arange(0, n)
    
    eps_corrected = estimate_eps(n, m, k)
    
    for c_sample in range(s):
        # Catch the failed EDP allocations and restard the current iteration
        failed_iterations = 0
        while failed_iterations <= 10:
            try:
                # Generate a profile and clustering
                profile = profile_generator.generate_mallows_mixture_profile(agents, agents, [p, 1-p], [agents, agents], [0.0, 1.0])
                clustering = impartial.even_partition_order(sorted(agents, key=lambda j: random.random()), l)
                
                # Borda -- need to start at 1 to distinguish from non-review in the score matrix
                scores = np.arange(m, 0, -1)
                
                # Approval a-la PeerNomination
                nom_quota = (k/n)*m
                scores_pn = np.concatenate((np.ones(math.floor(nom_quota)),
                        [nom_quota-math.floor(nom_quota)],
                        np.zeros(m-math.ceil(nom_quota))))
                
                # Generate an m-regular assignment
                m_assignment = profile_generator.generate_approx_m_regular_assignment(agents, m, clustering, randomize=False)
                
                score_matrix = profile_generator.strict_m_score_matrix(profile, m_assignment, scores)
                for i in range(n):
                    if sum(score_matrix[i, :] != 0) != m or sum(score_matrix[:, i] != 0) != m:
                        is_m_regular_run[c_sample] = False
                score_matrix_pn = profile_generator.strict_m_score_matrix(profile, m_assignment, scores_pn)
                
                # Capture the winning sets
                ws = {}

                # Run peer nomination using the estimated epsilon       
                ws[Impartial.VANILLA] = [i for i,j in impartial.vanilla(score_matrix, k)]
                ws[Impartial.NOMINATION] = impartial.peer_nomination_lottery(score_matrix, k, eps_corrected)
                ws[Impartial.EXACT] = impartial.exact_dollar_partition_explicit(score_matrix, k, clustering, normalize=True)
                ws["EDP_new"] = impartial.exact_dollar_partition_explicit(score_matrix_pn, k, clustering, normalize=True)
                
                for x in [Impartial.VANILLA, Impartial.NOMINATION, Impartial.EXACT, "EDP_new"]:
                    key = (n, k, m, l, p, s, x)
                    gt_results[key] = gt_results.get(key, []) + [len(set(np.arange(0, k)) & set(ws[x]))]
                    pn_sizes[key] = pn_sizes.get(key, []) + [len(set(ws[x]))]
                
                key = (n, k, m, l, p)
                failed_iterations = 0
                break

            except BaseException:
                failed_iterations += 1
                # print("failed iterations", failed_iterations)
                continue

        if failed_iterations >= 10:
            print("Too many failed allocations! Skipping these parameters.")
            return None

        is_m_regular[(n, k, m, l, p, s,)] = is_m_regular_run
            
    print("Finished: " + ",".join([str(x) for x in [n, k, m, l, p, s]]))
    return gt_results, pn_sizes, is_m_regular

_DEBUG = False

if __name__ == '__main__':

    # Parameters

    #random.seed(15)

    s = 100
    test_n = [120]
    test_k = [20, 25]
    test_m = [7, 8, 9]
    test_l = [4]
    test_p = [0.5, 1]

    # Run the experiments in parallel.

    pool = multiprocessing.Pool()
    outputs = pool.starmap(run_experiment, itertools.product(test_n, test_k, test_m, test_l, test_p, [s]))

    gt_results_combined = {}
    pn_sizes_combined = {}
    is_m_reg_comb = {}
    for output in outputs:
        if output == None:
            continue
        gt_results, pn_sizes, is_m_reg = output
        gt_results_combined.update(gt_results)
        pn_sizes_combined.update(pn_sizes)
        is_m_reg_comb.update(is_m_reg)
        # print(gt_results_combined)
        # print()


    df = pd.DataFrame(gt_results_combined)
    df.columns.names = ['n', 'k', 'm', 'l', 'p', 's', 'algo']

    df_sizes = pd.DataFrame(pn_sizes_combined)
    df_sizes.columns.names = ['n', 'k', 'm', 'l', 'p', 's', 'algo']

    df.to_pickle("borda_results.pkl")
    df_sizes.to_pickle("borda_sizes.pkl")