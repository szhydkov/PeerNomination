import sys, os
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
    agents = np.arange(0, n)
    
    eps_corrected = estimate_eps(n, m, k)
    
    for c_sample in range(s):
        # Catch the failed EDP allocations and restard the current iteration
        failed_iterations = 0
        while failed_iterations <= 10:
            try:
                # Generate a profile and clustering
                profile = profile_generator.generate_mallows_mixture_profile(agents, agents, [p, 1-p], [agents, agents[::-1]], [0.8, 0.8])
                clustering = impartial.even_partition_order(sorted(agents, key=lambda j: random.random()), l)
                
                # Borda -- need to start at 1 to distinguish from non-review in the score matrix
                scores = np.arange(m, 0, -1)
                
                # Approval a-la PeerNomination
                nom_quota = (k/n)*m
                scores_pn = np.concatenate((np.ones(math.floor(nom_quota)),
                        [nom_quota-math.floor(nom_quota)],
                        np.zeros(m-math.ceil(nom_quota))))
                
                # Generate an m-regular assignment
                m_assignment = profile_generator.generate_sp_assignment(agents, m)

                score_matrix = profile_generator.strict_m_score_matrix(profile, m_assignment, scores)
                score_matrix_pn = profile_generator.strict_m_score_matrix(profile, m_assignment, scores_pn)
                
                # Capture the winning sets
                ws = {}

                # Run peer nomination using the estimated epsilon
                ws[Impartial.NOMINATION] = impartial.peer_nomination_lottery(score_matrix, k, eps_corrected)
                ws["PN_dist"] = impartial.weighted_peer_nomination(score_matrix, k, impartial.dist_weights, 0.0)
                ws["PN_maj"] = impartial.weighted_peer_nomination(score_matrix, k, impartial.maj_weights, eps_corrected)
                ws["PN_step"] = impartial.weighted_peer_nomination(score_matrix, k, impartial.step_weights, eps_corrected)

                for x in [Impartial.NOMINATION, "PN_dist", "PN_maj", "PN_step"]:
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
            
    print("Finished: " + ",".join([str(x) for x in [n, k, m, l, p, s]]))
    return gt_results, pn_sizes

_DEBUG = False

if __name__ == '__main__':

    # Parameters

    #random.seed(15)

    s = 1000
    test_n = [120]
    test_k = [20, 25, 30]
    test_m = [7, 8, 9, 10]
    test_l = [4]
    test_p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # Run the experiments in parallel.

    pool = multiprocessing.Pool()
    outputs = pool.starmap(run_experiment, itertools.product(test_n, test_k, test_m, test_l, test_p, [s]))

    gt_results_combined = {}
    pn_sizes_combined = {}
    for output in outputs:
        if output == None:
            continue
        gt_results, pn_sizes = output
        gt_results_combined.update(gt_results)
        pn_sizes_combined.update(pn_sizes)
        # print(gt_results_combined)
        # print()


    df = pd.DataFrame(gt_results_combined)
    df.columns.names = ['n', 'k', 'm', 'l', 'p', 's', 'algo']

    df_sizes = pd.DataFrame(pn_sizes_combined)
    df_sizes.columns.names = ['n', 'k', 'm', 'l', 'p', 's', 'algo']

    df.to_pickle("sp_asgn_results.pkl")
    df_sizes.to_pickle("sp_asgn_sizes.pkl")