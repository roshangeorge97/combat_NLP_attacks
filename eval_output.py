import os, sys
import time
import numpy as np


def get_summary_statistics(scores):
    scores = np.array(scores) # [N_CASE, N_AGENTS, N_ROUNDS]
    for ag in range(scores.shape[1]):
        for rd in range(scores.shape[2]):
            print(f"Agent {ag} round {rd} scores:")
            for percentile in [10,30,50,70,90]:
                print(f'\tP{percentile} value: {np.percentile(scores[:,ag,rd].squeeze(), percentile)}')
            print(f'\tmean: {np.mean(scores[:,ag,rd])}')
            print(f'\tPercentage>0.1: {np.mean(scores[:,ag,rd]>=0.1)}')

with open(sys.argv[1], 'r') as f:
    # read dataframe from file
    data = f.readlines()[1:]
    scores = [float(x.split(",")[-1]) for x in data]
    scores = np.array(scores).reshape(-1,1,1)
    get_summary_statistics(scores)

