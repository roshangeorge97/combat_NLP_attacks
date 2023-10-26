import json
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-file', type=str, default='data/red_teams.txt')
    parser.add_argument('--percentile', type=float, default=5,
                        help='take most toxic x%% of red team attempts')
    args = parser.parse_args()
    return(args)

def format_string(s):
    # returns first sentence of a red teaming transcript 's'
    return(s.split('\n\n')[1].replace('Human: ', ''))

if __name__ == '__main__':
    args = parse_args()
    with open('data/red_team_attempts.jsonl', 'r') as red_team_json:
        red_team_data = json.load(red_team_json)

    # get the inputs with highest toxicity score
    transcripts = [red_team['transcript'] for red_team in red_team_data]
    scores = np.array([red_team['min_harmlessness_score_transcript'] for red_team in red_team_data])
    above_threshold_indices = list(np.argwhere(scores < np.percentile(scores, args.percentile)).flatten())
    transcripts = [transcripts[idx] for idx in above_threshold_indices]

    # write first line of each transcript to file
    with open(args.output_file, 'w') as f:
        for transcript in transcripts:
            f.write(format_string(transcript) + '\n')