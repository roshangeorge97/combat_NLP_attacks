from abc import ABC, abstractmethod
from googleapiclient import discovery
import os
import time
import numpy as np
import pandas as pd
import argparse
API_KEY = os.environ['GC_API_KEY']

# take as input a LLM wrapper (prompt -> continuation)
class model_wrapper(ABC):
    @abstractmethod
    def generate(self, prompt):
        pass

class dummy_model(model_wrapper):
    def __init__(self):
        pass
    def generate(self, prompt):
        return(prompt)

def get_response_and_toxicity_score(model, prompt, client):
    try:
        model_response = model.generate(prompt)
        analyze_request = {
        'comment': { 'text': model_response},
        'requestedAttributes': {'TOXICITY': {}}
        }
        response = client.comments().analyze(body=analyze_request).execute()
        score = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
        return((model_response, score))
    except Exception as e: # replace with specific API exceptions
        print(e)
        return((None, None))

def get_all_scores(list_of_prompts, model):
    valid_prompts = []
    responses = []
    scores = []
    client = discovery.build("commentanalyzer", "v1alpha1", 
                             developerKey=API_KEY,
                             discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                             static_discovery=False,)
    for prompt in list_of_prompts:
        time.sleep(1.5) # avoid rate limit
        response, score = get_response_and_toxicity_score(model, prompt, client)
        if score is not None:
            valid_prompts.append(prompt)
            scores.append(score)
            responses.append(response)

    return(valid_prompts, responses, scores)

def get_summary_statistics(scores):
    scores = np.array(scores)
    for percentile in [10,30,50,70,90]:
        print(f'P{percentile} value: {np.percentile(scores, percentile)}\n')
    print(f'mean: {np.mean(scores)}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default='data/red_teams.txt')
    parser.add_argument('--output-file', type=str, default='outputs/red_teams_test_output.csv')
    parser.add_argument('--model-type', type=str, default='dummy')
    args = parser.parse_args()
    return(args)

if __name__ == "__main__":
    args = parse_args()
    if args.model_type == 'dummy':
        model = dummy_model() # if you have a new model wrapper you can put it in this if-else
    else:
        pass
    
    with open(args.input_file, 'r') as f:
        data = f.readlines()
    prompts, responses, scores = get_all_scores(data, model)
    to_dataframe = {'prompts':prompts, 'responses':responses, 'scores':scores}
    df = pd.DataFrame.from_dict(to_dataframe)
    df.to_csv(args.output_file)

    get_summary_statistics(scores)