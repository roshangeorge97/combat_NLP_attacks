rounds=2
agents=2


python ./evaluate_model.py \
    --input-file ./data/red_team_small.txt \
    --output-file ./data/davinci002-single.txt \
    --model-type gpt \
    --openai-model-name text-davinci-002 

python ./evaluate_model.py \
    --input-file ./data/red_team_small.txt \
    --output-file ./data/turbo-single.txt \
    --model-type gpt \
    --openai-model-name gpt-3.5-turbo-0301 

'''
python ./evaluate_model.py \
    --input-file ./data/red_team_small.txt \
    --output-file ./data/davinci002-r${rounds}-a${agents}.txt \
    --openai-model-name text-davinci-002 \
    --model-type gpt-multiagent \
    --multiagent-rounds $rounds \
    --multiagent-agents $agents
'''

'''
python ./evaluate_model.py \
    --input-file ./data/red_team_small.txt \
    --output-file ./data/turbo-r${rounds}-a${agents}.txt \
    --openai-model-name gpt-3.5-turbo-0301 \
    --model-type gpt-multiagent \
    --multiagent-rounds $rounds \
    --multiagent-agents $agents
'''