modelname=llama2_7b_chat_uncensored
modelname=llama-2-7b-chat-hf
intent=harmful
modelname=$1
intent=$2

'''
sbatch -c 8 -N 1 --mem=45000 -p gpu --gres=gpu:1 --exclude=boston-2-[25,27,29,31,35] \
    --wrap="python3 -u ./evaluate_model.py \
    --input-file ./data/red_teams_small.txt \
    --output-file ./responses/${modelname}-${intent}-single-agent.txt \
    --model-type llama \
    --llama-model-name $modelname \
    --agent-intention $intent"

srun -c 8 -N 1 --mem=45000 -p gpu --gres=gpu:1 --exclude=boston-2-[25,27,29,31] \
    python3 -u ./evaluate_model.py \
    --input-file ./data/red_teams_small.txt \
    --output-file ./responses/tr${modelname}-${intent}-single-agent.txt \
    --model-type llama \
    --llama-model-name $modelname \
    --agent-intention $intent
'''
srun -c 16 -N 1 --mem=90000 -p gpu --gres=gpu:2 --exclude=boston-2-[25,27,29,31] \
    python3 -u ./evaluate_model.py \
    --input-file ./data/red_teams_small.txt \
    --output-file ./responses/tr${modelname}-${intent}-self-reflect.txt \
    --model-type multiagent \
    --agent-modelname $modelname \
    --agent-intention $intent \
    --multiagent-agents 2 \
    --multiagent-rounds 2