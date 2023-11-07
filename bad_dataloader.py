import json

def fit_to_prompt_template(messages, labels):
    prompt = f'Given the following list of statements, label the final statement as {labels["ok"]} or {labels["notok"]}\n'
    for idx, message in enumerate(messages):
        prompt += message
        prompt += '\n'
    prompt += 'Label: '
    return(prompt)

with open('/home/andyliu/Miniconda3/envs/anlp-hw/lib/python3.10/site-packages/data/bot_adversarial_dialogue/dialogue_datasets/bot_adversarial_dialogue_datasets_with_persona/train.txt', 'r') as f:
    bad_data = f.read().split('\n')

to_json = []
labels = {'notok':'offensive', 'ok':'inoffensive'}
for line in bad_data:
    if line != '':
        text = line.split('\t')[0].replace('text:', '').split('\\n')
        prompt = fit_to_prompt_template(text, labels)
        label = labels[line.split('\t')[1].replace('labels:', '').replace('_', '')]
        to_json.append({'text':prompt, 'label':label})

with open('bad_data.json', 'w') as f:
    json.dump(to_json, f)