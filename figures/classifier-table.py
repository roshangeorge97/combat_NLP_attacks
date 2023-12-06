import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

x = np.arange(2)*2
labels = ['Single-Turn', 'Multi-Turn']
models = ['BAD Classifier', 'Finetuned Llama-7b', 'Perspective API']
bad_scores = [0.783, 0.803]
llama_scores = [0.746, 0.763]
perspective_scores = [0.464, 0.464]

scores = [bad_scores, llama_scores, perspective_scores]

plt.bar(x - 1/3, scores[0], label=models[0], align='center', color='dimgray', width=0.3)
plt.bar(x, scores[1], label=models[1], align='center', color='lightblue', width=0.3)
plt.bar(x + 1/3, scores[2], label=models[2], align='center', color='darkgray', width=0.3)

plt.ylabel('BAD Classification F1')
plt.xticks(x, labels)
plt.legend()
plt.title('Toxicity Classification Performance by Model')
plt.savefig('new_table.png')