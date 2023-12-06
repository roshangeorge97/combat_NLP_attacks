import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

p10 = [0.323, 0.173, 0.152, 0.165, 0.151, 0.145]
mean = [0.62, 0.467, 0.449, 0.423, 0.411, 0.392]
p90 = [0.865, 0.836, 0.829, 0.779, 0.801, 0.746]
x = np.arange(len(p10))

plt.plot(x, p10, label='p10 toxicity score')
plt.plot(x, mean, label='mean toxicity score')
plt.plot(x, p90, label='p90 toxicity score')
plt.legend()
plt.title('Repeated Debate Lowers Toxicity')
plt.ylabel('Toxicity of Model Responses')
plt.xlabel('Rounds of Debate')
plt.savefig('repeated_interaction.png')