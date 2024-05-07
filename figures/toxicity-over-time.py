import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

p10 = [0.03, 0.02, 0.02, 0.01, 0.02, 0.03]
mean = [0.12, 0.06, 0.08, 0.09, 0.13, 0.18]
p90 = [0.22,  0.09, 0.12, 0.14, 0.13, 0.14]
x = np.arange(len(p10))

plt.plot(x, p10, label='p10 tox. score')
plt.plot(x, mean, label='mean tox. score')
plt.plot(x, p90, label='p90 tox. score')
plt.legend()
plt.title('GPT-3.5 tubo. toxicity (toxic prompt+harmless feedback)')
plt.ylabel('Response toxicity score')
plt.xlabel('Rounds of Debate')
plt.savefig('repeated_interaction1.png')

plt.figure()
p10 = [0.02, 0.02, 0.032, 0.038, 0.036, 0.022]
mean= [0.08, 0.06, 0.07, 0.08, 0.07, 0.09]
p90 = [0.13, 0.06, 0.04, 0.04, 0.06, 0.08]
x = np.arange(len(p10))

plt.plot(x, p10, label='p10 tox. score')
plt.plot(x, mean, label='mean tox. score')
plt.plot(x, p90, label='p90 tox. score')
plt.legend()
plt.title('GPT-3.5 tubo. toxicity (neutral prompt+toxic feedback)')
plt.ylabel('Response toxicity score')
plt.xlabel('Rounds of Debate')
plt.savefig('repeated_interaction2.png')