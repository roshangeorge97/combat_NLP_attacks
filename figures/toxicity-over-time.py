import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

p10 = [0.323, 0.173, 0.152, 0.165, 0.151, 0.145]
mean = [0.62, 0.467, 0.449, 0.423, 0.411, 0.392]
p90 = [0.865, 0.836, 0.829, 0.779, 0.801, 0.746]
x = np.arange(len(p10))

plt.plot(x, p10, label='p10 tox. score')
plt.plot(x, mean, label='mean tox. score')
plt.plot(x, p90, label='p90 tox. score')
plt.legend()
plt.title('Llama-chat-unc. toxicity (toxic prompt+harmless feedback)')
plt.ylabel('Response toxicity score')
plt.xlabel('Rounds of Debate')
plt.savefig('repeated_interaction1.png')

plt.figure()
p10 = [0.123, 0.193, 0.200, 0.191, 0.156, 0.169]
mean= [0.358, 0.441, 0.435, 0.426, 0.414, 0.418]
p90 = [0.795, 0.818, 0.748, 0.771, 0.751, 0.774]
x = np.arange(len(p10))

plt.plot(x, p10, label='p10 tox. score')
plt.plot(x, mean, label='mean tox. score')
plt.plot(x, p90, label='p90 tox. score')
plt.legend()
plt.title('Llama-chat-unc. toxicity (harmless prompt+toxic feedback)')
plt.ylabel('Response toxicity score')
plt.xlabel('Rounds of Debate')
plt.savefig('repeated_interaction2.png')