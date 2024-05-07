# Defending LLMs Against Adversarial Attacks using Multi-Agent Debate

## Authors:
- Bharathi Mohan G.
- Balaji M.
- Prasanna Kumar R.
- Roshan George
- Mohamed Faheem M.

## Affiliation:
Dept. of CSE- AI, Amrita School of Computing, Amrita Vishwa Vidyapeetham, Chennai, India

## Contact:
- Bharathi Mohan G.: bharathimohan@ch.amrita.edu
- Balaji M.: balajimurugan5710@gmail.com
- Prasanna Kumar R.: prasannakumar@ch.amrita.edu
- Roshan George: roshangeorge2003@gmail.com
- Mohamed Faheem M.: mdfaheempasha@gmail.com

## Abstract:
The research study investigates the susceptibility of language models to adversarial attacks during inference and assesses the effectiveness of multi-agent debate in enhancing their robustness. Extensive experimentation with different adversarial attack methods and evaluation metrics demonstrates that multi-agent debate can significantly reduce model toxicity and improve overall robustness. The study concludes that multi-agent debate is a promising approach to enhance language model robustness in realistic environments.

## Index Terms:
- Conversational Chatbot
- Natural Language Processing
- Sequential Neural Network
- Bidirectional Encoder Representations from Transformers

## Introduction:
Previous studies have shown that large language models (LLMs) are vulnerable to attacks during training and inference. Adversarial methods such as data poisoning and adversarial generated prompts can cause LLMs to generate inappropriate outputs, posing risks to users. Enhancing the robustness of these models against adversarial prompts is crucial for real-world deployment. Multi-agent debate, where instances of a language model evaluate each other's responses, has shown promise in enhancing language models' factuality, reasoning, and performance on downstream tasks. This paper explores the effectiveness of multi-agent debate in mitigating adversarial attacks on LLMs during inference.

## Literature Review:
The literature review highlights the vulnerability of large language models to adversarial attacks and explores various defense methodologies. Multi-agent debate emerges as a promising approach to improve model robustness. However, more research is needed on debate dynamics against different adversarial threats and optimal agent configuration.

## Methodology:
### Multi-Agent Debate:
The study implements a multi-agent debate framework among LLMs, using prompt engineering to simulate the effects of poisoned models participating in debate. Various experiments are conducted under multi-agent settings to assess the effectiveness of multi-agent debate in mitigating adversarial vulnerabilities in LLMs.

### Red-Teaming Evaluation:
Adversarial prompts sourced from Anthropic's red teaming dataset are used to evaluate model responses to adversarial attacks. Clustering techniques are employed to classify different attack types observed during experimentation.

## Experimental Setup:
The methodology employed to investigate the effectiveness of multi-agent debate in mitigating adversarial vulnerabilities in language models is described. Data preparation, model selection, prompt design, experimental design, evaluation metrics, and implementation details are outlined.

## Results and Discussion:
The study evaluates the effectiveness of multi-agent debate in protecting LLMs against adversarial attacks. Findings indicate that multi-agent debate lowers response toxicity compared to baselines like Self-Refine. However, an LLM agent generating toxic content may negatively influence other agents. Future research directions are discussed to improve the efficacy of multi-agent debate in real-world applications.

## Conclusion:
The research demonstrates the potential of multi-agent debate in enhancing LLM robustness against adversarial attacks during inference. While not a perfect solution, multi-agent debate shows promise and warrants further exploration to develop more efficient and effective model guardrails.

