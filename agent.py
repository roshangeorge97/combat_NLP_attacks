import openai
import os
import numpy as np
from abc import ABC, abstractmethod
from utils import generate_response_openai

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
    
class gpt_agent(model_wrapper):
    def __init__(self, modelname="gpt-3.5-turbo-0301"):
        self.agent_modelname = modelname

    def generate(self, context):
        if isinstance(context, str):
            context = [{"role": "user", "content": context}]
        completion = generate_response_openai(context, self.agent_modelname)
        return completion

    def construct_assistant_message_from_completion(self, completion):
        return {"role": "assistant", "content": completion}
    
    def construct_initial_message(self, prompt):
        return {"role": "user", "content": prompt}
        
    def construct_discussion_message(self, topic, feedback_ls):

        if len(feedback_ls)==0:
            return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer for the topic '''{}'''.".format(topic)}
        
        prefix_string = "These are the recent/updated opinions from other agents: "

        for feedback in feedback_ls:
            response = "One agent response: ```{}```".format(feedback)
            prefix_string = prefix_string + response

        prefix_string = prefix_string + " Use these opinions carefully as additional advice, can you provide an updated answer for the topic '''{}'''?".format(topic)

        return {"role": "user", "content": prefix_string}      

    
class gpt_agent_group(model_wrapper):
    def __init__(self, n_agents=2, n_discussion_rounds=0, modelname="gpt-3.5-turbo-0301"): 
        self.n_agents = n_agents
        self.n_discussion_rounds = n_discussion_rounds
        self.agents = [gpt_agent(modelname) for _ in range(n_agents)]

    def select_final_response(self, agent_contexts):
        return agent_contexts[0][-1]["content"]

    def generate(self, initial_prompt, n_agents=0, n_discussion_rounds=0):
        agent_contexts = [   
            [ag.construct_initial_message(initial_prompt)] for ag in self.agents
        ]
        for i, ag in enumerate(self.agents):
            agent_response = ag.construct_assistant_message_from_completion(
                    ag.generate(agent_contexts[i]))
            agent_contexts[i].append(agent_response)
        
        for _ in range(self.n_discussion_rounds):
            for i, ag in enumerate(self.agents):
                # Create new prompt from other agents' feedback or self-reflection
                agents_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                feedback_ls = [context[-1]["content"] for context in agents_contexts_other]
                discussion_message = ag.construct_discussion_message(initial_prompt, feedback_ls)
                agent_contexts[i].append(discussion_message)

                # Create new answer
                agent_response = ag.construct_assistant_message_from_completion(ag.generate(agent_contexts[i]))
                agent_contexts[i].append(agent_response)

        for i, agent_context in enumerate(agent_contexts):
            print(f"Agent {i}: ")
            for message in agent_context:
                print(f"\t{message['role']}: {message['content']}\n")
        return self.select_final_response(agent_contexts)  


            

