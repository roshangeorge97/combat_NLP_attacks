import openai
import os
import json
import numpy as np
from abc import ABC, abstractmethod
from utils import generate_response_openai, generate_response_llama
import torch
from transformers import AutoTokenizer

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
    def __init__(self, modelname="gpt-3.5-turbo-0301", intention="neutral", idx=0):
        self.agent_modelname = modelname
        self.intention = intention
        self.idx = idx
        print(f"Using model {self.agent_modelname} with intention {self.intention}.")
        self.cached_response = None 

    def generate(self, context):
        if self.cached_response is not None and (self.intention=="harmless" or self.intention=="harmful"):
            return self.cached_response
        
        if isinstance(context, str):
            context = [{"role": "user", "content": context}]
        completion = generate_response_openai(context, self.agent_modelname)
        if self.intention=="harmless" or self.intention=="harmful":
            self.cached_response = completion
        
        return completion

    def construct_assistant_message_from_completion(self, completion):
        return {"role": "assistant", "content": completion}
    
    def construct_initial_message(self, prompt):
        self.cached_response = None
        if self.intention == "harmless" or self.intention == "harmful":
            prompt = single_agent_prompts[self.intention].format(prompt)
        return {"role": "user", "content": prompt}
        
    def construct_discussion_message(self, topic, feedback_ls):

        if len(feedback_ls)==0:
            return {"role": "user", "content": single_agent_prompts[self.intention].format(topic)}
        
        prefix_string = "These are the recent/updated opinions from other agents: "

        for feedback in feedback_ls:
            response = "One agent response: ```{}```".format(feedback)
            prefix_string = prefix_string + response

        prefix_string = prefix_string + " Use these opinions carefully as additional advice, can you provide an updated answer for the topic '''{}'''?".format(topic)

        return {"role": "user", "content": prefix_string}      
    
class llama_agent(model_wrapper):
    def __init__(self, modelname="llama-2-7b-chat-hf", intention="neutral", idx=0, device="cuda:0"):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        self.agent_modelname = modelname
        self.intention = intention
        self.idx = idx
        self.device = f"cuda:{idx}"
        if "uncensored" in modelname:
            self.prompts = json.load(open("./prompts/llamachat-unc_prompts.json", "r"))
        elif "chat" in modelname:
            self.prompts = json.load(open("./prompts/llamachat_prompts.json", "r"))
        else:
            self.prompts = json.load(open("./prompts/llama_prompts.json", "r"))
        self.cached_response = None 
        print(f"Using model {self.agent_modelname} with intention {self.intention}.")


        # Load the model
        model_name_or_path = modelname
        int8 = False
        self.model = AutoModelForCausalLM.from_pretrained(os.environ["LLAMA_ROOT"]+model_name_or_path,
            torch_dtype=torch.float16,
            load_in_8bit=int8,
            max_memory=self.get_max_memory(),
        ).to(self.device)
        print("Model loaded")
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(os.environ["LLAMA_ROOT"]+model_name_or_path, use_fast=False)
        
        self.pipeline=pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=4096,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
            device=self.device,
        )

    def generate(self, context):
        if isinstance(context, str):
            context = [self.construct_initial_message(context)]
        # For models with intentions, directly output the result.
        #if self.cached_response is not None and (self.intention=="harmless" or self.intention=="harmful"):
        #    return self.cached_response
        # If it is the initial prompt, set up the context as a list of dicts.
        #completion = generate_response_llama(self.model, self.tokenizer, context)
        completion = generate_response_llama(self.pipeline, context)

        if self.intention=="harmless" or self.intention=="harmful":
            self.cached_response = completion
        
        return completion

    def construct_assistant_message_from_completion(self, completion):
        return {"role": "assistant", "content": completion}
    
    def construct_initial_message(self, prompt):
        self.cached_response = None
        return {"role": "user", "content": self.prompts["init"][self.intention].replace("<TOPIC>", prompt)}
        
    def construct_discussion_message(self, topic, feedback_ls):
        if len(feedback_ls)==0:
            return {"role": "user", "content": self.prompts["self-reflect"][self.intention].replace("<TOPIC>", topic)}
        feedbacks = []
        for feedback in feedback_ls:
            feedbacks.append("One agent response: ```{}```".format(feedback))
        feedbacks = " ".join(feedbacks)
        prefix_string = self.prompts["discussion"][self.intention].replace("<TOPIC>", topic).replace("<FEEDBACK>", feedbacks)

        return {"role": "user", "content": prefix_string}   

    def get_max_memory(self):
        """Get the maximum memory available for the current GPU for loading models."""
        free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
        max_memory = f'{free_in_GB-6}GB'
        n_gpus = torch.cuda.device_count()
        max_memory = {i: max_memory for i in range(n_gpus)}
        return max_memory   
    
class agent_group(model_wrapper):
    def __init__(self, n_agents=2, n_discussion_rounds=0, modelname="llama-2-7b-chat", intention="neutral"): 
        self.n_agents = n_agents
        self.n_discussion_rounds = n_discussion_rounds
        if isinstance(modelname, str):
            modelname = [modelname] * n_agents
        if isinstance(intention, str):
            intention = [intention] * n_agents
        self.agents = []
        for i in range(n_agents):
            if "llama" in modelname[i]:
                self.agents.append(llama_agent(modelname[i], intention[i], i))
            else:
                self.agents.append(gpt_agent(modelname[i], intention[i]))

    def select_final_response(self, agent_contexts):
        ret = []
        for i in range(self.n_agents):
            ret.append([msg["content"] for msg in agent_contexts[i][1::2]])
        if len(ret[0]) > 0:
            return ret
        return "No answer."

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
        '''
        for i, agent_context in enumerate(agent_contexts):
            print(f"Agent {i}: ")
            for message in agent_context:
                print(f"\t{message['role']}: {message['content']}\n")
        '''
        return self.select_final_response(agent_contexts)  
