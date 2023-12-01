import sys, os
import openai
import time

def generate_response_openai(context, modelname):
    fails = 0
    if modelname.startswith("gpt-3.5"):
        while(fails < 5):
            try:
                completion = openai.ChatCompletion.create(
                    model=modelname,
                    messages=context,
                    n=1,
                    temperature=1,
                    max_tokens=256,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                return completion["choices"][0]["message"]["content"].strip().replace("\n", " ")
            except Exception as e:
                print("retrying due to an error......")
                print(e)
                time.sleep(2)
                fails += 1
        raise Exception("API failed to respond after 5 attempts.") 

    elif modelname.startswith("davinci") or modelname.startswith("text-davinci"):
        context_str = "\n".join(
            [f"{message['role']}: {message['content']}" for message in context[-1:]]
        )
        while(fails < 5):
            try:
                response = openai.Completion.create(
                engine=modelname,
                prompt=context_str,
                n=1,
                temperature=1.0,
                max_tokens=512,
                top_p=1.0,
                frequency_penalty=0,
                presence_penalty=0
                )
                return response["choices"][0]["text"].strip().replace("\n", " ")
            except:
                print("retrying due to an error......")
                time.sleep(2)
                fails += 1
        raise Exception("API failed to respond after 5 attempts.")
    else:
        raise NotImplementedError(f"{modelname} model not implemented.")

def generate_response_llama(pipeline, dialog):
    context_str = " ".join([message["content"] for message in dialog])
    num_try = 0
    print("\n\nCONTEXT: ", context_str, "\n\n")
    while num_try<=5:
        outputs = pipeline(context_str)
        generation = outputs[0]["generated_text"][len(context_str):].strip()
        generation = " ".join(generation.split()[:256])
        '''
        if "### RESPONSE:" in generation:
            # Uncensored LLAMA
            generation = generation.split("### RESPONSE:")[-1].strip().replace("\n", " ")
        else:
            # Normal LLAMA
            generation = generation.split("[/INST]")[-1].strip().replace("\n", " ")
        '''
        generation = generation.strip().replace("\n", " ")
        if len(generation.strip().split()) > 20:
            return generation
        num_try += 1
    print("No long answers.")
    return generation