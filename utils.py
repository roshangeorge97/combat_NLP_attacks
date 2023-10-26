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
            [f"{message['role']}: {message['content']}" for message in context]
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
        raise NotImplementedError("Model not implemented.")
    