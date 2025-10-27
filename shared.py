import json
import os
import shutil
import time

import nltk
import numpy as np
import regex
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

einfra_models_all = ["gpt-oss-120b",
                     "deepseek-r1",
                     "medgemma:27b-it",
                     "mistral-small3.2:24b-instruct-2506-q8_0",
                     "phi4:14b-q8_0",
                     "aya-expanse:32b",
                     "llama-4-scout-17b-16e-instruct",
                     "mistral-small3.1:24b-instruct-2503-q8_0",
                     "llama3.3:latest",
                     "gemma3:27b-it"]

einfra_models_selected = ["medgemma:27b-it",
                          "mistral-small3.2:24b-instruct-2506-q8_0",
                          "phi4:14b-q8_0",
                          "aya-expanse:32b",
                          "llama-4-scout-17b-16e-instruct",
                          "mistral-small3.1:24b-instruct-2503-q8_0",
                          "gemma3:27b-it", "gpt-oss-120b"]
def probstostr(probs):
    return [";".join([f"{y.token}:{y.logprob:.2f}" for y in x]) for x in probs]


class ModelWrapper:
    def __init__(self, client: Agent, model_name="gpt-4.1-nano"):
        self.client = client
        self.model_name = model_name

    def run(self, user_prompt,
            system_prompt="You are a professor grading a student test from given material, create the OUTPUT string as requested.",
            top_p=0.1, logprobs=False, marker=""):
        top_p = 0.001 if top_p == 0 else top_p
        model_name = self.model_name
        client = self.client
        if model_name == "gpt-5-nano" or model_name == "gemini-2.0-flash-lite":
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system",
                     "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            answer = response.choices[0].message.content.strip()
        elif model_name.startswith("gpt-4"):
            response = client.chat.completions.create(
                model=model_name,
                top_p=top_p,
                logprobs=logprobs,
                top_logprobs=5 if logprobs else None,
                frequency_penalty=0 if top_p <= 0.1 else 2,
                messages=[
                    {"role": "system",
                     "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            answer = response.choices[0].message.content.strip()
        else:
            model_settings = OpenAIChatModelSettings.fromkeys(client.model.settings.items())
            model_settings["top_p"] = top_p
            client = Agent(client.model, system_prompt=system_prompt, model_settings=model_settings)
            response = client.run_sync(user_prompt=user_prompt)
            answer = str(response.output)
        if os.path.isfile("debug.txt"):
            with open("debug.txt", "a") as f:
                sysp = f"SYSTEM PROMPT:\n{system_prompt}\n" if False else ""
                f.write(
                    sysp + f"USER PROMPT:\n{user_prompt[-2000:]}\nASSISTANT RESPONSE:\n{answer}\n{marker}{'-' * 80}\n")
        return answer, [[y for y in x.top_logprobs] for x in
                        response.choices[0].logprobs.content] if logprobs else []

def make_clienteinfra(index, system_prompt):
    model = OpenAIChatModel(
        einfra_models_selected[index],
        provider=OpenAIProvider(
            base_url='https://chat.ai.e-infra.cz/api',
            api_key=os.getenv('E_INFRA_API_TOKEN'),
        ),
        settings=ModelSettings(top_p=0.1)
    )
    agent = Agent(model=model, system_prompt=system_prompt)
    return ModelWrapper(agent, einfra_models_selected[index])


def split_context(l, context):
    context = context.replace("\n", " ")
    context_sents = nltk.sent_tokenize(context)
    marked_context = [[]]
    lnew = 0
    for x in context_sents:
        lnew += len(x)
        if lnew > l:
            marked_context.append([])
            lnew = len(x)
        marked_context[-1].append(x)
    return marked_context