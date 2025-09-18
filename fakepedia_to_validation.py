import json
import os
import torch
from openai import OpenAI
from itertools import islice
class Namespace1:
    def __init__(self):
        self.token = os.environ.get("HUGGINGFACE_TOKEN", None)
        self.fakepedia_path = "validation_fakepedia.json"
        ms = ["unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit", "meta-llama/Llama-3.2-1B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]
        self.model_name_path = ms[0]
        self.bfloat16 = True

def make_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    quantize = torch.cuda.is_available() and args.bfloat16
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                             torch_dtype=torch.bfloat16) if quantize else None
    model = AutoModelForCausalLM.from_pretrained(args.model_name_path, token=args.token,
                                                                           force_download=False,
                                                                           quantization_config=quantization_config,
                                                                           device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_path, token=args.token, force_download=False,
                                              add_bos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer

class QA:
    def __call__(self, i, x):
        if i % 2 == 0:
            x[
                "query"] = f"Question: {x['query']} Answer: {x['object']}. Ratings: obviously false from context, obviously true from context, obviously"
            x["subject"] = x["query"][:x["query"].find(" Answer")]
            x["object"] = "true"
            x["fact_parent"]["object"] = "false"
        else:
            x[
                "query"] = f"Question: {x['query']} Answer: {x['fact_parent']['object']}. Ratings: obviously false from context, obviously true from context, obviously"
            x["subject"] = x["query"][:x["query"].find(" Answer")]
            x["object"] = "false"
            x["fact_parent"]["object"] = "true"

class Simple:
    def __call__(self, i, x):
        if i % 2 == 0:
            x[
                "query"] = f"Statement: {x['query']} {x['object']}. Ratings: obviously false from context, obviously true from context, obviously"
            x["subject"] = x["query"][:x["query"].find("Ratings")]
            x["object"] = "true"
            x["fact_parent"]["object"] = "false"
        else:
            x[
                "query"] = f"Statement: {x['query']} {x['fact_parent']['object']}. Ratings: obviously false from context, obviously true from context, obviously"
            x["subject"] = x["query"][:x["query"].find("Ratings")]
            x["object"] = "false"
            x["fact_parent"]["object"] = "true"

class UsingOpenAI:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    def __call__(self, i, x):
        statement = f"{x['query']} {x['object'] if i % 2 == 0 else x['fact_parent']['object']}"
        response = self.client.chat.completions.create(
            model="gemini-2.5-flash-preview-05-20",
            reasoning_effort="low",
            messages=[
                {"role": "system",
                 "content": "Provide the answer the user requests and step by step reasoning. Only use the information in the context during your reasoning. Limit your reasoning to 3 steps. Label your reasoning steps with Reasoning Step 1, Reasoning Step 2, Reasoning Step 3. Do not use any other labels. Do not use any other text in the output."},
                {"role": "user", "content": f"Please determine whether the following Statement is \"true\" or \"false\" given Context and give  step by step reasoning for your choice, please end the answer with \"true\" or \"false\". \nContext: {x['fact_paragraph']}\n.Statement: {statement}"},
            ],
            top_p=0.2,
        )
        output_text = response.choices[0].message.content.strip()
        print(output_text)
        x["query"] = f"{statement}\n{output_text}\nRatings: obviously false from reasoning, obviously true from reasoning, obviously"
        x["object"] = "false" if i % 2 == 0 else "true"
        x["fact_parent"]["object"] = "true" if i % 2 == 0 else "false"

def main():
    def read_json(json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    fakepedia = read_json("qa_fakepedia.json")
    m = QA()
    skip = 1 if True else 2
    fakepedia = list(islice(fakepedia, 0, 200, skip))
    for i, x in enumerate(fakepedia):
        m(i, x)
    json.dump(fakepedia, open("qa_validation_fakepedia.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()