import json
import logging
import os
from itertools import chain

import nltk
import numpy as np
import regex as re
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from tensorflow_datasets.text.unifiedqa import UnifiedQA

from causal_tracing_whisper1 import make_model
from shared import make_clienteinfra

name = "qasc_with_ir"
split = "train"
minl = 2200
from causal_tracing_whisper1 import Namespace1


def main():
    optionre = re.compile(r"\([a-zA-Z]\)")
    LLmodel1 = make_clienteinfra(4, "")
    LLmodel2 = make_clienteinfra(-1, "")
    n = Namespace1()
    n.model_name_path = "allenai/unifiedqa-t5-base"
    model, tokenizer = make_model(n)
    device = "cuda"
    model.to(device)
    cached_encoder_input = (None, None)

    def get_prob(string, model, tokenizer, device):
        encoder_input_text, decoder_input_text = string.split("<pad>")
        encoder_ids = tokenizer.encode(encoder_input_text)
        encoder_output = model.encoder(
            input_ids=torch.tensor([encoder_ids], device=device)
        )
        labels = torch.tensor([tokenizer.encode(decoder_input_text, add_special_tokens=False)], device=device)
        output_dict = model.forward(encoder_outputs=(encoder_output,),
                                    labels=labels,
                                    return_dict=True)
        prob = np.prod([torch.softmax(output_dict.logits[0][x], 0)[labels[0][x]].item() for x in
                        range(len(output_dict.logits[0]))])
        return prob

    def make_sentences(question, answers):
        s = '\n'.join(answers)
        resp = LLmodel1.run(f"QUESTION:\n{question}\nOPTIONS:\n{s}",
                            "Turn these options into a full sentences answering the question. Do this only by prepending new words. Return each on a separate line as plain text. Do not describe your output. Do not add any additional information not present in the options.")[
            0]
        print(f"QUESTION:\n{question}\nOPTIONS:\n{s}")
        print("RESPONSE:")
        print(resp)
        print("---")
        return [x.strip() for x in resp.split("\n")]

    def get_subjects(queries):
        s = "\n".join(queries)
        print(s)
        resp = LLmodel2.run(f"{s}",
                            "Return the shortest phrase which is the subjects for each of the input sentence fragments. Return each on a separate line")[
            0]
        print(resp)
        return [x.strip() for x in resp.split("\n")]

    unifiedqa_separator = "\\n"

    def make_fakepedia(context, answer, id, useLLM=True):
        question, answers, context_text = context.split(unifiedqa_separator)
        answers_list = [x.strip() for x in optionre.split(answers, maxsplit=50) if len(x) > 0]
        wrong_answers = [x for x in answers_list if x != answer and len(x) > 0]
        answers_list = [answer] + wrong_answers
        probs = [get_prob(context + "<pad>" + x, model, tokenizer, device) if False else 0 for x in answers_list]
        print(probs)
        answers_sents = make_sentences(question, [answer] + wrong_answers) if useLLM else [answer] + wrong_answers
        answer_sent = answers_sents[0]

        def make_one(wrong_answer_sent):
            shared_prefix_len = 0
            answer_words = answer_sent.split(" ")
            x_words = wrong_answer_sent.split(" ")
            for i in range(min(len(answer_words), len(x_words))):
                if answer_words[i] == x_words[i]:
                    shared_prefix_len += len(answer_words[i]) + 1
                else:
                    break
            query = answer_sent[:shared_prefix_len]
            return {"query": query, "fact_paragraph": unifiedqa_separator.join([question, context_text]),
                    "rel_lemma": sorted(zip(range(len(probs)), answers_list, probs), key=lambda x: x[2], reverse=True),
                    "subject": query, "object": answer_sent[shared_prefix_len:], "fact_parent": {
                    "object": wrong_answer_sent[shared_prefix_len:]}, "group_id": id}

        r = [make_one(x) for x in answers_sents[1:]]
        return r
    conf = [x for x in UnifiedQA.BUILDER_CONFIGS if x.name == name][0]
    devds = [x for x in tfds.load("unified_qa",
                                  builder_kwargs={
                                      "config": conf},
                                  split=split, shuffle_files=True) if
                           tf.strings.length(x["input"]).numpy() > minl and tf.strings.length(
                               x["input"]).numpy() < minl + 2000]
    print(conf.name, "len(devds)", len(devds))
    print([x.name for x in UnifiedQA.BUILDER_CONFIGS if "multiple-choice" in x.description.lower()])
    if os.path.exists(f"cache_{name}.json"):
        with open(f"cache_{name}.json") as f:
            cache = json.load(f)
    else:
        cache = {}

    def make_some(natural_sents, lim, start_id=0, reverse=False):
        k = f"{natural_sents}_{lim}_{start_id}_{reverse}"
        if k in cache:
            fakepedia = cache[k]
        else:
            devdataseti = [(x["input"].numpy(), x["output"].numpy()) for x in (reversed(devds) if reverse else devds)][:lim]
            print(len(devdataseti))
            devdataseti = [x for x in devdataseti if
                           natural_sents == ("." in x[0].decode("utf-8").split(unifiedqa_separator)[1])]
            print(len(devdataseti))
            fakepedia = list(
                chain.from_iterable(
                    [make_fakepedia(x[0].decode("utf-8"), x[1].decode("utf-8"), start_id + i, not natural_sents) for i, x in
                     enumerate(devdataseti)]))
            fakepedia = [x for x in fakepedia if len(nltk.word_tokenize(x["subject"])) > 2]
            cache[k] = fakepedia
            with open("cache.json", "w") as f:
                json.dump(cache, f)
        print(len(fakepedia))
        subjects_plus_fakepedia = []
        parts = 10
        for x in range(parts):
            values = [x for x in fakepedia[(len(fakepedia) * x) // parts: (len(fakepedia) * (x + 1)) // parts]]
            subjects_i = get_subjects([x["subject"] for x in values])
            if len(subjects_i) != len(values):
                logging.warning("len(subjects_i) != len(values)")
            else:
                subjects_plus_fakepedia += list(zip(subjects_i, values))
        for x in subjects_plus_fakepedia:
            s = x[0].lower().strip(" .")
            x[1]['query'] = x[1]['query'].lower().strip(" .")
            if not s in x[1]["query"]:
                logging.warning(f"subject {s} not in query {x[1]['query']}")
            else:
                x[1]["subject"] = s
        return [x[1] for x in subjects_plus_fakepedia]

    fakepedia = make_some(True, 20000)
    fakepedia2 = make_some(False, 100, start_id=100000)
    fakepedia3 = make_some(False, 100, start_id=200000, reverse=True)
    with open(f"fakepedia_{name}.json", "w") as f:
        json.dump([x for x in fakepedia + fakepedia2 + fakepedia3], f, indent=2)

    def gm(ds):
        return np.mean([int(x["rel_lemma"][0][0] == 0) for x in ds])

    print("only natural sents", gm(fakepedia), len(fakepedia))
    print("only fake sents train", gm(fakepedia2), len(fakepedia2))
    print("only fake sents test", gm(fakepedia3), len(fakepedia3))

    with open(f"fakepedia_{name}.json") as f:
        fakepedia = json.load(f)

    def gm(ds):
        return np.mean([float(x["rel_lemma"][-1][0] == 0) for x in ds])

    print("all", gm(fakepedia), len(fakepedia))

def get_lens():
    for i, x in enumerate(UnifiedQA.BUILDER_CONFIGS[:int(len(UnifiedQA.BUILDER_CONFIGS) * 1)]):
        if "dev" in x.name or "test" in x.name:
            continue
        if "multiple-choice" not in x.description.lower():
            continue
        devds = [x for x in tfds.load("unified_qa",
                                      builder_kwargs={
                                          "config": [y for y in UnifiedQA.BUILDER_CONFIGS if y.name == x.name][0]},
                                      split='train', shuffle_files=True) if tf.strings.length(
                                   x["input"]).numpy() > minl]
        if "qasc_with_ir" == x.name:
            with open("qasc_with_ir.json", "w") as f:
                json.dump([{k: v.numpy().decode("utf-8") for k,v in x.items()} for x in devds], f, indent=2)
        print(i, x.name, len(devds))

if __name__ == "__main__":
    #get_lens()
    main()

