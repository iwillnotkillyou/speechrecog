import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow_datasets.text.unifiedqa import UnifiedQA
import regex as re
from itertools import chain
import json
from shared import make_clienteinfra
import nltk
import torch
from causal_tracing_whisper1 import make_model
import os
import numpy as np
name = "arc_hard_with_ir_dev"
from causal_tracing_whisper1 import Namespace1
def main():
    optionre = re.compile(r"\([a-zA-Z]\)")
    LLmodel1 = make_clienteinfra(4, "")
    LLmodel2 = make_clienteinfra(-1, "")
    model, tokenizer = make_model(Namespace1())
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
        prob = np.prod([torch.softmax(output_dict.logits[0][x], 0)[labels[0][x]].item() for x in range(len(output_dict.logits[0]))])
        return prob

    def make_sentences(question, answers):
        s = '\n'.join(answers)
        resp = LLmodel1.run(f"QUESTION:\n{question}\nOPTIONS:\n{s}",
                  "Turn these options into a full sentences answering the question. Do this only by prepending new words. Return each on a separate line as plain text. Do not describe your output. Do not add any additional information not present in the options.")[0]
        print(f"QUESTION:\n{question}\nOPTIONS:\n{s}")
        print("RESPONSE:")
        print(resp)
        print("---")
        return [x.strip() for x in resp.split("\n")]

    def get_subjects(queries):
        s = "\n".join(queries)
        print(s)
        resp = LLmodel2.run(f"{s}",
                         "Return the shortest phrase which is the subjects for each of the input sentence fragments. Return each on a separate line")[0]
        print(resp)
        return [x.strip() for x in resp.split("\n")]

    unifiedqa_separator = "\\n"
    def make_fakepedia(context, answer, id, useLLM=True):
        question, answers, context_text = context.split(unifiedqa_separator)
        answers_list = [x.strip() for x in optionre.split(answers, maxsplit=50) if len(x) > 0]
        wrong_answers = [x for x in answers_list if x != answer and len(x) > 0]
        answers_list = [answer]+wrong_answers
        probs = [get_prob(context+"<pad>"+x, model, tokenizer, device) for x in answers_list]
        print(probs)
        answers_sents = make_sentences(question, [answer]+wrong_answers) if useLLM else [answer]+wrong_answers
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
            return {"query": query, "fact_paragraph": unifiedqa_separator.join([question, context_text]), "rel_lemma": sorted(zip(range(len(probs)), answers_list, probs), key= lambda x: x[2], reverse=True), "subject": query, "object": answer_sent[shared_prefix_len:], "fact_parent": {
                "object": wrong_answer_sent[shared_prefix_len:]}, "group_id": id}
        r = [make_one(x) for x in answers_sents[1:]]
        return r

    devds = [x for x in tfds.load("unified_qa",
                      builder_kwargs={"config": [x for x in UnifiedQA.BUILDER_CONFIGS if x.name == name][0]},
                      split='validation', shuffle_files=True)]
    print([x.name for x in UnifiedQA.BUILDER_CONFIGS if "multiple-choice" in x.description.lower()])
    l = 1500
    tol = 2000
    def make_some(natural_sents, lim, start_id = 0, reverse=False):
        devdataseti = [(x["input"].numpy(), x["output"].numpy()) for x in (reversed(devds) if reverse else devds) if
                       tf.strings.length(x["input"]).numpy() > l and tf.strings.length(
                           x["input"]).numpy() < l + tol][:lim]
        print(len(devdataseti))
        devdataseti = [x for x in devdataseti if natural_sents == ("." in x[0].decode("utf-8").split(unifiedqa_separator)[1])]
        print(len(devdataseti))
        fakepedia = list(
            chain.from_iterable(
                [make_fakepedia(x[0].decode("utf-8"), x[1].decode("utf-8"), start_id+i, not natural_sents) for i, x in enumerate(devdataseti)]))
        fakepedia = [x for x in fakepedia if len(nltk.word_tokenize(x["subject"])) > 2]
        print(len(fakepedia))
        subjects = get_subjects([x["subject"] for x in fakepedia])
        for x in zip(subjects, fakepedia):
            assert  x[0] in x[1]["query"], f"subject {x[0]} not in query {x[1]['query']}"
            x[1]["subject"] = x[0]
        return fakepedia
    fakepedia = make_some(True, 2000)
    fakepedia2 = make_some(False, 20, start_id = 100000)
    fakepedia3 = make_some(False, 20, start_id = 200000, reverse=True)
    with open(f"fakepedia_{name}.json", "w") as f:
        json.dump([x for x in fakepedia+fakepedia2+fakepedia3], f, indent=2)
    def gm(ds):
        return np.mean([int(x["rel_lemma"][0][0] == 0) for x in ds])
    print("only natural sents", gm(fakepedia), len(fakepedia))
    print("only fake sents train",gm(fakepedia2), len(fakepedia2))
    print("only fake sents test",gm(fakepedia3), len(fakepedia3))
if __name__ == "__main__":
    main()
    with open("fakepedia_arc_hard_with_ir_dev.json") as f:
        fakepedia = json.load(f)
    def gm(ds):
        return np.mean([float(x["rel_lemma"][-1][0] == 0) for x in ds])
    print("all", gm(fakepedia), len(fakepedia))
    with open("fakepedia_arc_hard_with_ir_dev.json", "w") as f:
        json.dump([x for x in fakepedia], f, indent=2)