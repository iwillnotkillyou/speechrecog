import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow_datasets.text.unifiedqa import UnifiedQA
import regex as re
from itertools import chain
import json
from shared import make_clienteinfra
import nltk
optionre = re.compile(r"\([a-zA-Z]\)")
model = make_clienteinfra(-1, "")

def make_sentences(question, answers):
    s = '\n'.join(answers)
    resp = model.run(f"QUESTION:\n{question}\nOPTIONS:\n{s}",
              "Turn these options into a full sentences answering the question. Do this only by prepending new words. Return each on a separate line. Do not add any additional information not present in the options.")[0]
    print(f"QUESTION:\n{question}\nOPTIONS:\n{s}")
    print("RESPONSE:")
    print(resp)
    print("---")
    return resp.split("\n")

def get_subjects(queries):
    s = "\n".join(queries)
    print(s)
    resp = model.run(f"{s}",
                     "Return the shortest phrase which is the subjects for each of the input sentence fragments. Return each on a separate line")[0]
    print(resp)
    return resp.split("\n")

unifiedqa_separator = "\\n"
def make_fakepedia(context, answer, id, useLLM=True):
    question, answers, context_text = context.split(unifiedqa_separator)
    answers_list = [x.strip() for x in optionre.split(answers, maxsplit=50) if len(x) > 0]
    wrong_answers = [x for x in answers_list if x != answer and len(x) > 0]
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
        return {"query": answer_sent[:shared_prefix_len], "fact_paragraph": unifiedqa_separator.join([question, context_text]), "subject": answer_sent[:shared_prefix_len], "object": answer_sent[shared_prefix_len:], "fact_parent": {
            "object": wrong_answer_sent[shared_prefix_len:]}, "group_id": id}
    return [make_one(x) for x in answers_sents[1:]]

name = "arc_hard_with_ir_dev"
print([x.name for x in UnifiedQA.BUILDER_CONFIGS if "multiple-choice" in x.description.lower()])
devds = tfds.load("unified_qa", builder_kwargs={"config": [x for x in UnifiedQA.BUILDER_CONFIGS if x.name==name][0]}, split='validation', shuffle_files=True)
l = 1500
tol = 2000
def make_some(natural_sents, lim, start_id = 0):
    devdataseti = [(x["input"].numpy(), x["output"].numpy()) for x in devds if
                   tf.strings.length(x["input"]).numpy() > l and tf.strings.length(
                       x["input"]).numpy() < l + tol]
    print(len(devdataseti))
    devdataseti = [x for x in devdataseti if natural_sents == ("." in x[0].decode("utf-8").split(unifiedqa_separator)[1])]
    print(len(devdataseti))
    fakepedia = list(
        chain.from_iterable(
            [make_fakepedia(x[0].decode("utf-8"), x[1].decode("utf-8"), start_id+i, not natural_sents) for i, x in enumerate(devdataseti[:lim])]))
    fakepedia = [x for x in fakepedia if len(nltk.word_tokenize(x["subject"])) > 2]
    print(len(fakepedia))
    subjects = get_subjects([x["subject"] for x in fakepedia])
    for x in zip(subjects, fakepedia):
        x[1]["subject"] = x[0]
    return fakepedia
fakepedia = make_some(True, 2000)
fakepedia += make_some(False, 50, 100000)
print(len(fakepedia))
json.dump([x for x in fakepedia], open(f"fakepedia_{name}.json", "w"), indent=2)