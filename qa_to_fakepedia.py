import json

import numpy as np


def get_subjects(fn):
    import stanza
    stanza.download('en')

    nlp = stanza.Pipeline('en')

    with open(f'{fn}.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    data_out = []
    for j in range(len(data)):
        # Parse a sentence
        doc = nlp(data[j]["question"])

        # Print out the dependency parse
        for sentence in doc.sentences:
            print("-------------")
            unset_pos = set(range(len(sentence.words)))
            subj_pos = set()
            for _ in range(len(sentence.words)):
                for i in unset_pos:
                    word = sentence.words[i]
                    if (word.deprel == "nsubj" and sentence.words[word.head-1].head == 0 or word.head-1 in subj_pos) and i not in subj_pos:
                        subj_pos.add(i)
            print([sentence.words[x].text for x in sorted(subj_pos)], sentence.text)
            data_out.append(dict(data[j], subj_pos = list(sorted(subj_pos))))
    json.dump(data_out, open(f"{fn}.json", "w"), indent=4)


from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
def qa_to_fakepedia(fn):
    with open(f"{fn}.jsonl", 'r', encoding='utf-8') as f:
        data = [x for x in [json.loads(line) for line in f] if len(x) > 0]
    data_out = []
    for x in data:
        dp = {"fact_parent": {}}
        lcp = 0
        ha = x["hallucinated_answer"]
        ri = x["right_answer"]
        hatok = list(tokenizer.span_tokenize(ha))
        ritok = list(tokenizer.span_tokenize(ri))
        for i in range(min(len(hatok), len(ritok))):
            if ha[:hatok[i][1]] != ritok[:ritok[i][1]]:
                break
            lcp = hatok[i][1]
        if lcp == min(len(ha), len(ri)):
            continue
        dp["query"] = x["question"] + (" " + x["hallucinated_answer"][:lcp] if lcp > 0 else "")
        dp["subject"] = x["question"]
        dp["object"] = x["right_answer"][lcp:]
        dp["fact_parent"]["object"] = x["hallucinated_answer"][lcp:]
        dp["fact_paragraph"] = x["knowledge"]
        data_out.append(dp)
    json.dump(data_out, open("qa_fakepedia.json", "w"), indent=4)

def read_json(json_path: str):
    if json_path.endswith(".jsonl"):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    return data

if __name__ == "__main__":
    qa_to_fakepedia("qa")

    fakepedia = read_json("base_fakepedia.json")
    data_out = []
    for x in fakepedia:
        dp = dict(x, fact_parent = dict(x["fact_parent"]))
        dp["subject"] = dp["query"]
        data_out.append(dp)
    json.dump(data_out, open("question_subject_fakepedia.json", "w"), indent=4)