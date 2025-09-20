from datasets import load_dataset
import torch
import os
import pylcs
import json
import re
#import stanza
#from stanza.models.common.doc import Word
#stanza.download('en')
import nltk
#nltk.download('punkt_tab')
#nltk.download('averaged_perceptron_tagger_eng')
#nltk.download('universal_tagset')
from nltk.tokenize import RegexpTokenizer
from nltk.tag import pos_tag
from itertools import combinations, chain
from nltk.tokenize import word_tokenize
import numpy as np
from nltk.collocations import BigramCollocationFinder
tokenizernltk = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
def words(x):
    return tokenizernltk.tokenize(x)
def num_words(x):
    return len(tokenizernltk.tokenize(x))
def post(x):
    return pos_tag([x], tagset='universal')[0][1]
class Namespace1:
    def __init__(self):
        self.token = os.environ.get("HUGGINGFACE_TOKEN")
        self.fakepedia_path = "qa_validation_fakepedia.json"
        ms = ["unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit", "unsloth/Llama-3.2-1B-unsloth-bnb-4bit",
              "meta-llama/Llama-3.2-1B", "google/gemma-3-1b-pt", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]
        self.model_name_path = ms[0]
        self.prompt_template = "{context}{query}"
        self.num_grounded = 40
        self.num_unfaithful = 40
        self.prepend_space = True
        self.bfloat16 = False
        self.resume_dir = "PrefixTuningQAValidation"
        self.subset_size = 100
        self.skip_creation = True
        self.isTTS = False

def make_model(args, mock=False):
    class mock_model:
        def __init__(self):
            from types import SimpleNamespace
            self.config = SimpleNamespace()
            self.config.pad_token_id = 0
            self.config.eos_token_id = 0
            self.config.vocab_size = 1000
            self.device_map = "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    quantize = torch.cuda.is_available() and args.bfloat16
    max_memory_mapping = {0: "8GB", "cpu": "0GB"} if torch.cuda.is_available() else None
    if quantize:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                             torch_dtype=torch.bfloat16)
        model = mock_model() if mock else AutoModelForCausalLM.from_pretrained(args.model_name_path, token=args.token,
                                                                            force_download=False,
                                                                            quantization_config=quantization_config,
                                                                            device_map="auto",
                                                                            max_memory = max_memory_mapping)
    else:
        quantization_config = None
        model = mock_model() if mock else AutoModelForCausalLM.from_pretrained(args.model_name_path, token=args.token,
                                                                            force_download=False,
                                                                            device_map="auto",
                                                                            max_memory = max_memory_mapping)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_path, token=args.token, force_download=False,
                                              add_bos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer

args = Namespace1()
ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled", token=args.token)
print(ds)
data = ds['train']
print(np.unique([x["Difficulty Level"] for x in data], return_counts=True))
print(np.unique([x["Category of Hallucination"] for x in data], return_counts=True))
print(data)
model, tokenizer = make_model(args, mock=True)
splits = 4
def get_split(x):
    toks = x.split(" ")
    return [" ".join(toks[(i * len(toks)) // splits:((i + 1) * len(toks)) // splits]) for i in range(splits)]
    toks = tokenizer.encode(x, add_special_tokens=False)
    return [tokenizer.decode(toks[(i*len(toks))//splits:((i+1)*len(toks))//splits]) for i in range(splits)]

def get_token_lcs(x, y):
    wsx = words(x)
    wsy = words(y)
    if False:
        s = set(wsy)
        return [z for z in wsx if z not in s], [i for i in range(len(wsx)) if wsx[i] not in s]
    al = sorted(set(wsx + wsy))
    toksx = [al.index(z) for z in wsx]
    toksy = [al.index(z) for z in wsy]
    match_list = pylcs.lcs_sequence_idx("".join([chr(z) for z in toksx]), "".join([chr(z) for z in toksy]))
    return " ".join([al[toksy[z]] if z >= 0 else "." for z in match_list]), [i for i in range(len(match_list)) if match_list[i] < 0]

print("len(data)", len(data))

data_out = []
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit", force_download=False,
                                              add_bos_token=True)
def clean(x):
    return x.replace("„", "").replace("”", "").replace('"', '').replace("‘", "").replace("’", "").replace("'", "").replace("—", "").replace("“", "")

def hasnum(x):
    return bool(re.search(r'\d', x))

bigram_measures = nltk.collocations.BigramAssocMeasures()
all_words = tokenizernltk.tokenize(" ".join([clean(x['Hallucinated Answer'] + " " + x['Ground Truth'] + " " + " ".join(x['Knowledge']) + " " + x['Question']) for x in data]))
def filterf(x):
    return any(not y.isalpha() for y in x) or post(x) not in ("ADJ", "NOUN", "PROPN")
all_noun_like = set(filter(lambda x: not(filterf(x)), all_words))
finder = BigramCollocationFinder.from_words(all_words)
finder.apply_word_filter(filterf)
banned_pairs = finder.nbest(bigram_measures.pmi, len(all_noun_like)//4)
banned_dict = {k: set() for _, k in banned_pairs}
for v, k in banned_pairs:
    banned_dict[k].add(v)
print(banned_pairs[:10], len(all_noun_like))
def make_dp(sel, seltok, hallucinated, question, knowledge, disallowed, fact_id):
    subj_tail_len = 2
    obj_tail_len = 0
    ignore_last = 0
    psubj = set(x.lower() for x in ["I", "He", "She", "They", "We", "It", "This", "That", "These", "Those"])
    def selector(word, tag, allowed_tags = ("ADJ", "NOUN", "PROPN", "X")):
        return (tag in allowed_tags or "".join([y for y in word.lower() if y.isalpha()]) in psubj) and len(word) > 2

    #pos_tag(tokenizernltk.tokenize(sel), tagset='universal')
    words = [sel[seltok[i][0]:seltok[i][1]] for i in range(len(seltok))]
    pos_tags_w_words = pos_tag(words, tagset='universal')
    pos_tags = [x[1] for x in pos_tags_w_words]
    doprint = False
    for x in pos_tags_w_words:
        if "strabismus" in x[0]:
            print(x)
            print([(words[i], pos_tags[i]) for i in range(len(seltok))])
    #print(pos_tags)
    nounlike_inds = [i for i in range(len(seltok)) if selector(words[i], pos_tags[i])]
    nounlike_inds = nounlike_inds[:len(nounlike_inds)-ignore_last]
    dps = []
    for nounlike_ind_ind in range(len(nounlike_inds)):
        word_ind = nounlike_inds[nounlike_ind_ind]
        if word_ind == 0 or words[word_ind] in disallowed or (words[word_ind] in banned_dict and words[word_ind-1] in banned_dict[words[word_ind]]):
            continue
        other_far_nounlike_inds = [i for i in nounlike_inds[:nounlike_ind_ind] if word_ind - i > obj_tail_len and pos_tags[i] in ("NOUN", "PROPN", "X")]
        if doprint:
            print([words[x] for x in nounlike_inds])
            print([words[x] for x in other_far_nounlike_inds])
        if len(other_far_nounlike_inds) < 1:
            continue
        other_far_nounlike_ind = other_far_nounlike_inds[-1]
        def get_subj_start(end, max_len):
            nounlike = [x for x in reversed(range(max(0, end - max_len), end)) if selector(words[x], pos_tags[x], ("NOUN", "PROPN", "ADJ", "X")) or words[x] == "the" or words[x] == "a" or words[x] == "an"]
            if len(nounlike) < 1:
                return end
            return nounlike[0]

        last_other_nounlike = sel[
                              seltok[get_subj_start(other_far_nounlike_ind, subj_tail_len)][0]:seltok[other_far_nounlike_ind][
                                  1]]
        if len(last_other_nounlike) < 1:
            continue
        # print("final", sel, last_other_nounlike)
        pos = seltok[word_ind][0]
        obj = sel[pos:]
        cont = " " + obj
        dp = {"fact_parent": {}}
        dp["group_id"] = str(fact_id)
        dp["query"] = ("QUESTION:" + question + "\nANSWER:" + sel[:pos]).strip()
        dp["subject"] = last_other_nounlike
        dp["object"] = None if hallucinated else cont
        dp["fact_parent"]["object"] = cont if hallucinated else None
        dp["fact_paragraph"] = knowledge
        dp["disallowed"] = disallowed
        def check(x):
            if x is None:
                return False
            return not len([sub for sub in tokenizer.tokenize(" "+x.lstrip())[0] if sub.isalpha()]) > 2 or hasnum(x)

        if check(dp["object"]) or check(dp["fact_parent"]["object"]):
            continue
        dps.append(dp)
    if len(dps) == 0 or len(dps) > 7:
        return None
    return dps
dp_grounded = []
dp_unfaithful = []
for i, x in enumerate(data):
    #if x["Difficulty Level"] != "hard":
    #    continue
    ha = x['Hallucinated Answer']
    ri = x['Ground Truth']
    knowledge = " ".join(x['Knowledge'])
    question = x['Question']
    hatok = list(tokenizernltk.span_tokenize(ha))
    ritok = list(tokenizernltk.span_tokenize(ri))
    #subj_obj = get_subj_obj_in_last_sent(ha, ha[lcp:lcpe])
    def get_words(x, xtok):
        return [x[xtok[i][0]:xtok[i][1]] for i in range(len(xtok))]
    perc_same = 0.5
    if len(set(get_words(ri, ritok)).intersection(set(get_words(ha, hatok)))) < min(len(set(hatok)), len(set(ritok)))*perc_same:
        continue
    dp = make_dp(ha, hatok, True, question, knowledge, get_words(ri, ritok), i)
    dp_unfaithful.append(dp)
    dp = make_dp(ri, ritok, False, question, knowledge, get_words(ha, hatok), i)
    dp_grounded.append(dp)


data_out = [(dp_grounded[x] if dp_grounded[x] is not None else dp_unfaithful[x]) if x%2 == 0 else (dp_unfaithful[x] if dp_unfaithful[x] is not None else dp_grounded[x]) for x in range(len(dp_grounded))]
data_out = [x for x in data_out if x is not None]
print(np.unique([len(x) for x in data_out], return_counts=True))
print(sum([len(x) for x in data_out]))
print(len([x for x in data_out if x[0]["fact_parent"]["object"] == None]))
print(len(data_out))
with open("med_fakepedia.json", "w") as f:
    json.dump(list(chain(*data_out)), f, indent=4, ensure_ascii=False)
print([sub for sub in tokenizer.tokenize(" "+" -severe hypoglycaemia is significantly associated with increased risk for cardiovascular events in high-risk diabetic populations.".lstrip())[0] if sub.isalpha()])