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
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('universal_tagset')
from nltk.tokenize import RegexpTokenizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import numpy as np
tokenizernltk = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
def words(x):
    return tokenizernltk.tokenize(x)
def num_words(x):
    return len(tokenizernltk.tokenize(x))
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

def clean(x):
    return x.replace("„", "").replace("”", "").replace('"', '').replace("‘", "").replace("’", "").replace("'", "").replace("—", "").replace("“", "")

def hasnum(x):
    return bool(re.search(r'\d', x))

for i, x in enumerate(data):
    #if x["Difficulty Level"] != "hard":
    #    continue
    ha = x['Hallucinated Answer']
    ri = x['Ground Truth']
    knowledge = " ".join(x['Knowledge'])
    question = x['Question']
    def post(x):
        return pos_tag([x], tagset='universal')[0][1]
    hatok = list(tokenizernltk.span_tokenize(ha))
    ritok = list(tokenizernltk.span_tokenize(ri))
    #subj_obj = get_subj_obj_in_last_sent(ha, ha[lcp:lcpe])
    def make_dp(sel, seltok, hallucinated, disallowed):
        subj_tail_len = 4
        obj_tail_len = 1
        ignore_last = 2
        psubj = set(x.lower() for x in ["I", "He", "She", "They", "We", "It", "This", "That", "These", "Those"])
        def selector(x):
            return (post(x) == "NOUN" or "".join([y for y in x.lower() if y.isalpha()]) in psubj) and x not in disallowed

        nounlike_inds = [i for i in range(len(seltok)) if selector(sel[seltok[i][0]:seltok[i][1]])][:-ignore_last]
        other_far_nounlike_inds = [i for i in nounlike_inds[:-1] if nounlike_inds[-1] - i > obj_tail_len]
        if len(other_far_nounlike_inds) < 1:
            return None
        other_far_nounlike_ind = other_far_nounlike_inds[-1]
        last_other_nounlike = sel[seltok[max(0, other_far_nounlike_ind - subj_tail_len)][0]:seltok[other_far_nounlike_ind][1]]
        if len(last_other_nounlike) < 1:
            return None
        print("final", sel, last_other_nounlike)
        pos = seltok[nounlike_inds[-1]][0]
        obj = sel[pos:]
        cont = " " + obj
        dp = {"fact_parent": {}}
        dp["query"] = ("QUESTION:" + question + "\nANSWER:" + sel + "\nCORRECTION:" + sel[:pos]).strip()
        dp["subject"] = last_other_nounlike
        dp["object"] = None if hallucinated else cont
        dp["fact_parent"]["object"] = cont if hallucinated else None
        dp["fact_paragraph"] = knowledge
        dp["disallowed"] = disallowed
        return dp
    def get_words(x, xtok):
        return [x[xtok[i][0]:xtok[i][1]] for i in range(len(xtok))]
    perc_same = 0.6
    if len(set(get_words(ri, ritok)).intersection(set(get_words(ha, hatok)))) < min(len(set(hatok)), len(set(ritok)))*perc_same:
        continue
    if i % 2 == 0:
        dp = make_dp(ha, hatok, True, get_words(ri, ritok))
    else:
        dp = make_dp(ri, ritok, False, get_words(ha, hatok))
    if dp is None or (dp["object"] is not None and hasnum(dp["object"])) or (dp["fact_parent"]["object"] is not None and hasnum(dp["fact_parent"]["object"])):
        continue
    data_out.append(dp)
    if len(data_out)%10 == 0:
        with open("med_fakepedia.json", "w") as f:
            json.dump(data_out, f, indent=4, ensure_ascii=False)
print(len(data_out))