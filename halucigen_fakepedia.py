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
tokenizernltk = RegexpTokenizer(r'\w+')
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
ds = load_dataset("NLP-RISE/HalluciGen", "de_en_translation", token = args.token)
print(ds)
data = ds['test']
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

def get_subj_obj_in_last_sent(data, word):
    nlp = stanza.Pipeline('en', verbose=False)
    doc = nlp(data)
    sentence = doc.sentences[-1]
    subj_pos = []
    ws = [x for x in sentence.words if x.text == word]
    if len(ws) == 0:
        return None
    w = ws[0]
    if w.head == 0:
        return None
    w = sentence.words[w.head - 1]
    while w.head > 0:
        subj_pos.append((w.text, w.deprel))
        if w.head == 0:
            subj_pos += [(x.text, x.deprel) for x in sentence.words if x.head == w.id]
            break
        w = sentence.words[w.head - 1]
    return subj_pos
def clean(x):
    return x.replace("„", "").replace("”", "").replace('"', '').replace("‘", "").replace("’", "").replace("'", "").replace("—", "").replace("“", "")

def hasnum(x):
    return bool(re.search(r'\d', x))

for i, x in enumerate(data):
    assert x['label'] == 'hyp2' or x['label'] == 'hyp1'
    ha = clean(x['hyp1'] if x['label'] == 'hyp1' else x['hyp2'])
    ri = clean(x['hyp2'] if x['label'] == 'hyp1' else x['hyp1'])
    src = clean(x['source'])
    if False:
        sf = get_split if False else lambda x: x
        print(sf(x['source']))
        print(sf(ri))
        print(sf(ha))
        lcsc = get_token_lcs(ri, ha)
        lcsic = get_token_lcs(ha, ri)
        print(lcsc)
        print(lcsic)
    lcp = 0
    lcpe = 0
    lcpi = 0
    hatok = list(tokenizernltk.span_tokenize(ha))
    ritok = list(tokenizernltk.span_tokenize(ri))
    for i in range(min(len(hatok), len(ritok))):
        lcp = hatok[i][0]
        lcpe = hatok[i][1]
        lcpi = i
        if ha[lcp:lcpe] != ri[lcp:lcpe]:
            print(ha[:lcp])
            print(ha[lcp:lcpe], "-vs-", ri[lcp:lcpe])
            break
    def post(x):
        return pos_tag(word_tokenize(x), tagset='universal')[0][1]
    postagha = post(ha[lcp:lcpe])
    postagri = post(ri[lcp:lcpe])
    if lcp == min(len(ha), len(ri)) or lcp < 8 or ((postagha != "NOUN" or postagri != "NOUN") if True else postagha != postagri):
        continue
    print(lcp, lcpi, len(ha), len(ri), ha[lcp:lcpe], postagha, ri[lcp:lcpe], postagri)
    #subj_obj = get_subj_obj_in_last_sent(ha, ha[lcp:lcpe])
    subj_tail_len = 4
    cont_min_len = 0
    other_nouns_inds = [i for i in range(lcpi-cont_min_len) if post(ha[hatok[i][0]:hatok[i][1]]) == "NOUN"]
    if len(other_nouns_inds) < 2:
        continue
    last_noun = ha[hatok[max(0, other_nouns_inds[-1]-subj_tail_len)][0]:hatok[other_nouns_inds[-1]][1]]
    if len(last_noun) < 1:
        continue
    print("final", ha, last_noun)
    dp = {"fact_parent": {}}
    dp["query"] = ("TRANSLATION:" + ha + "\nCORRECTION:" + ha[:lcp]).strip()
    dp["subject"] = last_noun
    dp["object"] = " "+ri[lcp:]
    dp["fact_parent"]["object"] = " "+ha[lcp:]
    dp["fact_paragraph"] = src
    if hasnum(dp["object"]) or hasnum(dp["fact_parent"]["object"]):
        continue
    data_out.append(dp)
    with open("transl_fakepedia.json", "w") as f:
        json.dump(data_out, f, indent=4, ensure_ascii=False)
print(len(data_out))