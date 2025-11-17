"""Dataset converters for different HuggingFace dataset formats to probing format."""
import logging
from typing import Callable, Dict, List, Optional
from datasets import Dataset, load_dataset
from dataclasses import dataclass
from typing import List
import os
import numpy as np
import torch
import nltk
from shared import make_clienteinfra
import json
import regex
import dotenv
dotenv.load_dotenv()
class Namespace1:
    def __init__(self):
        self.token = os.environ.get("HUGGINGFACE_TOKEN")
        self.fakepedia_path = "fakepedia_arc_hard_with_ir_dev.json"
        ms = ["allenai/unifiedqa-t5-large", "google/gemma-3-1b-pt", "google/t5gemma-l-l-prefixlm-it", "google/t5gemma-2b-2b-prefixlm-it", "google/t5gemma-b-b-prefixlm", "google/gemma-3-1b-it",
              "unsloth/Llama-3.2-1B-unsloth-bnb-4bit",
              "unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit",
              "meta-llama/Llama-3.2-1B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "unsloth/llama-3-8b-Instruct-bnb-4bit"]
        self.model_name_path = ms[2]
        self.bfloat16 = True

def make_model(args, mock=False):
    class mock_model:
        def __init__(self):
            from types import SimpleNamespace
            self.config = SimpleNamespace()
            self.config.pad_token_id = 0
            self.config.eos_token_id = 0
            self.config.vocab_size = 1000
            self.device_map = "cpu"

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
    quantize = torch.cuda.is_available() and args.bfloat16
    max_memory_mapping = {0: "8GB", "cpu": "0GB"} if torch.cuda.is_available() else None
    if quantize:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                             torch_dtype=torch.bfloat16, llm_int8_enable_fp32_cpu_offload=True)
        model = mock_model() if mock else AutoModelForSeq2SeqLM.from_pretrained(args.model_name_path, token=args.token,
                                                                            force_download=False,
                                                                            quantization_config=quantization_config,
                                                                            device_map="auto",
                                                                            max_memory = max_memory_mapping)
    else:
        quantization_config = None
        model = mock_model() if mock else AutoModelForSeq2SeqLM.from_pretrained(args.model_name_path, token=args.token,
                                                                            force_download=False,
                                                                            device_map="auto",
                                                                            max_memory = max_memory_mapping)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_path, token=args.token, force_download=False,
                                              add_bos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    print(model)
    return model, tokenizer

@dataclass
class AnnotatedSpan:
    """A text span with its hallucination label."""

    span: str  # The span text
    label: float  # 1.0 for hallucination, 0.0 for supported, -100.0 for ignored
    index: int  # Start index in the completion


@dataclass
class ProbingItem:
    """A single item containing prompt, completion and annotated spans."""

    prompt: str
    completion: str
    spans: List[AnnotatedSpan]

# Mapping from text labels to numeric values for probe training
_MAP_LABEL_TO_SCALAR = {
    'Not Supported': 1.0,
    'NS': 1.0,  # the probe should output 1.0 on text containing unsupported claims
    'Insufficient Information': 1.0,  # the probe should also output 1.0 if the label is 'Insufficient Information'
    'Supported': 0.0,
    'S': 0.0,
    'N/A': -100.0,
    None: -100.0
}


def prepare_longform_dataset(dataset: Dataset) -> List[ProbingItem]:
    """Prepare dataset from the one-shot pipeline labeling format."""
    probing_items: List[ProbingItem] = []

    for hf_item in dataset:
        prompt = hf_item['conversation'][-2]['content']
        completion = hf_item['conversation'][-1]['content']
        annotations: List[dict] = hf_item['annotations']

        annotated_spans: List[AnnotatedSpan] = []

        for entity in annotations:
            if entity is None or 'index' not in entity or not isinstance(entity['index'], int):
                continue

            entity_text = entity['span']
            label = entity['label']
            idx = entity['index']

            if idx is None:
                print(f"Entity {repr(entity_text)}'s idx set to None, discarding entity")
                continue
            elif not entity_text or entity_text not in completion:
                print(f"Entity {repr(entity_text)} not found in completion, discarding entity")
                continue

            annotated_spans.append(
                AnnotatedSpan(
                    span=entity_text,
                    label=_MAP_LABEL_TO_SCALAR[label],
                    index=idx
                )
            )

        probing_items.append(
            ProbingItem(
                prompt=prompt,
                completion=completion,
                spans=annotated_spans
            )
        )

    return probing_items


def prepare_longform_dataset_old_format(dataset: Dataset) -> List[ProbingItem]:
    """Prepare dataset from the one-shot pipeline labeling format."""
    probing_items: List[ProbingItem] = []

    for hf_item in dataset:
        prompt = hf_item['conversation'][0]['content']
        completion = hf_item['completion'] if 'completion' in hf_item else hf_item['conversation'][-1]['content']
        annotations: List[dict] = hf_item['verified_entities']

        annotated_spans: List[AnnotatedSpan] = []

        for entity in annotations:
            if entity is None or 'idx' not in entity or not isinstance(entity['idx'], int):
                continue

            entity_text = entity['text']
            label = entity['label']
            idx = entity['idx']

            if idx is None:
                print(f"Entity {repr(entity_text)}'s idx set to None, discarding entity")
                continue
            elif not entity_text or entity_text not in completion:
                print(f"Entity {repr(entity_text)} not found in completion, discarding entity")
                continue

            annotated_spans.append(
                AnnotatedSpan(
                    span=entity_text,
                    label=_MAP_LABEL_TO_SCALAR[label],
                    index=idx
                )
            )

        probing_items.append(
            ProbingItem(
                prompt=prompt,
                completion=completion,
                spans=annotated_spans
            )
        )

    return probing_items


def prepare_triviaqa(dataset: Dataset) -> List[ProbingItem]:
    """
    Pre-processes TriviaQA dataset.
    The greedy completion (labeled by an LLM) is at `gt_completion`
    The label is at `llm_judge_label` and it's a string containing `S`, `NS`, `N/A` or some undefined string
    The annotated spans will be the *whole completion*
    """
    assert 'question' in dataset[0] or 'conversation' in dataset[0]

    LABEL_FIELD: str = 'llm_judge_label' if 'llm_judge_label' in dataset.features else 'label'
    COMPLETION_FIELD: str = 'gt_completion'
    VALID_LABELS: List[str] = ['S', 'NS', 'N/A']
    EXACT_ANSWER_FIELD: str = 'exact_answer'

    probing_items = []
    for item in dataset:
        if item[LABEL_FIELD] not in VALID_LABELS:
            print(f"Invalid label {item[LABEL_FIELD]} for item, skipping")
            continue

        prompt = item['question'] if 'question' in item else item['conversation'][0]['content']
        completion = item[COMPLETION_FIELD]
        exact_answer = item[EXACT_ANSWER_FIELD] if EXACT_ANSWER_FIELD in item else ""

        if exact_answer is None or exact_answer not in completion:
            print(f"Exact answer {repr(exact_answer)} not found in completion {repr(completion)}")
            return None

        exact_answer_start_idx = completion.find(exact_answer)

        # The whole completion is labeled with the given label
        label_value: float = _MAP_LABEL_TO_SCALAR[item[LABEL_FIELD]]

        annotated_spans = [
            AnnotatedSpan(
                span=exact_answer,
                label=label_value,
                index=exact_answer_start_idx
            )
        ]

        probing_items.append(
            ProbingItem(
                prompt=prompt,
                completion=completion,
                spans=annotated_spans
            )
        )

    return probing_items


def prepare_synthetic(dataset: Dataset) -> List[ProbingItem]:
    """Loads the synthetic dataset from the hub."""
    FIELD = 'probing_item_with_hallucinations'

    probing_items = []
    for i, item in enumerate(dataset):
        probing_item = item[FIELD]
        annotated_spans = [
            AnnotatedSpan(
                span=span['text'],
                label=span['label'],
                index=span['start_idx']
            )
            for span in probing_item['spans']
        ]

        # Sort spans by their index in the text
        completion = probing_item['completion']

        if len(completion) <= 500:
            print(f"For item {i} completion is too short ({len(completion)} characters): {repr(completion)}")
            continue

        annotated_spans = sorted(annotated_spans, key=lambda x: x.index)

        if not all(completion[span.index:span.index + len(span.span)] == span.span for span in annotated_spans):
            print(f"For item {i} spans are not aligned with the completion")
            for span in annotated_spans:
                if completion[span.index:span.index + len(span.span)] != span.span:
                    print(f"- Span: {span.span} | {span.index} | {span.label}")
            continue

        probing_items.append(ProbingItem(
            prompt=probing_item['prompt'],
            completion=probing_item['completion'],
            spans=annotated_spans
        ))
    return probing_items


def get_prepare_function(
        hf_repo: str,
        subset: Optional[str] = None
) -> Callable[[Dataset], List[ProbingItem]]:
    """Get the appropriate preparation function based on dataset name."""
    if 'one_shot_pipeline' in str(subset) or 'hallucination-heads' in hf_repo:
        return prepare_longform_dataset_old_format
    elif 'modified' in str(subset) and 'synthetic-hallucinations' in hf_repo:
        return prepare_synthetic
    elif 'trivia_qa' in str(subset) or 'triviaqa' in hf_repo:
        return prepare_triviaqa
    else:
        # Default to one-shot pipeline format
        return prepare_longform_dataset
ds_all = []
for x in ['Llama-3.3-70B-Instruct'] + (['Meta-Llama-3.1-8B-Instruct', 'Mistral-Small-24B-Instruct-2501', 'Qwen2.5-7B-Instruct', 'gemma-2-9b-it'] if True else []):
    dsi = load_dataset("obalcells/longfact-augmented-annotations", x, split="train", token=os.environ.get("HF_TOKEN"))
    dsi = prepare_longform_dataset(dsi)
    ds_all.extend(dsi)
ds = ds_all
print(len(ds))
ds0 = [x for x in ds if any(span.label == 1.0 for span in x.spans)]
print(len(ds0))
ds1 = [x for x in ds if any(span.label == 1.0 and not any([x.isdigit() for x in span.span]) for span in x.spans)]
print(len(ds1))
ds = [x for x in ds if any(span.label == 1.0 and span.span.isalpha() and not "\n" in span.span for span in x.spans)]
print(len(ds))
lens = [len(x.prompt) for x in ds]
quants = np.quantile(lens, [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
print(quants)
ds = [x for x in ds if len(x.prompt) > quants[2]]
print(len(ds))
fakepedia = []

LLmodel2 = make_clienteinfra(-1, "")

def get_subjects(queries):
    s = "\n".join(queries)
    print(s)
    resp = LLmodel2.run(f"{s}",
                        "Return the shortest phrase which is the corresponding subject of the last phrase for each of the input sentence fragments or None if the last phrase is not an object. Return each on a separate line")[0]
    print(resp)
    return [x.strip() for x in resp.split("\n")]

do_LLM = False
if do_LLM:
    model, tokenizer = make_model(Namespace1())
sentence_regex = regex.compile(r'[\.!?]+ *$', regex.M)
for i, x in enumerate(ds):
    for span in x.spans:
        completion = x.completion[:span.index]
        assert x.completion[span.index:span.index+len(span.span)].endswith(span.span), f"Span {repr(span.span)} not found at index {span.index} in completion {repr(x.completion)}"
        if len(completion) == 0 or sentence_regex.match(completion) or not span.span.isalpha() or "\n" in span.span:
            continue
        stripped = completion.replace("**:", "** ").split("** ")[-1].split("\n")[-1]
        sents = list(nltk.tokenize.PunktSentenceTokenizer().span_tokenize(stripped))
        if len(sents) == 0:
            continue
        presubject = stripped[sents[-1][0]:].lstrip("* (\t")
        if any(x.isdigit() for x in presubject) or len(presubject) < 20 or any(x == "\n" for x in presubject):
            continue
        prompt = f"This is the prompt {x.prompt}. What is the most important word in this sentence for this answer {span.span}."
        dp = {"query": completion, "fact_paragraph": x.prompt,
                "presubject": presubject, "object": span.span if span.label == 0 else None,
                "unfaithful_object": span.span if span.label == 1 else None, "group_id": i}
        if do_LLM:
            r = model.generate(tokenizer.apply_chat_template([{"role": "user", "context": prompt}], return_tensors='pt').to("cuda"), max_new_tokens=50)
            print(tokenizer.decode(r[0]))
        fakepedia.append(dp)
print(len(fakepedia))
per = 50
fakepedia = fakepedia[:500]
for x in range(0, len(fakepedia), per):
    values = [x for x in fakepedia[x:x+per]]
    subjects_i = get_subjects([x["presubject"]+(x["object"] or x["unfaithful_object"]) for x in values])
    if len(subjects_i) != len(values):
        logging.warning("len(subjects_i) != len(values)")
    else:
        for x in range(len(subjects_i)):
            values[x]["subject"] = subjects_i[x] if subjects_i[x] != "None" else None

print(len(fakepedia))
print("objectd", np.unique([x["object"] is None for x in fakepedia], return_counts=True))
print("subjectd", np.unique([x["subject"] is None for x in fakepedia], return_counts=True))
print("\n".join([str((x["query"], x["object"], x["unfaithful_object"])) for x in fakepedia[:10]]))
with open("longfact_fakepedia.json", "w") as f:
    json.dump(fakepedia, f, indent=2)