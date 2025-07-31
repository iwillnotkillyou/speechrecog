# %% md
# Data
# %%
# !wget -nc -O data.zip "https://www.dropbox.com/scl/fo/6rets9zuu0nvbbokxvf0e/AIa4cSpy_sjdd7Pz9aTxXa0?rlkey=vqj56pc1sp6qlggldrz5vdw32&st=6mf25abu&dl=0"
# !unzip -o data.zip
# !pip install bitsandbytes
# %%
import copy
from dataclasses import dataclass
from typing import Dict


def fact_from_dict(fact_dict: Dict):
    if fact_dict["fact_parent"] is None:
        return Fact(**fact_dict)
    else:
        fact_dict = copy.deepcopy(fact_dict)
        fact_parent_entry = fact_dict.pop("fact_parent")
        fact_parent = Fact(**fact_parent_entry)
        return Fact(**fact_dict, fact_parent=fact_parent)


@dataclass
class Fact:
    subject: str
    rel_lemma: str
    object: str
    rel_p_id: str
    query: str
    fact_paragraph: str = None
    fact_parent: "Fact" = None
    intermediate_paragraph: str = None

    def get_subject(self) -> str:
        return self.subject

    def get_relation_property_id(self) -> str:
        return self.rel_p_id

    def get_object(self) -> str:
        return self.object

    def get_relation(self) -> str:
        return self.rel_lemma

    def get_paragraph(self) -> str:
        return self.fact_paragraph

    def get_intermediate_paragraph(self) -> str:
        return self.intermediate_paragraph

    def get_parent(self) -> "Fact":
        return self.fact_parent

    def get_query(self) -> str:
        return self.query

    def as_tuple(self):
        return self.subject, self.rel_p_id, self.object

    def as_dict(self):
        output = copy.deepcopy(self.__dict__)
        if self.fact_parent is not None:
            output["fact_parent"] = output["fact_parent"].as_dict()
        return output

    def __eq__(self, o: "Fact") -> bool:
        return o.subject == self.subject and o.object == self.object and o.rel_p_id == self.rel_p_id


# %%
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import AbstractContextManager
from typing import Dict


class ResumeAndSaveDataset(AbstractContextManager, ABC):
    """
    A context manager that resumes processing a data from where it left off, saves the output periodically,
    and ensures the output is saved in case of an exception.
    """

    def __init__(self, path, save_interval=1000):
        """
        Initializes the ResumeAndSaveDataset context manager.

        Args:
            path (str): The path of the data file.
            save_interval (int, optional): The number of entries to process before saving the output data.
                                           Defaults to 20.
        """
        self.path = path
        self.output_dataset = self.load_output_dataset()
        self.save_interval = save_interval
        self.entries_since_last_save = 0

    def load_output_dataset(self):
        """
        Loads the output data from the specified file. If the file is not found, an empty list is returned.

        Returns:
            List: The output data loaded from the file or an empty list if the file is not found.
        """
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            get_logger().info(f"Loaded {len(data)} previously computed entries from the data stored in {self.path}.")
            return data
        except FileNotFoundError:
            return []

    @abstractmethod
    def is_input_processed(self, inp: dict):
        """
        Check if the input dictionary has been processed.

        Args:
            inp (dict): A dictionary containing the input data to be checked.

        Returns:
            bool: True if the input has been processed, False otherwise.
        """
        pass

    def add_entry(self, entry):
        """
        Appends a new data entry to the output_dataset and saves the output data if the save_interval is reached.

        Args:
            entry: The data entry to be added.
        """
        self.output_dataset.append(entry)
        self.entries_since_last_save += 1

        if self.entries_since_last_save >= self.save_interval:
            self.save_output_dataset()
            self.entries_since_last_save = 0

    def save_output_dataset(self):
        """
        Saves the output data to the file.
        """
        with open(self.path, "w+") as f:
            json.dump(self.output_dataset, f, indent=4)

    def __enter__(self):
        """
        The enter method for the context manager.

        Returns:
            ResumeAndSaveDataset: The instance of the context manager.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        The exit method for the context manager. Saves the output data to the file and prints a message.

        Args:
            exc_type: The type of the exception, if any.
            exc_value: The instance of the exception, if any.
            traceback: A traceback object, if any.

        Returns:
            bool: False to propagate the exception, True to suppress it.
        """
        self.save_output_dataset()
        if exc_type is not None:
            get_logger().info(f"Output data saved in the following location due to an exception: {self.path}")
        else:
            get_logger().info(f"Output data saved in the following location: {self.path}")
        return False


class ResumeAndSaveFactDataset(ResumeAndSaveDataset):
    def __init__(self, path, save_interval=20):
        super().__init__(path, save_interval)
        self.entry_processed = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: False))))

        for entry in self.output_dataset:
            entry_fact = fact_from_dict(entry["fact"] if "fact" in entry else entry)
            subject = entry_fact.get_subject()
            rel = entry_fact.get_relation_property_id()
            obj = entry_fact.get_object()
            intermediate_paragraph = entry_fact.get_intermediate_paragraph()
            self.entry_processed[subject][rel][obj][intermediate_paragraph] = True

    def is_input_processed(self, fact: Fact):
        return self.entry_processed[fact.get_subject()][fact.get_relation_property_id()][fact.get_object()][
            fact.get_intermediate_paragraph()
        ]

    def add_entry(self, entry: Dict):
        super().add_entry(entry)

        if "fact" in entry:
            fact = fact_from_dict(entry["fact"])
        else:
            fact = fact_from_dict(entry)

        self.entry_processed[fact.get_subject()][fact.get_relation_property_id()][fact.get_object()][
            fact.get_intermediate_paragraph()
        ] = True


# %% md
# Tracer
# %%
import re
from typing import Union, List
import torch
from operator import attrgetter
import subprocess
import torchaudio
from transformers.models.whisper.modeling_whisper import WhisperDecoder
from transformers import AutoProcessor


# Code partially adapted from https://github.com/kmeng01/rome

def run_festival(text, tempname):
    with open(f"{tempname}.txt", "w") as f:
        f.write(text)
    subprocess.run(f"text2wave {tempname}.txt -o {tempname}.wav".split(" "))
    waveform, sample_rate = torchaudio.load(f"{tempname}.wav")
    return waveform[0].numpy(), sample_rate


class ModelForwarder:
    def __init__(self, tempname):
        self.processor = AutoProcessor.from_pretrained("openai/whisper-base.en")
        self.tempname = tempname

    def forward(self, model, tokenizer, prompt, device, repeat, obj):
        samplep = run_festival(prompt, self.tempname)
        samples = run_festival(obj, self.tempname)
        sample_rate = samplep[1]
        sample = np.concatenate([samplep[0], samples[0] + np.random.normal(0, np.full_like(samples[0], 0.2))], 0)
        input_features = self.processor([sample], sampling_rate=sample_rate, return_tensors="pt",
                                        pad_to_multiple_of=8).input_features[
                         -model.config.max_source_positions + 5:].to(device).to(model.dtype)
        decoder_input = torch.tensor([tokenizer.encode(prompt)[:-1]], device=device)[
                        -model.config.max_target_positions + 5:]
        logits = model.forward(torch.cat([input_features] * 2, 0) if repeat else input_features,
                               decoder_input_ids=torch.cat([decoder_input] * 2, 0) if repeat else decoder_input)
        return logits


def get_next_token(model, tokenizer, prompt, device, model_forwarder, repeat, obj):
    # Prepare inputs

    # Feed model
    with torch.no_grad():
        next_token_logits = model_forwarder.forward(model, tokenizer, prompt, device, repeat, obj)["logits"].detach()[0,
                            -1, :]

    # Find the token with the highest probability and its logit
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    max_prob_indices = torch.argsort(next_token_probs, descending=False)
    max_prob_indices = max_prob_indices[np.searchsorted(np.flip(np.cumsum(np.flip(max_prob_indices))), 0.9):]
    return [tokenizer.convert_ids_to_tokens(x.item()) for x in max_prob_indices], next_token_probs[
        max_prob_indices].tolist()


def get_next_token_probabilities(
        model, tokenizer, prompt: str, target_tokens: Union[str, List[str]], device, model_forwarder, repeat, obj
):
    # Make input as list of strings if a single string was given

    if type(target_tokens) == str:
        target_tokens = [target_tokens]

    # Prepare inputs
    target_token_ids = [tokenizer.convert_tokens_to_ids(next_token) for next_token in target_tokens]

    # Feed model
    with torch.no_grad():
        next_token_logits = model_forwarder.forward(model, tokenizer, prompt, device, repeat, obj)["logits"][:, -1, :]

    # Extract target token logits and probabilities
    target_token_probs = torch.softmax(next_token_logits, dim=-1)[:, target_token_ids]

    return target_token_probs


def adapt_target_tokens(tokenizer, target_tokens: List[str], preprend_space: bool):
    """
    Make sure that target_tokens contain correspond to only a single token
    """
    if preprend_space:
        target_tokens = [" " + token.lstrip() for token in target_tokens]

    target_tokens = [tokenizer.tokenize(token)[0] for token in target_tokens]

    return target_tokens


def find_substring_range(tokenizer, string, substring):
    string_ids = tokenizer(
        string,
        return_tensors=None,
        return_token_type_ids=False,
    )["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(string_ids)
    string = "".join(tokens)

    substring_ids = tokenizer.tokenize(substring)
    substring = "".join(substring_ids)

    char_loc = string.rindex(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(tokens):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break

    return tok_start, tok_end


def get_module_name(model, kind, num=None):
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind == "hidden" else "." + kind}'
    if isinstance(model, WhisperDecoder):
        if kind == "embed":
            return "model.decoder.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        if kind == "mlp":
            kind = "fc2"
        return f'model.decoder.layers.{num}{"" if kind == "hidden" else "." + kind}'
    if hasattr(model, "model"):
        if kind == "embed":
            return "model.decoder.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        if kind == "mlp":
            kind = "fc2"
        return f'model.decoder.layers.{num}{"" if kind == "hidden" else "." + kind}'
    assert False, "unknown transformer structure"


def get_num_layers(model):
    return len(
        [n for n, m in model.named_modules() if (re.match(r"^(transformer|model|model.decoder)\.(h|layers)\.\d+$", n))])


def get_num_tokens(tokenizer, string):
    tokens_ids = tokenizer(
        string,
        return_tensors=None,
        return_token_type_ids=False,
    )["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(tokens_ids)
    return len(tokens)


def find_submodule(module, name):
    """
    Finds the named module within the given model.
    """
    for n, m in module.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def get_embedding(model, token_id, device):
    # Prepare inputs
    token_ids = torch.tensor([[token_id]], device=device)

    # Feed model
    embed_module = model.model.decoder.embed_tokens
    embedding = embed_module(token_ids)[0, 0, :]

    return embedding


# %%
import torch
import torch.nn as nn
from tokenizers import Tokenizer


# Code partially adapted from https://github.com/kmeng01/rome

class MaskedCausalTracer:
    def __init__(self, model: nn.Module, tokenizer: Tokenizer, mask_token: str, model_forwarder):
        self.device = next(model.parameters()).device
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token = mask_token
        self.mask_token_embedding = self._get_mask_token_embedding(mask_token)
        self.model_forwarder = model_forwarder

    def _get_mask_token_embedding(self, mask_token):
        token_attr = f"{mask_token}_token_id"
        if getattr(self.tokenizer, token_attr, None) is not None:
            mask_token_id = getattr(self.tokenizer, token_attr)
        else:
            raise ValueError("No such token in the tokenizer.")
        with torch.no_grad():
            corrupted_token_embedding = get_embedding(self.model, mask_token_id, self.device).clone()
        return corrupted_token_embedding

    def trace_with_patch(
            self,
            prompt,
            range_to_mask,  # A tuple (start, end) of tokens to corrupt
            target_tokens,  # Tokens whose probabilities we are interested in
            states_to_patch,  # A list of tuples (token index, modules) of states to restore
            embedding_module_name,  # Name of the embedding layer
            obj
    ):
        def untuple(x):
            return x[0] if isinstance(x, tuple) else x

        hooks = []

        # Add embedding hook
        def hook_embedding(module, input, output):
            output[1, range_to_mask[0]: range_to_mask[1]] = self.mask_token_embedding.clone()
            return output

        embedding_module = find_submodule(self.model, embedding_module_name)
        embedding_hook = embedding_module.register_forward_hook(hook_embedding)
        hooks.append(embedding_hook)

        # Add hooks for the modules to restore
        for token_to_restore, modules_to_restore in states_to_patch:
            for module_name in modules_to_restore:
                def restoring_hook(module, input, output):
                    h = untuple(output)
                    h[1, token_to_restore] = h[0, token_to_restore].clone()
                    return output

                module = find_submodule(self.model, module_name)
                module_hook = module.register_forward_hook(restoring_hook)
                hooks.append(module_hook)
        with torch.no_grad():
            probs = get_next_token_probabilities(self.model, self.tokenizer, prompt, target_tokens, self.device,
                                                 self.model_forwarder, True, obj)
            clean_probs = probs[0, :]
            corrupted_probs = probs[1, :]

        for hook in hooks:
            hook.remove()
        return clean_probs, corrupted_probs


# %%
import os
import numpy as np
from tokenizers import Tokenizer
import torch
from tqdm import tqdm
from torch import nn


# Code partially adapted from https://github.com/kmeng01/rome

class Feature:
    def __init__(self, name):
        self.name = name
        self.ids = []
        self.d = []

    def get_name(self):
        return self.name

    def to_array(self):
        return np.array(self.d)

    def add(self, v):
        self.d.append(v)
        self.ids.append(len(self.d))

    def avg(self):
        np_array = np.array(self.d)
        return np.mean(np_array[~np.isnan(np_array)])

    def std(self):
        np_array = np.array(self.d)
        return np.std(np_array[~np.isnan(np_array)])

    def __len__(self):
        return len(self.d)

    def get_w_id(self, i):
        return self.d[i], self.ids[i]

    def get(self, i):
        return self.d[i]


def group_results(facts, bucket):
    labels = ["subj-first", "subj-middle", "subj-last", "cont-first", "cont-middle", "cont-last"]

    corrupted_probs = Feature("corr")
    clean_probs = Feature("clean")
    results = {kind: {labels[i]: Feature(labels[i]) for i in range(len(labels))} for kind in ["hidden", "mlp", "attn"]}

    target_token = f"{bucket}_token"

    for processed_fact in facts:
        processed_fact = processed_fact["results"]
        corrupted_score = processed_fact["corrupted"][target_token]["probs"]
        clean_score = processed_fact["clean"][target_token]["probs"]

        # If there is a zero interval, skip the fact
        interval_to_explain = max(clean_score - corrupted_score, 0)
        if interval_to_explain == 0:
            continue

        corrupted_probs.add(corrupted_score)
        clean_probs.add(clean_score)

        for kind in ["hidden", "mlp", "attn"]:
            (
                avg_first_subject,
                avg_middle_subject,
                avg_last_subject,
                avg_first_after,
                avg_middle_after,
                avg_last_after,
            ) = results[kind].values()

            tokens = processed_fact["tokens"]
            started_subject = False
            finished_subject = False
            temp_mid = 0.0
            count_mid = 0

            for token in tokens:
                interval_explained = max(token[kind][target_token]["probs"] - corrupted_score, 0)
                token_effect = min(interval_explained / interval_to_explain, 1)

                if "subject_pos" in token:
                    if not started_subject:
                        avg_first_subject.add(token_effect)
                        started_subject = True

                        if token["subject_pos"] == -1:
                            avg_last_subject.add(token_effect)
                    else:
                        subject_pos = token["subject_pos"]
                        if subject_pos == -1:
                            avg_last_subject.add(token_effect)
                        else:
                            temp_mid += token_effect
                            count_mid += 1
                else:
                    if not finished_subject:
                        # Process all subject middle tokens
                        if count_mid > 0:
                            avg_middle_subject.add(temp_mid / count_mid)
                            temp_mid = 0.0
                            count_mid = 0
                        else:
                            avg_middle_subject.add(0.0)
                        avg_first_after.add(token_effect)
                        finished_subject = True

                        if token["pos"] == -1:
                            avg_last_after.add(token_effect)
                    else:
                        token_pos = token["pos"]
                        if token_pos == -1:
                            avg_last_after.add(token_effect)
                        else:
                            temp_mid += token_effect
                            count_mid += 1

            if count_mid > 0:
                avg_middle_after.add(temp_mid / count_mid)
            else:
                avg_middle_after.add(0.0)

    return results, corrupted_probs, clean_probs


def find_all_substring_range(tokenizer, string, substring):
    string_ids = tokenizer(
        string,
        return_tensors=None,
        return_token_type_ids=False,
    )["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(string_ids)
    string = "".join(tokens)
    tokens1 = tokens

    substring_ids = tokenizer.tokenize(substring)
    substring = "".join(substring_ids)
    r = []
    while len(string) > 0 and string.count(substring) > 0:
        char_loc = string.rindex(substring)
        loc = 0
        tok_start, tok_end = None, None
        for i, t in enumerate(tokens1):
            loc += len(t)
            if tok_start is None and loc > char_loc:
                tok_start = i
            if tok_end is None and loc >= char_loc + len(substring):
                tok_end = i + 1
                break
        r.append((tok_start, tok_end))
        string = string[:char_loc]
        tokens1 = tokens1[:tok_start]
    return r, tokens


def process_entry(causal_tracer: MaskedCausalTracer, prompt: str, subject: str, obj: str, target_token: str,
                  bucket: str):
    output = dict()

    embedding_module_name = get_module_name(causal_tracer.model, "embed", 0)
    subject_tokens_range = find_substring_range(causal_tracer.tokenizer, prompt, subject)

    object_tokens_range, tokens = find_all_substring_range(causal_tracer.tokenizer, prompt, obj)
    output["object_tokens_range"] = object_tokens_range
    output["subject_tokens_range"] = subject_tokens_range

    # Get corrupted run results
    clean_probs, corrupted_probs = causal_tracer.trace_with_patch(
        prompt, subject_tokens_range, [target_token], [(None, [])], embedding_module_name, obj
    )
    corrupted_output = {"token": target_token, "probs": corrupted_probs[0].item()}
    clean_output = {"token": target_token, "probs": clean_probs[0].item()}

    output["results"] = {
        "corrupted": {
            f"{bucket}_token": corrupted_output,
        },
        "clean": {
            f"{bucket}_token": clean_output,
        },
    }

    # Get patched runs results
    num_tokens = get_num_tokens(causal_tracer.tokenizer,
                                prompt) - 1  # TODO the -1 should not be necesary nd it does not belong here for models other than Whisper
    output["results"]["tokens"] = list()
    # We start the loop from the first subject token as patching previous tokens has no effect
    for token_i in (list(range(subject_tokens_range[0], num_tokens))):
        d = {}
        d["pos"] = token_i - num_tokens
        d["val"] = tokens[token_i]

        # If token is part of the subject, store its relative negative position
        if subject_tokens_range[0] <= token_i < subject_tokens_range[1]:
            d["subject_pos"] = token_i - subject_tokens_range[1]
        nl = get_num_layers(causal_tracer.model)
        params = [(kind, last_layer) for kind in ["hidden", "mlp", "attn"] for
                  last_layer in range(1, nl + 1, 1)]
        patches = [(0, len(tokens))]
        params_all = [(kind, last_layer, patch) for kind, last_layer in params
                      for patch in patches]
        for kind, last_layer, patch in params_all:
            states_to_patch = (
                token_i,
                [
                    get_module_name(causal_tracer.model, kind, L)
                    for L in range(
                    0,
                    last_layer,
                )
                ],
            )
            _, patched_probs = causal_tracer.trace_with_patch(
                prompt, subject_tokens_range, [target_token], [states_to_patch], embedding_module_name, obj
            )
            patched_output = {"token": target_token, "probs": patched_probs[0].item()}
            patched_results = {
                f"{bucket}_token": patched_output,
            }
            if kind not in d:
                d[kind] = {}
            d[kind][last_layer] = patched_results
        output["results"]["tokens"].append(d)
    return output


def construct_prompt(fact: Fact, prompt_template):
    prompt = prompt_template.format(query=fact.get_query(), context=fact.get_paragraph())
    return prompt


def run_causal_tracing_analysis(
        model: nn.Module,
        tokenizer: Tokenizer,
        fakepedia,
        prompt_template,
        num_grounded,
        num_unfaithful,
        prepend_space,
        resume_dir,
        model_forwarder,
        skip_creation=True
):
    # We keep the results in two different files: unfaithful and grounded
    #
    # For each fact:
    #
    # Verify if the answer of the model is the unfaithful object or the grounded object. If the answer is another token, then skip the fact.
    # Put the fact in the corresponding list.
    #
    # Once we have processed all the facts, for each list and for each fact of the list we run the causal tracer.
    # Finally, we save the results in the corresponding file.

    device = next(model.parameters()).device
    logger = get_logger()

    if resume_dir is None:
        resume_dir = get_output_dir()
    os.makedirs(resume_dir, exist_ok=True)

    partial_path = os.path.join(resume_dir, "partial.json")

    if not skip_creation:
        with ResumeAndSaveFactDataset(partial_path, 10) as partial_dataset:
            for entry in tqdm(fakepedia, desc="Filtering facts"):
                fact = fact_from_dict(entry)
                if partial_dataset.is_input_processed(fact):
                    continue

                # Adapt unfaithful and grounded objects
                target_tokens = adapt_target_tokens(
                    tokenizer, [fact.get_parent().get_object(), fact.get_object()], prepend_space
                )

                # Predict most likely next token
                prompt = construct_prompt(fact, prompt_template)
                most_likely_next_token, _ = get_next_token(model, tokenizer, prompt, device, model_forwarder, False,
                                                           fact.get_object())
                unfaithful = target_tokens[0] in most_likely_next_token and not target_tokens[
                                                                                    1] in most_likely_next_token
                grounded = target_tokens[1] in most_likely_next_token and not target_tokens[0] in most_likely_next_token
                partial_dataset.add_entry(
                    {
                        "fact": fact.as_dict(),
                        "partial_results": {
                            "prompt": prompt,
                            "next_token": target_tokens[0] if unfaithful else target_tokens[1] if grounded else
                            most_likely_next_token[0],
                            "unfaithful_token": target_tokens[0],
                            "grounded_token": target_tokens[1],
                            "is_unfaithful": unfaithful,
                            "is_grounded": grounded,
                        },
                    }
                )

    partial_dataset = read_json(partial_path)

    unfaithful_facts = []
    grounded_facts = []

    for entry in partial_dataset:
        if entry["partial_results"]["is_grounded"]:
            grounded_facts.append(entry)
        elif entry["partial_results"]["is_unfaithful"]:
            unfaithful_facts.append(entry)

    logger.info(f"Found {len(unfaithful_facts)} unfaithful facts and {len(grounded_facts)} grounded facts")

    causal_tracer = MaskedCausalTracer(model, tokenizer, "eos", model_forwarder)

    for bucket in ["grounded", "unfaithful"]:
        if bucket == "unfaithful":
            if num_unfaithful == -1:
                num_unfaithful = len(unfaithful_facts)
            facts = unfaithful_facts[:num_unfaithful]
        else:
            if num_grounded == -1:
                num_grounded = len(grounded_facts)
            facts = grounded_facts[:num_grounded]

        num_facts = len(facts)

        causal_traces_path = os.path.join(resume_dir, f"{bucket}.json")

        logger.info(f"Running causal tracing on {num_facts} {bucket} facts")
        with ResumeAndSaveFactDataset(causal_traces_path, save_interval=1) as dataset:
            for entry in tqdm(facts, desc=f"Running causal tracing on {bucket} facts"):

                fact = fact_from_dict(entry["fact"])

                if dataset.is_input_processed(fact):
                    continue

                prompt = entry["partial_results"]["prompt"]
                target_token = entry["partial_results"]["next_token"]

                output_entry = process_entry(causal_tracer, prompt, fact.get_subject(), fact.get_object(), target_token,
                                             bucket)

                output_entry["fact"] = fact.as_dict()

                dataset.add_entry(output_entry)


# %% md
# Logger and detection
# %%
import logging
import os
from datetime import datetime
from logging import Logger
from typing import Any

import pytz

import inspect


def prepare_logger(output_dir: str) -> Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")

    # Log to file
    fh = logging.FileHandler(os.path.join(output_dir, "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def prepare_output_dir(base_dir: str = "./runs/") -> str:
    experiment_dir = os.path.join(
        base_dir, datetime.now(tz=pytz.timezone("Europe/Zurich")).strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


output_dir = prepare_output_dir()
logger = prepare_logger(output_dir)


def freeze_args(args: Any) -> None:
    # Retrieve caller filename
    caller_frame = inspect.stack()[1]
    caller_filename_full = caller_frame.filename
    caller_filename_only = os.path.splitext(os.path.basename(caller_filename_full))[0]

    # Save args to json file
    save_json(args.__dict__, os.path.join(output_dir, f"{caller_filename_only}_args"))


def get_output_dir() -> str:
    return output_dir


def get_logger() -> Logger:
    return logger


# %%
from typing import Dict
import os
import json


def save_json(data: object, json_path: str) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_json(json_path: str) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# %%
from typing import Tuple
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree


def generate_datasets(
        grounded_results,
        unfaithful_results,
        train_ratio=0.8,
        n_samples_per_label=2000,
        ablation_only_clean=False,
        ablation_include_corrupted=False,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    logger = get_logger()
    buckets = [grounded_results, unfaithful_results]

    if ablation_only_clean:
        feature_names = [grounded_results[2].get_name()]

        if ablation_include_corrupted:
            feature_names.append(grounded_results[1].get_name())

    else:
        feature_names = [
            f"{kind}-{feature}" for kind, features in grounded_results[0].items() for feature in features.keys()
        ]

    logger.info(f"Feature names: {feature_names}")

    all_samples = []
    all_labels = []

    for label, bucket_results in enumerate(buckets):
        kinds_results, corr_probs, clean_probs = bucket_results

        num_samples = len(corr_probs)

        logger.info("Number of samples: {}".format(num_samples))

        current_label_samples = []

        for i in range(num_samples):
            if ablation_only_clean:
                candidate_example = [clean_probs.get(i)]

                if ablation_include_corrupted:
                    candidate_example.append(corr_probs.get(i))

            else:
                candidate_example = [
                    feature_results.get(i)
                    for kind_results in kinds_results.values()
                    for feature_results in kind_results.values()
                ]

                if any([feature is None for feature in candidate_example]):
                    continue

            current_label_samples.append((candidate_example, label))

        if len(current_label_samples) < n_samples_per_label:
            raise ValueError(
                f"Bucket {label} has fewer than {n_samples_per_label} valid samples! In particular, there are {len(current_label_samples)} samples."
            )

        # Shuffle the samples for this label and take the first n_samples_per_label samples
        np.random.shuffle(current_label_samples)
        all_samples.extend([sample[0] for sample in current_label_samples[:n_samples_per_label]])
        all_labels.extend([sample[1] for sample in current_label_samples[:n_samples_per_label]])

    # Convert all_samples and all_labels to np arrays
    all_samples_array = np.array(all_samples)
    all_labels_array = np.array(all_labels)

    # Calculate lengths for each split
    total_size = len(all_samples_array)
    train_size = int(total_size * train_ratio)

    # Shuffle and split the dataset
    indices = np.arange(total_size)
    np.random.shuffle(indices)

    train_dataset = (all_samples_array[indices[:train_size]], all_labels_array[indices[:train_size]])
    test_dataset = (all_samples_array[indices[train_size:]], all_labels_array[indices[train_size:]])
    return train_dataset, test_dataset, feature_names


def load_metrics(save_dir):
    with open(os.path.join(save_dir, "results.json"), "r") as file:
        results = json.load(file)
    return results


def save_metrics(results, feature_names, save_dir):
    with open(os.path.join(save_dir, "results.json"), "w") as file:
        json.dump(results, file, indent=4)

    if "feature_importances" in results:
        importances = results["feature_importances"]
        indices = np.argsort(importances)

        # Logic to determine the kind for colors
        colors = {"hidden": "grey", "mlp": "blue", "attn": "orange", "corr": "grey", "clean": "grey"}

        def determine_kind(verbose_name):
            for kind, color in colors.items():
                if kind in verbose_name.lower():
                    return color
            print(f"Unmatched feature: {verbose_name}")
            raise ValueError("Unknown feature kind.")

        bar_colors = [determine_kind(name) for name in feature_names]

        # Make the font size larger
        plt.rcParams.update({"font.size": 21})

        # Change the font family
        plt.rcParams["font.family"] = "serif"

        plt.figure(figsize=(15, 15))
        plt.barh(
            range(len(indices)),
            [importances[i] for i in indices],
            align="center",
            color=[bar_colors[i] for i in indices],
            edgecolor="white",
        )
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "feature_importances.png"))
        plt.close()


def save_decision_tree_plot(tree, feature_names, class_names, save_dir):
    plt.figure(figsize=(200, 100))
    plot_tree(tree, feature_names=feature_names, class_names=class_names, filled=True)
    plt.savefig(os.path.join(save_dir, "decision_tree.png"))
    plt.close()


def plot_metrics_comparison(metrics_by_model, save_dir):
    """
    metrics_by_model: dict, keys are model names (like 'Logistic Regression', 'DecisionTree', 'XGBoost') and values are
                      dictionaries of metrics (keys are metric names, values are metric values)
    save_dir: directory where plots will be saved
    """
    model_colors = {"LogisticRegression": "grey", "DecisionTree": "orange", "XGBoost": "blue"}

    # Validate that all models in metrics_by_model are known
    for model in metrics_by_model:
        if model not in model_colors:
            raise Exception(f"Unknown model: {model}")

    n_models = len(metrics_by_model)
    n_metrics = len(metrics_by_model[next(iter(metrics_by_model))])

    # Set bar width, distance between bars in a group, and positions
    bar_width = 0.2
    distance = 0.05  # distance between bars in a group
    r1 = np.arange(n_metrics)  # positions for first model
    r2 = [x + bar_width + distance for x in r1]  # positions for second model
    r3 = [x + bar_width + distance for x in r2]  # positions for third model

    # Make the font size larger
    plt.rcParams.update({"font.size": 21})

    # Change the font family
    plt.rcParams["font.family"] = "serif"

    plt.figure(figsize=(15, 10))

    # Plotting bars for each model
    all_metric_values = []
    for idx, (model, metrics) in enumerate(metrics_by_model.items()):
        metric_values = [metrics[metric] for metric in metrics]
        all_metric_values.extend(metric_values)
        positions = [r1, r2, r3][idx]
        plt.bar(positions, metric_values, color=model_colors[model], width=bar_width, edgecolor="white", label=model)

    # Adjust y-axis limit
    plt.ylim(bottom=min(all_metric_values) * 0.9)

    plt.xlabel("Metrics")
    plt.ylabel("Score")
    xtick_positions = [r2[i] for i in range(n_metrics)]  # Averages of r1 and r2 positions
    plt.xticks(xtick_positions, list(metrics_by_model[next(iter(metrics_by_model))]))

    # Place the legend outside the plot on the right
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "all_metrics_comparison.png"), bbox_inches="tight")
    plt.close()


def train_and_save(models, train_data, test_data, feature_names, class_names, seed, replot_only=False):
    save_dir = get_output_dir()
    plt.rcParams["font.size"] = max(1, plt.rcParams["font.size"])

    metrics_by_model = {}

    for model_name, model_info in models.items():
        model_save_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)

        if not replot_only:
            X_train, y_train = train_data
            X_test, y_test = test_data

            if "random_state" in model_info["model"].get_params():
                model_info["model"].set_params(random_state=seed)

            clf = GridSearchCV(model_info["model"], model_info["param_grid"], cv=5, verbose=10)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_train_pred = clf.predict(X_train)
            y_train_proba = clf.predict_proba(X_train)[:, 1]
            with open(f"{model_name}_confusion_matrix.json", "w") as f:
                json.dump(confusion_matrix(y_test, y_pred).tolist(), f)
            results = {
                "train": {
                    "accuracy": accuracy_score(y_train, y_train_pred),
                    "precision": precision_score(y_train, y_train_pred, average='weighted'),
                    "recall": recall_score(y_train, y_train_pred, average='weighted'),
                    "f1_score": f1_score(y_train, y_train_pred, average='weighted'),
                    # "roc_auc": roc_auc_score(y_train, y_train_proba,
                    #                         multi_class = "ovr", average='weighted'),
                },
                "test": {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average='weighted'),
                    "recall": recall_score(y_test, y_pred, average='weighted'),
                    "f1_score": f1_score(y_test, y_pred, average='weighted'),
                    # "roc_auc": roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1],
                    #                         multi_class = "ovr", average='weighted'),
                },
                "best_hyperparameters": clf.best_params_,
            }

            if hasattr(clf.best_estimator_, "feature_importances_"):
                # If there is an importance type attribute, print it
                if hasattr(clf.best_estimator_, "importance_type"):
                    print(f"Feature importances: {clf.best_estimator_.importance_type}")
                results["feature_importances"] = list(clf.best_estimator_.feature_importances_)
                results["feature_importances"] = [float(val) for val in results["feature_importances"]]

            if isinstance(clf.best_estimator_, LogisticRegression):
                # Taking the absolute values of the coefficients
                results["feature_importances"] = [float(abs(val)) for val in clf.best_estimator_.coef_.flatten()]

            save_metrics(results, feature_names, model_save_dir)

            if model_name == "DecisionTree":
                save_decision_tree_plot(clf.best_estimator_, feature_names, class_names, model_save_dir)
        else:
            results = load_metrics(model_save_dir)
            save_metrics(results, feature_names, model_save_dir)

        metrics_by_model[model_name] = results["test"]

    plot_metrics_comparison(metrics_by_model, save_dir)


# %% md
# Main
# %%
import argparse
from types import SimpleNamespace
import torch
from multiprocessing import Pool

locs = ["openai/whisper-base.en"]


class Namespace1:
    def __init__(self):
        self.token = None
        self.fakepedia_path = "base_fakepedia.json"
        self.model_name_path = locs[-1]
        self.prompt_template = "{context} {query}"
        self.num_grounded = 10
        self.num_unfaithful = 10
        self.prepend_space = True
        self.bfloat16 = True
        self.resume_dir = "./specific_runs/run4"
        self.subset_size = 1000
        self.skip_creation = False


class mock_model:
    def __init__(self):
        self.config = SimpleNamespace()
        self.config.pad_token_id = 0
        self.config.eos_token_id = 0
        self.config.vocab_size = 1000
        self.device_map = "cpu"


def make_model(args, mock=False):
    from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, BitsAndBytesConfig
    cuda = torch.cuda.is_available()
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) if cuda else None
    model = mock_model() if mock else AutoModelForSpeechSeq2Seq.from_pretrained(args.model_name_path, token=args.token,
                                                                                force_download=False,
                                                                                quantization_config=quantization_config,
                                                                                device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_path, token=args.token, force_download=False,
                                              add_bos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer


def run_causal_tracing_analysis_wrapper(params):
    i, c, resume_dir, args = params
    fakepedia = read_json(args.fakepedia_path)[:args.subset_size]
    fakepedia = fakepedia[(i * len(fakepedia)) // c:((i + 1) * len(fakepedia)) // c]
    model, tokenizer = make_model(args)
    run_causal_tracing_analysis(
        model,
        tokenizer,
        fakepedia,
        args.prompt_template,
        args.num_grounded,
        args.num_unfaithful,
        args.prepend_space,
        resume_dir,
        ModelForwarder(f"./temp/temp{i:02d}"),
        args.skip_creation
    )


def run_causal_tracing(args):
    logger = get_logger()

    logger.info("Loading fakepedia...")
    # 23 kinds of relations 1673 unique templates.
    fakepedia = read_json(args.fakepedia_path)
    print(len(set([x["subject"] for x in fakepedia])),
          len(set([x["rel_p_id"] for x in fakepedia])))

    logger.info("Starting causal tracing...")
    pool_size = 5
    with Pool(pool_size) as p:
        params = [(i, pool_size, f"{args.resume_dir}/{i:02d}", args) for i in range(pool_size)]
        for x in params:
            os.makedirs(x[2], exist_ok=True)
        p.map(run_causal_tracing_analysis_wrapper, params)


def main1():
    args = Namespace1()
    freeze_args(args)
    run_causal_tracing(args)


# %%
import os
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import os
import argparse
import os
import random
import numpy as np
import torch
import shutil


def set_seed_everywhere(seed: int) -> None:
    # Set torch and numpy and random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Namespace2:
    def __init__(self):
        self.causal_traces_dir = "./"
        self.dataset_name = "simple"
        self.model_name = "LLama"
        self.output_dir = "out"
        self.balance = False
        if False:
            self.features_to_include = ["subj-first", "subj-middle", "subj-last", "cont-first", "cont-middle",
                                        "cont-last"]
        else:
            self.features_to_include = ["subj-middle", "subj-last", "cont-middle", "cont-last"]
        self.kinds_to_include = ["hidden", "mlp"]
        self.train_ratio = 0.5
        self.ablation_only_clean = False
        self.ablation_include_corrupted = False
        self.seed = 2
        self.num_classes = 10
        self.n_samples_per_label = 2
        self.min_count = 2


def get_args():
    return Namespace2()


def process_facts2(target_token, facts, class_map, results, corrupted_probs, clean_probs, tokenizer):
    for processed_fact in facts:
        clas = class_map(processed_fact)
        corrupted_score = processed_fact["results"]["corrupted"][target_token]["probs"]
        clean_score = processed_fact["results"]["clean"][target_token]["probs"]

        # If there is a zero interval, skip the fact
        interval_to_explain = max(clean_score - corrupted_score, 0)

        corrupted_probs[clas].add(corrupted_score)
        clean_probs[clas].add(clean_score)

        for kind in ["hidden", "mlp", "attn"]:
            (
                avg_first_subject,
                avg_middle_subject,
                avg_last_subject,
                avg_first_after,
                avg_middle_after,
                avg_last_after,
            ) = results[clas][kind].values()

            tokens = processed_fact["results"]["tokens"]
            started_subject = False
            finished_subject = False
            temp_mid = 0.0
            count_mid = 0

            for token in tokens:
                interval_explained_average = 0
                for layer in token[kind]:
                    interval_explained_average += max(token[kind][layer][target_token]["probs"] - corrupted_score,
                                                      0) / len(token[kind])
                token_effect = min(interval_explained_average / interval_to_explain, 1)

                if "subject_pos" in token:
                    if not started_subject:
                        avg_first_subject.add(token_effect)
                        started_subject = True

                        if token["subject_pos"] == -1:
                            avg_last_subject.add(token_effect)
                    else:
                        subject_pos = token["subject_pos"]
                        if subject_pos == -1:
                            avg_last_subject.add(token_effect)
                        else:
                            temp_mid += token_effect
                            count_mid += 1
                else:
                    if not finished_subject:
                        # Process all subject middle tokens
                        if count_mid > 0:
                            avg_middle_subject.add(temp_mid / count_mid)
                            temp_mid = 0.0
                            count_mid = 0
                        else:
                            avg_middle_subject.add(0.0)
                        avg_first_after.add(token_effect)
                        finished_subject = True

                        if token["pos"] == -1:
                            avg_last_after.add(token_effect)
                    else:
                        token_pos = token["pos"]
                        if token_pos == -1:
                            avg_last_after.add(token_effect)
                        else:
                            temp_mid += token_effect
                            count_mid += 1

            if count_mid > 0:
                avg_middle_after.add(temp_mid / count_mid)
            else:
                avg_middle_after.add(0.0)


def filter_facts(processed_fact, target_token):
    corrupted_score = processed_fact["results"]["corrupted"][target_token]["probs"]
    clean_score = processed_fact["results"]["clean"][target_token]["probs"]

    interval_to_explain = max(clean_score - corrupted_score, 0)
    return interval_to_explain == 0


def group_results2(facts_grounded, facts_unfaithful, tokenizer, args):
    labels = ["subj-first", "subj-middle", "subj-last", "cont-first", "cont-middle", "cont-last"]
    num_classes = args.num_classes
    corrupted_probs = [Feature("corr") for _ in range(num_classes)]
    clean_probs = [Feature("clean") for _ in range(num_classes)]
    # Here results is a list of dictionaries, each dictionary contains the results for one class (i.e. label i.e. bucket), the bucket is decided in the process_facts2 function
    results = [
        {kind: {labels[i]: Feature(labels[i]) for i in range(6)} for kind in ["hidden", "mlp", "attn"]}
        for _ in range(num_classes)]
    if args.balance:
        # Some rel_lemma are not present in all sets.
        lemmauf = set(x["fact"]["rel_lemma"] for x in facts_unfaithful)
        lemmaf = set(x["fact"]["rel_lemma"] for x in facts_grounded)
        print("in facts_grounded not in facts_unfaithful", lemmaf.difference(lemmauf))
        print("in facts_unfaithful not in facts_grounded", lemmauf.difference(lemmaf))
        tempsuf = set(x["fact"]["subject"] for x in facts_unfaithful)
        tempsf = set(x["fact"]["subject"] for x in facts_grounded)
        # Templates not uniformly distributed, half of in unfaithful never appear in grounded
        print("num unique templates facts_unfaithful",
              len(tempsuf))
        print("num unique templates facts_grounded",
              len(tempsf))
        print("num templates in facts_grounded not in facts_unfaithful",
              len(tempsf.difference(tempsuf)))
        print("num templates in facts_unfaithful not in facts_grounded",
              len(tempsuf.difference(tempsf)))
        # Remove trivial samples
        trivial = tempsf.symmetric_difference(tempsuf)
        facts_unfaithful = [x for x in facts_unfaithful if x["fact"]["subject"] not in trivial]
        facts_grounded = [x for x in facts_grounded if x["fact"]["subject"] not in trivial]

    process_facts2("unfaithful_token", facts_unfaithful,
                   lambda x: 0, results, corrupted_probs, clean_probs, tokenizer)
    print([x["fact"]["object"] for x in facts_grounded if filter_facts(x, "grounded_token")])
    facts_grounded = [x for x in facts_grounded if not filter_facts(x, "grounded_token")]
    ps = [x["results"]["clean"]["grounded_token"]["probs"] for x in facts_grounded]
    class_boundaries = np.quantile(ps, np.linspace(0, 1, num_classes))[1:-1]

    def classify(fact):
        return np.digitize(fact["results"]["clean"]["grounded_token"]["probs"], class_boundaries)

    process_facts2("grounded_token", facts_grounded, classify
                   , results, corrupted_probs, clean_probs, tokenizer)
    # print("corrupted_probs, clean_probs", [x.d for x in corrupted_probs], [x.d for x in clean_probs])
    vs = list(
        (x, i, len(x[0]["hidden"]["subj-first"])) for i, x in enumerate(zip(results, corrupted_probs, clean_probs)))
    # print([x[2] for x in vs])
    vs = [(x, i) for x, i, l in vs if l >= args.min_count]
    # in next experiment try ["grounded", "confidently grounded"]
    return [x for x, i in vs], [f"p:{i}" for x, i in vs]


def generate_datasets2(buckets,
                       train_ratio=0.8,
                       n_samples_per_label=2000
                       ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Any]:
    logger = get_logger()
    feature_names = [
        f"{kind}-{feature}" for kind, features in buckets[0][0].items() for feature in features.keys()
    ]

    logger.info(f"Feature names: {feature_names}")

    all_samples = []
    all_labels = []

    for label, bucket_results in enumerate(buckets):
        kinds_results, corr_probs, clean_probs = bucket_results
        # This is the correct number of samples the feature adding is done in a wierd way but there is always
        # exactly one value in each feature for each sample
        num_samples = len(corr_probs)

        logger.info("Number of samples: {}".format(num_samples))
        print(label, [[(kind_name, feature_name, len(feature_results),
                        feature_results.avg(), feature_results.std())
                       for feature_name, feature_results in kind_results.items()] for kind_name, kind_results in
                      kinds_results.items()])

        current_label_samples = []

        for i in range(num_samples):
            candidate_example = [
                feature_results.get(i)
                for kind_results in kinds_results.values()
                for feature_results in kind_results.values()
            ]

            if any([feature is None for feature in candidate_example]):
                continue

            current_label_samples.append((candidate_example, label))

        if False and len(current_label_samples) < n_samples_per_label:
            raise ValueError(
                f"Bucket {label} has fewer than {n_samples_per_label} valid samples! In particular, there are {len(current_label_samples)} samples."
            )

        # Shuffle the samples for this label and take the first n_samples_per_label samples
        np.random.shuffle(current_label_samples)
        all_samples.extend([sample[0] for sample in current_label_samples[:n_samples_per_label]])
        all_labels.extend([sample[1] for sample in current_label_samples[:n_samples_per_label]])

    # Convert all_samples and all_labels to np arrays
    all_samples_array = np.array(all_samples)
    all_labels_array = np.array(all_labels)

    # Calculate lengths for each split
    total_size = len(all_samples_array)
    train_size = int(total_size * train_ratio)

    # Shuffle and split the dataset
    indices = np.arange(total_size)
    np.random.shuffle(indices)

    train_dataset = (all_samples_array[indices[:train_size]], all_labels_array[indices[:train_size]])
    test_dataset = (all_samples_array[indices[train_size:]], all_labels_array[indices[train_size:]])
    print(test_dataset[0].shape, test_dataset[0].shape, feature_names)
    return train_dataset, test_dataset, feature_names


"""
"param_grid": {
                "max_depth": [None, 5, 10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
"param_grid": {
                "max_depth": [3, 4],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
"""


def train_detector(args):
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000),
            "param_grid": {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
        },
        "DecisionTree": {
            "param_grid": {
                "max_depth": [5],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "ccp_alpha": [0, 0.01, 0.1, 0.5, 0.9]
            },
            "model": DecisionTreeClassifier(),

        }
    }

    buckets = ["grounded", "unfaithful"]
    buckets_paths = [
        os.path.join(args.causal_traces_dir, args.dataset_name, args.model_name, f"{bucket}.json") for bucket in buckets
    ]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai/whisper-base.en", force_download=False,
                                              add_bos_token=True)
    print([len(read_json(x)) for x in buckets_paths])
    results, buckets = group_results2(read_json(buckets_paths[0]),
                                      read_json(buckets_paths[1]),
                                      tokenizer, args)

    # If we are only including certain kinds, filter the kinds
    if args.kinds_to_include is not None:
        results = [
            (
                {kind: bucket_results[0][kind] for kind in bucket_results[0] if kind in args.kinds_to_include},
                bucket_results[1],
                bucket_results[2],
            )
            for bucket_results in results
        ]

    # If we are only including certain features, filter the features
    if args.features_to_include is not None:
        results = [
            (
                {
                    kind: {
                        feature: bucket_results[0][kind][feature]
                        for feature in bucket_results[0][kind]
                        if feature in args.features_to_include
                    }
                    for kind in bucket_results[0]
                },
                bucket_results[1],
                bucket_results[2],
            )
            for bucket_results in results
        ]
    print([[[len(z) for z in y] for y in x[0].values()] for x in results])
    # Generate the datasets
    train_data, test_data, feature_names = generate_datasets2(
        results,
        n_samples_per_label=args.n_samples_per_label,
        train_ratio=args.train_ratio
    )
    print(len(train_data), len(train_data[0]), train_data[1])
    print(len(test_data), len(test_data[0]), test_data[1])

    # Train the models and save the results
    train_and_save(models, train_data, test_data, feature_names, class_names=buckets, seed=args.seed)


def main2():
    sp = "./specific_runs/run4"
    p = './simple/LLama'
    names = ['grounded.json', 'unfaithful.json']
    for name in names:
        ds = []
        for x in os.listdir(sp):
            if os.path.exists(x + "/" + name):
                with open(x + "/" + name) as f:
                    ds.extend(json.load(f))
        with open(os.path.join(p, name), "w") as f:
            json.dump(ds, f, indent=4)
    os.makedirs(p, exist_ok=True)
    args = get_args()
    freeze_args(args)
    set_seed_everywhere(args.seed)
    train_detector(args)

if __name__ == "__main__":
    if True:
        main1()
    else:
        main2()
        # !rm -r LLama
        from sklearn.metrics import ConfusionMatrixDisplay

        models = {
            "LogisticRegression": {
                "model": LogisticRegression(max_iter=1000),
                "param_grid": {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            },
            "DecisionTree": {
                "model": DecisionTreeClassifier(),
                "param_grid": {
                    "max_depth": [5, 10, 15, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            }
        }
        for model_name in models:
            with open(f"{model_name}_confusion_matrix.json") as f:
                confm = json.load(f)
            disp = ConfusionMatrixDisplay(confusion_matrix=np.array(confm))
            disp.plot()
            plt.show()
    # %%
    from sklearn.metrics import ConfusionMatrixDisplay

    models = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000),
            "param_grid": {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(),
            "param_grid": {
                "max_depth": [None, 5, 10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        }
    }
    for model_name in models:
        with open(f"{model_name}_confusion_matrix.json") as f:
            confm = json.load(f)
        disp = ConfusionMatrixDisplay(confusion_matrix=np.array(confm))
        disp.plot()
        plt.show()