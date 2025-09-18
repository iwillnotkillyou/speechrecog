# %% md
# Data
# %%
# !wget -nc -O data.zip "https://www.dropbox.com/scl/fo/6rets9zuu0nvbbokxvf0e/AIa4cSpy_sjdd7Pz9aTxXa0?rlkey=vqj56pc1sp6qlggldrz5vdw32&st=6mf25abu&dl=0"
# !unzip -o data.zip
# !pip install bitsandbytes
# %%
import copy
import inspect
import json
import logging
import os
import random
import re
import subprocess
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from logging import Logger
from multiprocessing import Pool
from types import SimpleNamespace
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union, List
import matplotlib.pyplot as plt
import numpy as np
import pytz
import torch
import torchaudio
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from torch import nn
from tqdm import tqdm


def fact_from_dict(fact_dict: Dict):
    if fact_dict["fact_parent"] is None:
        return Fact(**fact_dict)
    else:
        fact_dict = copy.deepcopy(fact_dict)
        fact_parent_entry = fact_dict.pop("fact_parent")
        fact_parent = Fact(**fact_parent_entry)
        return Fact(**fact_dict, fact_parent=fact_parent)

class Fact:
    subject: str
    rel_lemma: str
    object: str
    rel_p_id: str
    query: str
    fact_paragraph: str = None
    fact_parent: "Fact" = None
    intermediate_paragraph: str = None
    adaptor: List[int] = None

    def __init__(self, **kwargs):
        self.subject = kwargs.get("subject", "")
        self.rel_lemma = kwargs.get("rel_lemma", "")
        self.object = kwargs.get("object", "")
        self.rel_p_id = kwargs.get("rel_p_id", "")
        self.query = kwargs.get("query", "")
        self.fact_paragraph = kwargs.get("fact_paragraph", "")
        self.intermediate_paragraph = kwargs.get("intermediate_paragraph", "")
        self.adaptor = kwargs.get("adaptor", [])
        self.fact_parent = kwargs.get("fact_parent", None)

    def get_subject(self) -> str:
        return self.subject

    def get_relation_property_id(self) -> str:
        return self.rel_p_id

    def get_object(self) -> str:
        return self.object


    def get_adaptor(self) -> List[int]:
        return self.adaptor

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
        if len(self.output_dataset) > 0:
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


# Code partially adapted from https://github.com/kmeng01/rome

def run_festival(text, tempname):
    with open(f"{tempname}.txt", "w") as f:
        f.write(text)
    subprocess.run(f"text2wave {tempname}.txt -o {tempname}.wav".split(" "))
    waveform, sample_rate = torchaudio.load(f"{tempname}.wav")
    return waveform[0].numpy(), sample_rate


def get_next_token(model, tokenizer, prompt, device, model_forwarder, repeat, obj):
    # Prepare the model

    # Feed model
    with torch.no_grad():
        next_token_logits = model_forwarder.forward(model, tokenizer, prompt, device, repeat, obj)["logits"].detach()[0,
                            -1, :].cpu()

    # Find the token with the highest probability and its logit
    next_token_probs = torch.softmax(next_token_logits, dim=-1).numpy()
    max_prob_indices = np.argsort(next_token_probs)
    max_prob_indices = max_prob_indices[np.searchsorted(np.flip(np.cumsum(np.flip(max_prob_indices))), 0.9):]
    print(list(zip(next_token_probs[max_prob_indices][-10:].tolist(),
                   tokenizer.convert_ids_to_tokens(max_prob_indices)[-10:])))
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
    """
    Finds the token range of the last occurence of the substring in the string.
    """
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
    if hasattr(model, "model") and not hasattr(model.model, "decoder"):
        if kind == "embed":
            return "model.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        return f'model.layers.{num}{"" if kind == "hidden" else "." + kind}'
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
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


# %%


# Code partially adapted from https://github.com/kmeng01/rome

class MaskedCausalTracer:
    def __init__(self, model: nn.Module, tokenizer, mask_token: str, model_forwarder):
        self.device = next(model.parameters()).device
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token = mask_token
        self.model_forwarder = model_forwarder
        self.mask_token_embedding = self._get_mask_token_embedding(mask_token)

    def _get_mask_token_embedding(self, mask_token):
        token_attr = f"{mask_token}_token_id"
        if getattr(self.tokenizer, token_attr, None) is not None:
            mask_token_id = getattr(self.tokenizer, token_attr)
        else:
            raise ValueError("No such token in the tokenizer.")
        with torch.no_grad():
            corrupted_token_embedding = self.model_forwarder.get_embedding(self.model, mask_token_id,
                                                                           self.device).clone()
        return corrupted_token_embedding

    def trace_with_patch(
            self,
            prompt,
            range_to_mask,  # A tuple (start, end) of tokens to corrupt
            target_tokens,  # Tokens whose probabilities we are interested in
            states_to_patch,  # A list of tuples (token index, modules) of states to restore
            embedding_module_name,  # Name of the embedding layer
            obj,
            adaptor=None
    ):
        def untuple(x):
            return x[0] if isinstance(x, tuple) else x

        hooks = []

        self.model_forwarder.set_adaptor(adaptor)

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

        self.model_forwarder.clear_adaptor()
        return clean_probs, corrupted_probs


# %%


# Code partially adapted from https://github.com/kmeng01/rome

class Feature:
    def __init__(self, name):
        self.name = name
        self.ids = []
        self.d = []

    def get_name(self):
        return self.name

    def indexer(self, inds):
        f = Feature(self.name)
        f.d = [self.d[i] for i in inds]
        f.ids = [self.ids[i] for i in inds]
        return f

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
        return self.d[int(i)]


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


def get_quantiles(l, c=5, additional_inds = ()):
    if len(l) < c:
        return np.unique(l).tolist()
    ps = np.linspace(0, 1, c, endpoint=True)
    return np.unique(np.rint(np.quantile(l, ps, method="closest_observation")).astype(np.int32).tolist()+list(additional_inds)).tolist()


def construct_prompt(fact: Fact, prompt_template):
    prompt = prompt_template.format(query=fact.get_query(), context=fact.get_paragraph())
    return prompt

# %% md
# Logger and detection
# %%


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


def save_json(data: object, json_path: str) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_json(json_path: str) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# %% md
# Main
# %%


class mock_model:
    def __init__(self):
        self.config = SimpleNamespace()
        self.config.pad_token_id = 0
        self.config.eos_token_id = 0
        self.config.vocab_size = 1000
        self.device_map = "cpu"

def set_seed_everywhere(seed: int) -> None:
    # Set torch and numpy and random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)