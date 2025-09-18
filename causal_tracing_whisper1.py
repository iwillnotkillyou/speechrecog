import torch

from causal_tracing_whisper import *
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

from transformers import LlamaForCausalLM, LlamaTokenizer

def forward_with_prefix(model: LlamaForCausalLM, tokenizer, prompt, device, target, prefix_embedding, repeat = False):
    def hook_embedding(module, input, output):
        output[:, :prefix_embedding.shape[0]] = prefix_embedding
        return output

    embedding_module_name = get_module_name(model, "embed", 0)
    embedding_module = find_submodule(model, embedding_module_name)
    embedding_hook = embedding_module.register_forward_hook(hook_embedding)

    decoder_input = torch.tensor([[0]*prefix_embedding.shape[0] + tokenizer.encode(prompt)], device=device)
    output_dict = model.forward(torch.cat([decoder_input] * 2, 0) if repeat else decoder_input, labels = torch.cat([torch.full_like(decoder_input, -100)[:, 1:], torch.tensor([[target]], device=device)], -1) if target is not None else None, return_dict=True)

    embedding_hook.remove()
    return output_dict


def prefix_tuning(model: LlamaForCausalLM, tokenizer: LlamaTokenizer, prompt, device, model_forwarder, objects):
    mask_token_id = tokenizer.eos_token_id
    for x in model.parameters():
        x.requires_grad = False
    objects = np.array(objects)[[0, 1]] if False else np.random.permutation(objects)
    print(objects)
    target = tokenizer.encode((" " if not prompt.endswith(" ") else "")+objects[0], add_special_tokens=False)[0]
    if len(tokenizer.decode([target]).strip()) == 0:
        target = tokenizer.encode(objects[0], add_special_tokens=False)[0]
        if not prompt.endswith(" "):
            prompt = prompt + " "
    distractor = tokenizer.encode((" " if not prompt.endswith(" ") else "")+objects[1], add_special_tokens=False)[0]
    if len(tokenizer.decode([distractor]).strip()) == 0:
        target = tokenizer.encode(objects[1], add_special_tokens=False)[0]
        if not prompt.endswith(" "):
            prompt = prompt + " "
    prefix_embedding = model_forwarder.get_embedding(model, mask_token_id, device).clone()
    prefix_embedding = torch.stack([prefix_embedding] * 1, 0)
    prefix_embedding.requires_grad = True
    hist = []
    max_steps = 100
    succeeded = False
    for x in range(max_steps):
        xfrac = x/max_steps
        t = x % 2 == 0 if False else True
        with torch.enable_grad():
            output_dict = forward_with_prefix(model, tokenizer, prompt, device, target if t else distractor, prefix_embedding)
        next_token_logits = output_dict["logits"].detach()[0,-1, :].cpu()
        next_token_probs = torch.softmax(next_token_logits, dim=-1).numpy()
        max_prob_indices = np.argsort(next_token_probs)
        max_prob_indices = max_prob_indices[np.searchsorted(np.flip(np.cumsum(np.flip(max_prob_indices))), 0.9):]
        hist.append(list(zip(next_token_probs[max_prob_indices][-10:].tolist(),
                       tokenizer.convert_ids_to_tokens(max_prob_indices)[-10:]))[-3:])
        loss = output_dict["loss"]
        loss.backward()
        prefix_embedding.data = prefix_embedding.data + (- 1 if t else 1) * (0.00001*xfrac+(1-xfrac)*(0.0001)) * prefix_embedding.grad.data
        prefix_embedding.grad.zero_()
        succeeded = max_prob_indices[-1] == target or (max_prob_indices[-2] == target and max_prob_indices[-1] != distractor and max_prob_indices[-3] != distractor)
        if succeeded:
            break
    print("\n".join([str(x) for x in enumerate(hist)]))
    print(f"Target: {tokenizer.decode([target])}, Distractor: {tokenizer.decode([distractor])}")
    print(model_forwarder.get_closest_embedding(model, tokenizer, prefix_embedding[0]))
    return prefix_embedding.detach() if succeeded else None

class ModelForwarder:

    def __init__(self):
        self.adaptor = None

    def forward(self, model, tokenizer, prompt, device, repeat, obj):
        decoder_input = torch.tensor([tokenizer.encode(prompt)], device=device)
        with torch.no_grad():
            assert self.adaptor is not None
            if self.adaptor is not None:
                output_dict = forward_with_prefix(model, tokenizer, prompt, device, None, self.adaptor, repeat)
            else:
                output_dict = model.forward(torch.cat([decoder_input] * 10, 0) if repeat else decoder_input,
                                        return_dict=True)
        return output_dict

    def clear_adaptor(self):
        self.adaptor = None


    def set_adaptor(self, adaptor):
        self.adaptor = adaptor

    def get_closest_embedding(self, model, tokenizer, embedding):
        # Feed model
        embed_module = model.model.embed_tokens
        sims = embed_module.weight @ embedding
        inds = torch.argsort(sims)
        return list(zip(tokenizer.convert_ids_to_tokens(sims[inds][-10:]), sims[inds][-10:].tolist()))

    def get_embedding(self, model, token_id, device):
        # Prepare inputs
        token_ids = torch.tensor([[token_id]], device=device)

        # Feed model
        embed_module = model.model.embed_tokens
        embedding = embed_module(token_ids)[0, 0, :]

        return embedding


class ModelForwarderTTS:
    def __init__(self, tempname):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained("openai/whisper-base.en")
        self.tempname = tempname

    def get_embedding(self, model, token_id, device):
        # Prepare inputs
        token_ids = torch.tensor([[token_id]], device=device)

        # Feed model
        embed_module = model.model.decoder.embed_tokens
        embedding = embed_module(token_ids)[0, 0, :]

        return embedding

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
        output_dict = model.forward(torch.cat([input_features] * 2, 0) if repeat else input_features,
                                    decoder_input_ids=torch.cat([decoder_input] * 2, 0) if repeat else decoder_input)
        return output_dict

def make_model(args, mock=False):
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


def make_model_whisper(args, mock=False):
    from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, BitsAndBytesConfig
    quantize = torch.cuda.is_available() and args.bfloat16
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                             bnb_4bit_compute_dtype=torch.bfloat16) if quantize else None
    model = mock_model() if mock else AutoModelForSpeechSeq2Seq.from_pretrained(args.model_name_path, token=args.token,
                                                                                force_download=False,
                                                                                quantization_config=quantization_config,
                                                                                device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_path, token=args.token, force_download=False,
                                              add_bos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer




def process_entry(causal_tracer: MaskedCausalTracer, prompt: str, subject: str, obj: str, target_token: str,
                  bucket: str, adaptor):
    output = dict()

    embedding_module_name = get_module_name(causal_tracer.model, "embed", 0)
    subject_tokens_range = find_substring_range(causal_tracer.tokenizer, prompt, subject)

    object_tokens_range, tokens = find_all_substring_range(causal_tracer.tokenizer, prompt, obj)
    output["object_tokens_range"] = object_tokens_range
    output["subject_tokens_range"] = subject_tokens_range
    if adaptor is None:
        return output

    # Get corrupted run results
    clean_probs, corrupted_probs = causal_tracer.trace_with_patch(
        prompt, subject_tokens_range, [target_token], [(None, [])], embedding_module_name, obj, adaptor
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
    if corrupted_output["probs"] == clean_output["probs"]:
        return output

    # Get patched runs results
    num_tokens = get_num_tokens(causal_tracer.tokenizer,
                                prompt)  # -1 should not be necesary nd it does not belong here for models other than Whisper
    output["results"]["tokens"] = list()
    # We start the loop from the first subject token as patching previous tokens has no effect
    inds = (get_quantiles(list(range(subject_tokens_range[0], subject_tokens_range[1])), 5, [subject_tokens_range[0]+1, subject_tokens_range[1]-2]) + get_quantiles(
            list(range(subject_tokens_range[1], num_tokens)), 5, [subject_tokens_range[1]+1, num_tokens-2]))
    print(inds, num_tokens)
    for token_i in inds:
        d = {}
        d["pos"] = token_i - num_tokens
        print(token_i, num_tokens)
        d["val"] = tokens[token_i]

        # If token is part of the subject, store its relative negative position
        if subject_tokens_range[0] <= token_i < subject_tokens_range[1]:
            d["subject_pos"] = token_i - subject_tokens_range[1]
        nl = get_num_layers(causal_tracer.model)
        params = [(kind, last_layer) for kind in ["hidden", "mlp", "attn"] for
                  last_layer in get_quantiles(np.arange(1, nl + 1))]
        patches = [(0, len(tokens))]
        params_all = [(kind, last_layer, patch) for kind, last_layer in params
                      for patch in patches]
        for kind, last_layer, patch in params_all:
            states_to_patch = (
                token_i,
                [
                    get_module_name(causal_tracer.model, kind, L)
                    for L in range(
                    0, last_layer)
                ],
            )
            _, patched_probs = causal_tracer.trace_with_patch(
                prompt, subject_tokens_range, [target_token], [states_to_patch], embedding_module_name, obj, adaptor
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

def run_causal_tracing_analysis(
        model: nn.Module,
        tokenizer,
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

    if not skip_creation or not os.path.exists(partial_path):
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
                adaptor = prefix_tuning(model, tokenizer, prompt, device, model_forwarder, [fact.get_parent().get_object(), fact.get_object()])
                if adaptor is None:
                  partial_dataset.add_entry(
                    {
                        "fact": fact.as_dict(),
                        "partial_results": {
                            "object": fact.get_object(),
                            "prompt": prompt,
                            "adaptor": None,
                        },
                    }
                  )
                  continue

                model_forwarder.set_adaptor(adaptor)
                most_likely_next_token, _ = get_next_token(model, tokenizer, prompt, device, model_forwarder, False,
                                                           fact.get_object())
                most_likely2 = most_likely_next_token[-2:]

                def faithfullness(id):
                    top1 = most_likely_next_token[-1] == target_tokens[id]
                    top2 = target_tokens[id] in most_likely2 and not target_tokens[not id] in most_likely2
                    #top = target_tokens[id] in most_likely_next_token and not target_tokens[not id] in most_likely_next_token
                    return top1 or top2

                unfaithful = faithfullness(0)
                grounded = faithfullness(1)
                partial_dataset.add_entry(
                    {
                        "fact": fact.as_dict(),
                        "partial_results": {
                            "prompt": prompt,
                            "next_token": target_tokens[0] if unfaithful else target_tokens[1] if grounded else
                            most_likely_next_token[-1],
                            "unfaithful_token": target_tokens[0],
                            "grounded_token": target_tokens[1],
                            "is_unfaithful": unfaithful,
                            "is_grounded": grounded,
                            "adaptor": adaptor.cpu().numpy().tolist(),
                        },
                    }
                )
                model_forwarder.clear_adaptor()

    partial_dataset = [x for x in read_json(partial_path)]

    unfaithful_facts = []
    grounded_facts = []

    for entry in partial_dataset:
        if "is_grounded" in entry["partial_results"] and entry["partial_results"]["is_grounded"]:
            grounded_facts.append(entry)
        elif "is_unfaithful" in entry["partial_results"] and entry["partial_results"]["is_unfaithful"]:
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
                                             bucket, torch.tensor(entry["partial_results"]["adaptor"], device=device))

                output_entry["fact"] = fact.as_dict()

                dataset.add_entry(output_entry)


def run_causal_tracing_analysis_wrapper(params):
    i, c, resume_dir, args = params
    fakepedia = read_json(args.fakepedia_path)[:args.subset_size]
    fakepedia = fakepedia[(i * len(fakepedia)) // c:((i + 1) * len(fakepedia)) // c]
    model, tokenizer = make_model(args)
    try:
        run_causal_tracing_analysis(
            model,
            tokenizer,
            fakepedia,
            args.prompt_template,
            args.num_grounded,
            args.num_unfaithful,
            args.prepend_space,
            resume_dir,
            ModelForwarderTTS(f"./temp/temp{i:02d}") if args.isTTS else ModelForwarder(),
            args.skip_creation
        )
    except KeyboardInterrupt as e:
        del model
        raise e


def run_causal_tracing(args):
    logger = get_logger()

    logger.info("Loading fakepedia...")
    # 23 kinds of relations 1673 unique templates.
    fakepedia = read_json(args.fakepedia_path)
    print(len(set([x["subject"] for x in fakepedia])),
          (len(set([x["rel_p_id"] for x in fakepedia])) if "rel_p_id" in fakepedia[0] else 0))

    logger.info("Starting causal tracing...")
    pool_size = 1
    params = [(i, pool_size, f"{args.resume_dir}/{i:02d}", args) for i in range(pool_size)]
    if pool_size > 1:
        with Pool(pool_size) as p:
            for x in params:
                os.makedirs(x[2], exist_ok=True)
            p.map(run_causal_tracing_analysis_wrapper, params)
    else:
        run_causal_tracing_analysis_wrapper(params[0])


def prefix_search(model):
    pre =""
    mid = ""
    after = ""


class Namespace1:
    def __init__(self):
        self.token = os.environ.get("HUGGINGFACE_TOKEN")
        self.fakepedia_path = "transl_fakepedia.json"
        ms = ["unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit", "unsloth/Llama-3.2-1B-unsloth-bnb-4bit",
              "meta-llama/Llama-3.2-1B", "google/gemma-3-1b-pt", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]
        self.model_name_path = ms[0]
        self.prompt_template = "{context}\n{query}"
        self.num_grounded = 40
        self.num_unfaithful = 40
        self.prepend_space = True
        self.bfloat16 = False
        self.resume_dir = "PrefixTuningTransl"
        self.subset_size = 100
        self.skip_creation = False
        self.isTTS = False


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = Namespace1()
    freeze_args(args)
    run_causal_tracing(args)