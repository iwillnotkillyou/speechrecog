import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, StaticCache, AutoModelForSeq2SeqLM, \
    T5ForConditionalGeneration, T5Tokenizer

from causal_tracing_whisper import *
from transformers import LlamaForCausalLM, LlamaTokenizer, StaticCache, AutoModelForSeq2SeqLM, \
    T5ForConditionalGeneration, T5Tokenizer

from causal_tracing_whisper import *


def make_peft_model(model, modules):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(init_lora_weights="pissa", r=1, lora_alpha=16, target_modules=modules,
                        task_type="SEQ_2_SEQ_LM")
    model = get_peft_model(model, config)
    return model


def from_lora(model, tokenizer: LlamaTokenizer, device, path, encoder_input_text):
    model.model = make_peft_model(model.model, model.modules)
    model.model.load_adapter(path, "default")
    encoder_ids = tokenizer.encode(encoder_input_text)
    encoder_output = model.model.encoder(
        input_ids=torch.tensor([encoder_ids], device=device)
    )
    model.model = model.model.unload()
    return encoder_output


def from_lora_decoder(model, encoder_outputs, decoder_input, path, repeat):
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    output_dict = model.forward(encoder_outputs=(encoder_outputs,),
                                      decoder_input_ids=torch.cat([decoder_input] * 2, 0) if repeat else decoder_input,
                                      return_dict=True)
    #model.model = model.model.unload()
    return output_dict


def from_prefix(model, tokenizer: LlamaTokenizer, device, prefix_embedding, encoder_input_text):
    def hook_embedding(module, input, output):
        output[:, :prefix_embedding.shape[0]] = prefix_embedding
        return output

    embedding_module_name = get_module_name(model, "embed", 0)
    embedding_module = find_submodule(model, embedding_module_name)
    embedding_hook = embedding_module.register_forward_hook(hook_embedding)
    encoder_ids = tokenizer.encode(encoder_input_text)
    encoder_output = model.model.encoder(
        input_ids=torch.tensor([encoder_ids[:1] + [0] * (
            prefix_embedding.shape[0] - 1 if prefix_embedding.shape[0] - 1 > 0 else 0) + encoder_ids[1:]],
                               device=device)
    )
    embedding_hook.remove()
    return encoder_output


def forward_with_encoder_prefix(model, tokenizer: LlamaTokenizer, prompt, device, target, prefix_embedding, repeat,
                                encoder_input_text):
    encoder_output = from_prefix(model, tokenizer, device, prefix_embedding, encoder_input_text).last_hidden_state

    decoder_input = torch.tensor(
        [[model.model.config.decoder_start_token_id] + tokenizer.encode(prompt, add_special_tokens=False)],
        device=device)

    decoder_input_f = torch.cat([decoder_input] * 2, 0) if repeat else decoder_input
    labels = torch.cat(
        [torch.full_like(decoder_input, -100)[:, 1:], torch.tensor([[target]], device=device)],
        -1) if target is not None and not repeat else None
    if encoder_output is None:
        output_dict = model.model.forward(decoder_input_f, labels=labels, return_dict=True)
    else:
        output_dict = model.model.forward(
            encoder_outputs=(encoder_output,),
            decoder_input_ids=decoder_input_f,
            labels=labels,
            return_dict=True)

    return output_dict


def forward_with_prefix(model, tokenizer: LlamaTokenizer, prompt, device, target, prefix_embedding, repeat,
                        encoder_output):
    def hook_embedding(module, input, output):
        output[:, :prefix_embedding.shape[0]] = prefix_embedding
        return output

    embedding_module_name = get_module_name(model, "embed", 0)
    embedding_module = find_submodule(model, embedding_module_name)
    embedding_hook = embedding_module.register_forward_hook(hook_embedding)
    decoder_input = torch.tensor([[model.model.config.decoder_start_token_id] + (
            [0] * (prefix_embedding.shape[0] - 1 if prefix_embedding.shape[0] - 1 > 0 else 0)) + tokenizer.encode(
        prompt, add_special_tokens=False)],
                                 device=device)

    decoder_input_f = torch.cat([decoder_input] * 2, 0) if repeat else decoder_input
    labels = torch.cat(
        [torch.full_like(decoder_input, -100)[:, 1:], torch.tensor([[target]], device=device)],
        -1) if target is not None and not repeat else None
    if encoder_output is None:
        output_dict = model.model.forward(decoder_input_f, labels=labels, return_dict=True)
    else:
        output_dict = model.model.forward(
            encoder_outputs=(encoder_output,),
            decoder_input_ids=decoder_input_f,
            labels=labels,
            return_dict=True)

    embedding_hook.remove()
    return output_dict


def fine_tuning(model: LlamaForCausalLM, tokenizer: LlamaTokenizer, prompt, device, model_forwarder, objects,
                greater_sep=False, max_steps=2, encoder_input_text=None):
    model.model = make_peft_model(model.model, model.modules)
    objects = np.array(objects)[[0, 1]] if False else np.random.permutation(objects)
    nan_inds = [i for i in range(len(objects)) if objects[i] is None]
    if len(nan_inds) == 2:
        raise Exception("At least one of the objects must not be nan.")
    if len(nan_inds) == 1:
        objects[1], objects[nan_inds[0]] = objects[nan_inds[0]], objects[1]
    print(objects)
    target, distractor = [x if x is None else tokenizer.convert_tokens_to_ids([x])[0] for x in objects]
    hist = []
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-3 * 5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=max_steps)
    for num_steps in range(max_steps):
        with torch.enable_grad():
            decoder_input_text = prompt
            encoder_outputs = model.model.encoder(
                input_ids=torch.tensor([tokenizer.encode(encoder_input_text)], device=device)
            ).last_hidden_state
            decoder_input = torch.tensor([[model.model.config.decoder_start_token_id] + tokenizer.encode(
                decoder_input_text, add_special_tokens=False)], device=device)
            train_all = True
            labels = torch.cat(
                [torch.full_like(decoder_input, -100)[:, 1:], torch.tensor([[target]], device=device)],
                -1) if not train_all else torch.tensor([[model.model.config.decoder_start_token_id] + tokenizer.encode(
                decoder_input_text, add_special_tokens=False) + [target]], device=device)
            output_dict = model.model.forward(encoder_outputs=(encoder_outputs,),
                                              decoder_input_ids=None if train_all else decoder_input, labels=labels,
                                              return_dict=True)
        next_token_logits = output_dict["logits"].detach()[0, -1, :].cpu()
        next_token_probs = torch.softmax(next_token_logits, dim=-1).numpy()
        max_prob_indices = np.argsort(next_token_probs)
        max_prob_indices = max_prob_indices[np.searchsorted(np.flip(np.cumsum(np.flip(max_prob_indices))), 0.9):]
        succeeded = max_prob_indices[-1] == target
        if False:
            succeeded |= max_prob_indices[-2] == target and max_prob_indices[-1] != distractor and (
                    max_prob_indices[-3] != distractor or not greater_sep)
        hist.append(list(zip(next_token_probs[max_prob_indices][-10:].tolist(),
                             tokenizer.convert_ids_to_tokens(max_prob_indices)[-10:]))[-3:])
        print((num_steps, hist[-1]))
        if succeeded:
            break
        loss = output_dict["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # hist = list(enumerate(hist))
    # print("\n".join([str(x) for x in hist]))
    # print("\n".join([str(x) for x in hist[:2]+hist[-2:]]))
    print(
        f"Success:{succeeded}, Target: {tokenizer.decode([target])}, Distractor: {distractor if distractor is None else tokenizer.decode([distractor])}")
    # print(model_forwarder.get_closest_embedding(model, tokenizer, prefix_embedding[0]))

    model_forwarder.num_finetunes += 1
    os.makedirs("adapters", exist_ok=True)
    path = f"adapters/{model_forwarder.num_finetunes}"
    model.model = model.model.merge_and_unload()
    model.save(path)
    model.reset()
    return path if succeeded and num_steps > 0 else None, num_steps if succeeded else max_steps


def prefix_tuning(model: LlamaForCausalLM, tokenizer: LlamaTokenizer, prompt, device, model_forwarder, objects,
                  greater_sep=False, max_steps=2, len_prefix=20, encoder_output=None, encoder_input_text=None):
    print(prompt)
    mask_token_id = tokenizer.eos_token_id
    for x in model.model.parameters():
        x.requires_grad = False
    objects = np.array(objects)[[0, 1]] if False else np.random.permutation(objects)
    nan_inds = [i for i in range(len(objects)) if objects[i] is None]
    if len(nan_inds) == 2:
        raise Exception("At least one of the objects must not be nan.")
    if len(nan_inds) == 1:
        objects[1], objects[nan_inds[0]] = objects[nan_inds[0]], objects[1]
    print(objects)
    target, distractor = [x if x is None else tokenizer.convert_tokens_to_ids([x])[0] for x in objects]
    prefix_embedding = model_forwarder.get_embedding(model, mask_token_id, device).clone()
    prefix_embedding = torch.stack([prefix_embedding] * len_prefix,
                                   0) if len_prefix > 0 else prefix_embedding.unsqueeze(0)[:0]
    prefix_embedding.requires_grad = True
    hist = []
    succeeded = False
    num_steps = 0
    for num_steps in range(max_steps):
        xfrac = num_steps / max_steps
        t = True
        with torch.enable_grad():
            if encoder_input_text is None:
                output_dict = forward_with_prefix(model, tokenizer, prompt, device, target if t else distractor,
                                                  prefix_embedding, False, encoder_output)
            else:
                output_dict = forward_with_encoder_prefix(model, tokenizer, prompt, device, target if t else distractor,
                                                          prefix_embedding, False, encoder_input_text)
        next_token_logits = output_dict["logits"].detach()[0, -1, :].cpu()
        next_token_probs = torch.softmax(next_token_logits, dim=-1).numpy()
        max_prob_indices = np.argsort(next_token_probs)
        max_prob_indices = max_prob_indices[np.searchsorted(np.flip(np.cumsum(np.flip(max_prob_indices))), 0.9):]
        succeeded = max_prob_indices[-1] == target or (
                max_prob_indices[-2] == target and max_prob_indices[-1] != distractor and (
                max_prob_indices[-3] != distractor or not greater_sep))
        if succeeded:
            break
        hist.append(list(zip(next_token_probs[max_prob_indices][-10:].tolist(),
                             tokenizer.convert_ids_to_tokens(max_prob_indices)[-10:]))[-3:])
        loss = output_dict["loss"]
        loss.backward()
        learning_rate0 = 0.1
        learning_rate1 = 0.01
        prefix_embedding.data = prefix_embedding.data + (- 1 if t else 1) * (
                learning_rate1 * xfrac + (1 - xfrac) * (learning_rate0)) * prefix_embedding.grad.data
        prefix_embedding.grad.zero_()
    hist = list(enumerate(hist))
    print("\n".join([str(x) for x in hist[:2] + hist[-2:]]))
    print(
        f"Target: {tokenizer.decode([target])}, Distractor: {distractor if distractor is None else tokenizer.decode([distractor])}")
    # print(model_forwarder.get_closest_embedding(model, tokenizer, prefix_embedding[0]))
    return prefix_embedding.detach() if succeeded and num_steps > 0 else None, num_steps if succeeded else max_steps


class attn_hook_wrapper:
    def __init__(self, layer, cache):
        self.layer = layer
        self.cache = cache

    def attn_hook(self, module, input, output):
        output[:, :self.cache[self.layer].shape[1], :] = self.cache[self.layer]
        return output


def forward_with_cache(model, decoder_input, repeat, cache):
    hooks = []

    for n, m in model.model.named_modules():
        if n.endswith("self_attn"):
            l = int(n.split(".")[-2])
            hooks.append(m.k_proj.register_forward_hook(attn_hook_wrapper(l, cache[0]).attn_hook))
            hooks.append(m.v_proj.register_forward_hook(attn_hook_wrapper(l, cache[1]).attn_hook))

    output_dict = model.model.forward(torch.cat([decoder_input] * 2, 0) if repeat else decoder_input,
                                      return_dict=True)
    for hook in hooks:
        hook.remove()
    return output_dict


def forward_with_huggingface_cache(model, decoder_input, repeat, cache, max_cache_len):
    past_key_values = copy.deepcopy(cache)
    assert decoder_input.shape[1] + 5 < max_cache_len, f"Input too long {decoder_input.shape[1]}+5 >= {max_cache_len}"
    output_dict = model.model.forward((torch.cat([decoder_input] * 2, 0) if repeat else decoder_input),
                                      return_dict=True, past_key_values=past_key_values)
    return output_dict


class ModelForwarder:

    def __init__(self, use_cache=True):
        self.adaptor = None
        self.cache = None
        self.cached_input = None
        self.max_cache_len = 0

    def make_cache(self, model, tokenizer, prompt, length, device, repeat, obj):
        decoder_input_full = torch.tensor([tokenizer.encode(prompt)], device=device)
        decoder_input = decoder_input_full[:length]
        with torch.no_grad():
            self.max_cache_len = int(decoder_input_full.shape[1] * 2)
            cache = StaticCache(config=model.model.config, max_batch_size=2, max_cache_len=self.max_cache_len,
                                device="cuda",
                                dtype=torch.half)
            output_dict = model.model.forward(torch.cat([decoder_input] * 2, 0) if repeat else decoder_input,
                                              return_dict=True, past_key_values=cache)
            self.cache = output_dict["past_key_values"]
            self.cached_input = decoder_input

    def forward(self, model, tokenizer, prompt, device, repeat, obj):
        decoder_input = torch.tensor([tokenizer.encode(prompt)], device=device)
        if self.cache is not None:
            if not torch.equal(self.cached_input[:, :decoder_input.shape[1]], decoder_input):
                print(tokenizer.decode(self.cached_input[0])[:40], tokenizer.decode(decoder_input[0])[:40])
                self.cache = None
                del self.cached_input

        with torch.no_grad():
            if self.cache is not None:
                output_dict = forward_with_huggingface_cache(model, decoder_input, repeat, self.cache,
                                                             self.max_cache_len)
            else:
                if self.adaptor is not None:
                    output_dict = forward_with_prefix(model, tokenizer, prompt, device, None, self.adaptor, repeat,
                                                      None)
                else:
                    output_dict = model.model.forward(torch.cat([decoder_input] * 2, 0) if repeat else decoder_input,
                                                      return_dict=True, past_key_values=None)
        return output_dict

    def clear_adaptor(self):
        self.adaptor = None

    def set_adaptor(self, adaptor):
        self.adaptor = adaptor

    def get_closest_embedding(self, model, tokenizer, embedding):
        # Feed model
        embedding_module_name = get_module_name(model.model, "embed", 0)
        embed_module = find_submodule(model, embedding_module_name)
        sims = embed_module.weight @ embedding
        inds = torch.argsort(sims)
        return list(zip(tokenizer.convert_ids_to_tokens(sims[inds][-10:]), sims[inds][-10:].tolist()))

    def get_embedding(self, model, token_id, device):
        # Prepare inputs
        token_ids = torch.tensor([[token_id]], device=device)

        embedding_module_name = get_module_name(model.model, "embed", 0)
        embed_module = find_submodule(model.model, embedding_module_name)
        embedding = embed_module(token_ids)[0, 0, :]

        return embedding

    def get_encoder_outputs(self, model, tokenizer, prompt, device):
        return None


class ModelForwarderEncDec:
    def __init__(self):
        self.cached_encoder_input = (None, None)
        self.adaptor = None
        self.cache = None
        self.num_finetunes = 0

    def get_embedding(self, model, token_id, device):
        # Prepare inputs
        token_ids = torch.tensor([[token_id]], device=device)

        embedding_module_name = get_module_name(model.model, "embed", 0)
        embed_module = find_submodule(model.model, embedding_module_name)
        embedding = embed_module(token_ids)[0, 0, :]

        return embedding

    def forward(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer, prompt, device, repeat, obj):
        encoder_input_text, decoder_input_text = prompt.split("<pad>")
        with torch.no_grad():
            encoder_outputs = self.get_encoder_outputs(model, tokenizer, prompt, device)
            decoder_input = torch.tensor([[model.model.config.decoder_start_token_id] + tokenizer.encode(
                decoder_input_text, add_special_tokens=False)], device=device)
            print(repeat, decoder_input.size())
            encoder_outputs = torch.cat([encoder_outputs] * 2, 0) if repeat else encoder_outputs
            print(encoder_outputs.size())

            if self.adaptor is not None:
                #model.model = make_peft_model(model.model, model.modules)
                #model.model.load_adapter(self.adaptor, "default")
                model.load(self.adaptor)
            output_dict = model.model.forward(encoder_outputs=(encoder_outputs,),
                                              decoder_input_ids=torch.cat([decoder_input] * 2,
                                                                          0) if repeat else decoder_input,
                                              return_dict=True)
            if self.adaptor is not None:
                #model.model = model.model.unload()
                model.reset()
        return output_dict

    def clear_adaptor(self):
        self.adaptor = None

    def make_cache(self, model, tokenizer, prompt, length, device, repeat, obj):
        pass

    def edit_prompt(self, prompt):
        return prompt.split("<pad>")

    def set_adaptor(self, adaptor):
        self.adaptor = adaptor

    def get_encoder_outputs(self, model, tokenizer, prompt, device):
        encoder_input_text, decoder_input_text = prompt.split("<pad>")

        with torch.no_grad():
            if self.cached_encoder_input[0] == encoder_input_text:
                encoder_outputs = self.cached_encoder_input[1]
            else:
                encoder_outputs = model.model.encoder(
                    input_ids=torch.tensor([tokenizer.encode(encoder_input_text)], device=device)
                ).last_hidden_state
                self.cached_encoder_input = (encoder_input_text, encoder_outputs)
        return encoder_outputs


class ModelForwarderTTS:
    def make_cache(self, model, tokenizer, prompt, length, device, repeat, obj):
        pass

    def __init__(self, tempname):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained("openai/whisper-base.en")
        self.tempname = tempname
        self.adaptor = None
        self.cache = None

    def get_embedding(self, model, token_id, device):
        # Prepare inputs
        token_ids = torch.tensor([[token_id]], device=device)

        embedding_module_name = get_module_name(model, "embed", 0)
        embed_module = find_submodule(model, embedding_module_name)
        embedding = embed_module(token_ids)[0, 0, :]

        return embedding

    def forward(self, model, tokenizer, prompt, device, repeat, obj):
        samplep = run_festival(prompt, self.tempname)
        samples = run_festival(obj, self.tempname)
        sample_rate = samplep[1]
        sample = np.concatenate([samplep[0], samples[0] + np.random.normal(0, np.full_like(samples[0], 0.2))], 0)
        input_features = self.processor([sample], sampling_rate=sample_rate, return_tensors="pt",
                                        pad_to_multiple_of=8).input_features[
                         -model.model.config.max_source_positions + 5:].to(device).to(model.model.dtype)
        decoder_input = torch.tensor([tokenizer.encode(prompt)[:-1]], device=device)[
                        -model.model.config.max_target_positions + 5:]
        output_dict = model.model.forward(torch.cat([input_features] * 2, 0) if repeat else input_features,
                                          decoder_input_ids=torch.cat([decoder_input] * 2,
                                                                      0) if repeat else decoder_input)
        return output_dict

    def clear_adaptor(self):
        self.adaptor = None

    def set_adaptor(self, adaptor):
        self.adaptor = adaptor


def make_model(args, mock=False):
    from transformers import AutoTokenizer
    quantize = torch.cuda.is_available() and args.bfloat16
    max_memory_mapping = {0: "8GB", "cpu": "0GB"} if torch.cuda.is_available() else None
    if quantize:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                                 torch_dtype=torch.bfloat16)
        model = mock_model() if mock else AutoModelForSeq2SeqLM.from_pretrained(args.model_name_path, token=args.token,
                                                                                force_download=False,
                                                                                quantization_config=quantization_config,
                                                                                device_map="auto",
                                                                                max_memory=max_memory_mapping)
    else:
        model = mock_model() if mock else AutoModelForSeq2SeqLM.from_pretrained(args.model_name_path, token=args.token,
                                                                                force_download=False,
                                                                                device_map="auto",
                                                                                max_memory=max_memory_mapping)
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
    model.model.config.pad_token_id = model.model.config.eos_token_id
    return model, tokenizer


def process_entry(causal_tracer: MaskedCausalTracer, prompt: str, subject: str, obj: str, target_token: str,
                  bucket: str, adaptor, args, pbar):
    output = dict()

    embedding_module_name = get_module_name(causal_tracer.model.model, "embed", 0)
    subject_tokens_range = find_substring_range(causal_tracer.tokenizer, prompt, subject)

    string_ids = causal_tracer.tokenizer(
        prompt,
        return_tensors=None,
        return_token_type_ids=False,
    )["input_ids"]
    tokens = causal_tracer.tokenizer.convert_ids_to_tokens(string_ids)
    if args.output_object_tokens_range:
        object_tokens_range, _ = find_all_substring_range(causal_tracer.tokenizer, prompt.replace("<pad>", " "), obj)
        output["object_tokens_range"] = object_tokens_range
    output["subject_tokens_range"] = subject_tokens_range

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
                                prompt.replace("\\\\n",
                                               " "))  # -1 should not be necessary and it does not belong here for models other than Whisper
    output["results"]["tokens"] = list()
    # We start the loop from the first subject token as patching previous tokens has no effect
    inds = (get_quantiles(list(range(subject_tokens_range[0], subject_tokens_range[1])), 5,
                          [subject_tokens_range[0] + 1, subject_tokens_range[1] - 2]) + get_quantiles(
        list(range(subject_tokens_range[1], num_tokens)), 5, [subject_tokens_range[1] + 1, num_tokens - 2]))
    print(f"\n{inds}, {num_tokens}")
    for token_i in inds:
        d = {}
        d["pos"] = token_i - num_tokens
        pbar.set_description(f"{token_i}, {num_tokens}")
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
                    get_module_name(causal_tracer.model.model, kind, L)
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

    device = next(model.model.parameters()).device
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

                target_tokens = adapt_target_tokens(
                    tokenizer, [fact.get_parent().get_object(), fact.get_object()], prepend_space
                )

                def run(finetuning_target_toks):

                    # Predict most likely next token
                    prompt = construct_prompt(fact, prompt_template)
                    adaptor, num_steps = None, 0

                    subject_tokens_range = find_substring_range(tokenizer, prompt, fact.get_subject())
                    assert subject_tokens_range is not None, f"Subject {fact.get_subject()} not found in prompt: {prompt}, {fact.get_subject()}"
                    if args.max_steps > 0:
                        adaptor, num_steps = fine_tuning(model, tokenizer,
                                                         model_forwarder.edit_prompt(prompt)[1] if hasattr(
                                                             model_forwarder, "edit_prompt") else prompt, device,
                                                         model_forwarder, finetuning_target_toks,
                                                         max_steps=args.max_steps,
                                                         encoder_input_text=model_forwarder.edit_prompt(prompt)[
                                                             0] if hasattr(model_forwarder, "edit_prompt") else None)
                    skip_no_adaptor = False
                    if skip_no_adaptor and adaptor is None:
                        partial_dataset.add_entry(
                            {
                                "fact": fact.as_dict(),
                                "partial_results": {
                                    "object": fact.get_object(),
                                    "prompt": prompt,
                                    "adaptor": None,
                                    "num_steps": num_steps,
                                    "group_id": fact.group_id
                                },
                            }
                        )
                        return

                    model_forwarder.set_adaptor(adaptor)
                    most_likely_next_token, _ = get_next_token(model, tokenizer, prompt, device, model_forwarder, False,
                                                               fact.get_object())
                    print("most_likely_next_token", most_likely_next_token[-1])
                    while most_likely_next_token[-1].startswith("Ġ_"):
                        most_likely_next_token = most_likely_next_token[:-1]

                    def faithfullness(id, most_likely_next_token):
                        top1 = most_likely_next_token[-1] == target_tokens[id]
                        top2 = most_likely_next_token[-2] == target_tokens[id] and not most_likely_next_token[-1] == \
                                                                                       target_tokens[1 - id]
                        # top = target_tokens[id] in most_likely_next_token and not target_tokens[not id] in most_likely_next_token
                        return top1 or top2

                    unfaithful = faithfullness(0, most_likely_next_token)
                    grounded = faithfullness(1, most_likely_next_token)
                    partial_dataset.add_entry(
                        {
                            "fact": fact.as_dict(),
                            "partial_results": {
                                "prompt": prompt,
                                "is_in_next_tokens": target_tokens[1] in most_likely_next_token[-10:],
                                "next_token": target_tokens[0] if unfaithful else target_tokens[1] if grounded else
                                most_likely_next_token[-1],
                                "unfaithful_token": target_tokens[0],
                                "grounded_token": target_tokens[1],
                                "is_unfaithful": unfaithful or (args.not_grounded_is_hallucinated and not grounded),
                                "is_grounded": grounded,
                                "adaptor": adaptor if adaptor is None or isinstance(adaptor,
                                                                                    str) else adaptor.cpu().numpy().tolist(),
                                "num_steps": num_steps,
                                "group_id": fact.group_id
                            },
                        }
                    )
                    model_forwarder.clear_adaptor()

                run(target_tokens)
                run(list(reversed(target_tokens)))

    partial_dataset = [x for x in read_json(partial_path)]

    unfaithful_facts = []
    grounded_facts = []

    for entry in partial_dataset:
        if "is_grounded" in entry["partial_results"] and entry["partial_results"]["is_grounded"]:
            grounded_facts.append(entry)
        elif "is_unfaithful" in entry["partial_results"] and entry["partial_results"]["is_unfaithful"]:
            unfaithful_facts.append(entry)

    logger.info(f"Found {len(unfaithful_facts)} unfaithful facts and {len(grounded_facts)} grounded facts")

    causal_tracer = MaskedCausalTracer(model, tokenizer, "eos", model_forwarder, args.use_cache)

    for bucket in ["grounded", "unfaithful"] if args.grounded_first else ["unfaithful", "grounded"]:
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
            for entry in (pbar := tqdm(facts, desc=f"Running causal tracing on {bucket} facts")):

                fact = fact_from_dict(entry["fact"])
                if dataset.is_input_processed(fact):
                    continue

                prompt = entry["partial_results"]["prompt"]
                target_token = entry["partial_results"]["next_token"]
                adaptor = entry["partial_results"]["adaptor"]
                output_entry = process_entry(causal_tracer, prompt, fact.get_subject(), fact.get_object(), target_token,
                                             bucket,
                                             adaptor if adaptor is None or isinstance(adaptor, str) else torch.tensor(
                                                 adaptor, device=device), args,
                                             pbar)

                output_entry["fact"] = fact.as_dict()

                dataset.add_entry(output_entry)


class ModelWrapper():
    def __init__(self, model: nn.Module):
        self.model = model
        self.modules = [x for x, y in self.model.named_modules() if
               "decoder" in x and "SelfAttention.k" in x and int(x.split(".")[2]) % 3 == 2]
        self.original_modules = dict()
        for x in self.modules:
            self.original_modules[x] = {k: torch.clone(v).detach() for k,v in find_submodule(self.model, x).state_dict().items()}

    def save(self, path):
        torch.save({x: find_submodule(self.model, x).state_dict() for x in self.modules}, path)

    def load(self, path):
        state_dicts = torch.load(path)
        for x in self.modules:
            find_submodule(self.model, x).load_state_dict(state_dicts[x])

    def reset(self):
        for x in self.modules:
            find_submodule(self.model, x).load_state_dict(self.original_modules[x])





def run_causal_tracing_analysis_wrapper(params):
    i, c, resume_dir, args = params
    fakepedia = read_json(args.fakepedia_path)
    print("total", len(fakepedia))
    fakepedia = fakepedia[:args.subset_size]
    fakepedia = fakepedia[(i * len(fakepedia)) // c:((i + 1) * len(fakepedia)) // c]
    model, tokenizer = make_model(args)
    model = ModelWrapper(model)
    if args.forwarder_kind == "encdec":
        model_forwarder = ModelForwarderEncDec()
    elif args.isTTS:
        model_forwarder = ModelForwarderTTS(f"./temp/temp{i:02d}")
    else:
        model_forwarder = ModelForwarder()
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
            model_forwarder,
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
    pre = ""
    mid = ""
    after = ""


class Namespace1:
    def __init__(self):
        self.token = os.environ.get("HUGGINGFACE_TOKEN")
        self.fakepedia_path = "fakepedia_arc_hard_with_ir_dev.json"
        ms = ["allenai/unifiedqa-t5-base", "google/gemma-3-1b-pt", "unsloth/Llama-3.2-1B-unsloth-bnb-4bit",
              "unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit",
              "unsloth/Llama-3.2-1B-unsloth-bnb-4bit",
              "meta-llama/Llama-3.2-1B", "google/gemma-3-1b-pt", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]
        self.model_name_path = ms[0]
        self.prompt_template = "{context}<pad>{query}"
        self.num_grounded = 1000
        self.num_unfaithful = 1000
        self.prepend_space = True
        self.bfloat16 = False
        self.resume_dir = "ARC_hard"
        self.subset_size = 10
        self.skip_creation = False
        self.forwarder_kind = "encdec"
        self.grounded_first = False
        self.output_object_tokens_range = False
        self.use_cache = False
        self.not_grounded_is_hallucinated = False
        self.max_steps = 0


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = Namespace1()
    freeze_args(args)
    run_causal_tracing(args)
