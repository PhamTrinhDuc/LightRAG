import copy
from functools import lru_cache
from typing import List


from transformers import AutoTokenizer, AutoModelForCausalLM
from lightrag.base import BaseKVStorage
from lightrag.utils import compute_args_has

@lru_cache(maxsize=1)
async def hf_model_if_cache(
    model,
    prompt,
    system_prompt: str = None,
    history_message: List = [],
    **kwargs
) -> str:
    
    hf_tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model,
        device_map = "auto",
        trust_remote_code = True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model,
        device_map = "auto",
        trust_remote_code = True
    )

    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    messages =[]
    messages.append({"role": "system",
                     "content": system_prompt})
    messages.extend(history_message)
    messages.append({"role": "user",
                     "content": prompt})
    
    hashing_kv: BaseKVStorage = kwargs.get("hashing_kv", None)
    if hashing_kv is not None:
        id_hash = compute_args_has(model, messages)
        if_cache_return = hashing_kv.get_by_id(id_hash)
        if if_cache_return is not None:
            return if_cache_return['response']
    
    input_prompt = ""
    try: 
        input_prompt = hf_tokenizer.apply_chat_template(
            conversation=messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except Exception:
        try:
            ori_message = copy.deepcopy(messages)
            if messages[0]['role'] == 'system':
                messages[1]['content'] = (
                    "<system>" + messages[0]['content'] + 
                    "</system>\n" + messages[1]['content']
                )
                messages = messages[1:]
                input_prompt = hf_tokenizer.apply_chat_template(
                    conversation=messages,
                    tokenize=False, 
                    add_generation_prompt=True
                )
        except Exception:
            len_message = len(ori_message)
            for mgid in range(len_message):
                input_prompt = (
                    input_prompt 
                    + "<" + ori_message[mgid]['role'] + ">" 
                    + "<" + ori_message[mgid]['content'] + ">\n"
                )
    
    input_ids = hf_tokenizer(
        text=input_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to("cuda")
    inputs = {k: v.to(hf_model.device) for k, v in input_ids.items()}
    output = hf_model.generate(**inputs, max_new_tokens=512, num_return_sequences=1, early_stopping=True)

    response_text = hf_tokenizer.decode(
        output[0][len(inputs['input_ids'][0]): ], skip_special_tokens=True
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {id_hash: {"response": response_text, "model": model}}
        )
    return response_text

