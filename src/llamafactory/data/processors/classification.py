import copy
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
import json
import ast
import torch

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from .processor_utils import infer_seqlen

logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import ImageInput, VideoInput
    from ..template import Template

json_file_path = './data/special tokens.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
_SPACE = data["input"]["space holder"]
_POS = data["output"]["postive"]
_NEG = data["output"]["negtive"]
_NAT = data["output"]["natural"]

def _encode_value_generation_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
    max_turn: int = 2
) -> Tuple[List[int], List[int], bool, bool]:
    if response[0]["content"]:  # desired example
        kto_tag = True
        messages = prompt + [response[0]]
    else:  # undesired example
        kto_tag = False
        messages = prompt + [response[1]]
    outlier_tag = response[-1]["step_val"]
    assert len(messages) // 2 == max_turn
    messages = template.mm_plugin.process_messages(messages, images, videos, processor)
    #prompt_ids, response_ids = template.encode_oneturn(tokenizer, messages, system, tools)
    input_ids, labels = template.mm_plugin.process_token_ids([], [], images, videos, tokenizer, processor)
    encoded_pairs = template.encode_multiturn(tokenizer, messages, system, tools)
    total_length = len(input_ids) + (1 if template.efficient_eos else 0)

    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        if total_length + len(source_ids)+len(target_ids) > cutoff_len:
            return None, None, None, None
        source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), cutoff_len - total_length)
        source_ids = source_ids[:source_len]
        target_ids = target_ids[:target_len]
        total_length += source_len + target_len
        if template.efficient_eos:
            source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len
        if turn_idx == max_turn - 1:
            target_label = target_ids[:-1] + [IGNORE_INDEX]
        else:
            target_label = target_ids

        input_ids += source_ids + target_ids
        labels += source_label + target_label

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    return input_ids, labels, kto_tag, outlier_tag

def _encode_generation_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int
) -> Tuple[List[int], List[int], bool, torch.Tensor]:
    if response[0]["content"]:  # desired example
        kto_tag = True
        messages = prompt + [response[0]]
    else:  # undesired example
        kto_tag = False
        messages = prompt + [response[1]]
    outlier_tag = response[-1]["step_val"]

    messages = template.mm_plugin.process_messages(messages, images, videos, processor)
    prompt_ids, response_ids = template.encode_oneturn(tokenizer, messages, system, tools)

    if template.efficient_eos:
        response_ids += [tokenizer.eos_token_id]

    prompt_ids, _ = template.mm_plugin.process_token_ids(prompt_ids, None, images, videos, tokenizer, processor)

    if len(response_ids)+len(prompt_ids) > cutoff_len:
        return None, None, None, None
    
    source_len, target_len = infer_seqlen(len(prompt_ids), len(response_ids), cutoff_len)
    prompt_ids = prompt_ids[:source_len]
    response_ids = response_ids[:target_len]

    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * source_len + response_ids
    #labels[-1] = IGNORE_INDEX
    try:
        outlier_tag = ast.literal_eval(outlier_tag)
    except:
        outlier_tag = [0]
    return input_ids, labels, kto_tag, torch.tensor(outlier_tag)

def _encode_multiclass_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
    stage: str = "orm"
) -> Tuple[List[int], List[int], bool]:
    if response[0]["content"]:  # desired example
        kto_tag = True
        messages = prompt + [response[0]]
    else:  # undesired example
        kto_tag = False
        messages = prompt + [response[1]]
    step_val = response[-1]["step_val"]

    messages = template.mm_plugin.process_messages(messages, images, videos, processor)

    goal_token = tokenizer.encode(f" {_SPACE}")[-1]
    pos_token = tokenizer.encode(f" {_POS}")[-1]
    neg_token = tokenizer.encode(f" {_NEG}")[-1]
    nat_token = tokenizer.encode(f" {_NAT}")[-1]
    #print(goal_token)
    #print(pos_token)
    #print(neg_token)
    #print(nat_token)
    assert pos_token != goal_token
    assert neg_token != pos_token
    assert nat_token != neg_token
    candidate_tokens_dict = {"postive": pos_token, "negtive": neg_token, "natural": nat_token}

    replacements_list = step_val.split(",")
    assert len(replacements_list) > 0

    prompt_ids, response_ids = template.encode_oneturn(tokenizer, messages, system, tools)

    if template.efficient_eos:
        response_ids += [tokenizer.eos_token_id]

    prompt_ids, _ = template.mm_plugin.process_token_ids(prompt_ids, None, images, videos, tokenizer, processor)

    if len(response_ids)+len(prompt_ids) > cutoff_len:
        return None, None, None
    
    source_len, target_len = infer_seqlen(len(prompt_ids), len(response_ids), cutoff_len)
    prompt_ids = prompt_ids[:source_len]
    response_ids = response_ids[:target_len]

    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * source_len + response_ids

    if stage == "prm":
        for i in range(len(labels)):
            if labels[i] == goal_token: #goal_token: space holder
                rep = replacements_list.pop(0) #replacements_list:label list for the space holders
                token = candidate_tokens_dict[rep] #candidate_tokens_dict: a dict to record encoded idx for each label
                labels[i] = token
            else:
                labels[i] = IGNORE_INDEX
        assert len(replacements_list) == 0
    elif stage == "orm":
        #The following form is equivalent to replacing the last [SPACE] with a specified category
        for i in range(len(labels)):
            labels[i] = IGNORE_INDEX
        assert len(replacements_list) == 1
        rep = replacements_list[-1]
        labels[-2] =  candidate_tokens_dict[rep]
    else:
        raise NotImplementedError
    
    return input_ids, labels, kto_tag

def process_multiclass_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
    stage: str = "orm"
) -> Dict[str, List[Any]]:
    # create unrelated input-output pairs for estimating the KL term by flipping the matched pairs
    model_inputs = defaultdict(list)
    if stage == "orm":
        raise NotImplementedError
        #encode_func = _encode_multiclass_example 
    elif stage == "orm_g":
        encode_func = _encode_generation_example
    elif stage == "orm_dg":
        encode_func = _encode_generation_example #_encode_value_generation_example
    else:
        raise NotImplemented
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
            logger.warning("Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i]))
            raise
            #continue

        input_ids, labels, kto_tag, outlier_tag = encode_func(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len
        )
        if input_ids is None:
            continue
        
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])
        model_inputs["kto_tag"].append(kto_tag)
        model_inputs["outlier_tag"].append(outlier_tag)
        model_inputs["idx"].append(i)

    return model_inputs

def preprocess_process_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    return process_multiclass_dataset(examples, template, tokenizer, processor, data_args, stage="prm")

def preprocess_answer_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    return process_multiclass_dataset(examples, template, tokenizer, processor, data_args, stage="orm")

def preprocess_answer_g_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    return process_multiclass_dataset(examples, template, tokenizer, processor, data_args, stage="orm_g")

def preprocess_answer_dg_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    return process_multiclass_dataset(examples, template, tokenizer, processor, data_args, stage="orm_dg")