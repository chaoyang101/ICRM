# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser
from transformers import AutoTokenizer
import ast
import json

def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="google/gemma-2b-it")
    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)
    sampling_group.add_argument("--sampling-num", type=int)
    sampling_group.add_argument("--prompt-file", type=str)
    sampling_group.add_argument("--record-file", type=str)
    sampling_group.add_argument("--sampling-range", type=str)

    return parser


def main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    sampling_num = args.pop("sampling_num")
    prompt_file = args.pop("prompt_file")
    record_file = args.pop("record_file")
    sampling_range = args.pop("sampling_range")
    sampling_range = ast.literal_eval(sampling_range)

    # Create an LLM
    llm = LLM(**args)

    with open(prompt_file, "r", encoding="utf-8") as f:
        train_inputs = json.load(f)
        train_inputs = train_inputs[int(sampling_range[0]):int(sampling_range[1])]
    prompts = [
        {
            "role": "user",
            "content": prompt,
        } for prompt in train_inputs
    ]

    # Create a sampling params object
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k
    if sampling_num is not None:
        sampling_params.n = sampling_num
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    '''
    prompts = [
        {
            "role": "user",
            "content": "Hello, my name is",
        },
        {
            "role": "user",
            "content": "The president of the United States is",
        },
        {
            "role": "user",
            "content": "The capital of France is",
        },
        {
            "role": "user",
            "content": "The future of AI is",
        }
    ]
    '''
    tokenizer = AutoTokenizer.from_pretrained(args["model"])
    for i in range(len(prompts)):
        prompts[i] = tokenizer.apply_chat_template([prompts[i]], tokenize=False)
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("-" * 50)
    all_generated_outputs = []
    assert len(train_inputs) == len(outputs)
    for (i_input, output) in zip(train_inputs, outputs):
        prompt = output.prompt
        if i_input not in prompt:
            print(prompt)
            print(i_input)
        inner_output_list = []
        for inner_output in output.outputs:
            generated_text = inner_output.text
            inner_output_list.append(generated_text)
        sample = {
            "question":
            {
                "role": "user",
                "content": i_input
            },
            "p_question":
            {
                "role": "user",
                "content": prompt
            },
            "responses":[
                {
                    "role": "assistant",
                    "content": re
                } for re in inner_output_list
            ]
        }
        all_generated_outputs.append(sample)
        #generated_text = output.outputs[0].text
        #print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        #print("-" * 50)
    write_file_name = prompt_file.split('.')[0]
    with open(record_file, 'w') as json_file:
        json.dump(all_generated_outputs, json_file, indent=2)

if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)