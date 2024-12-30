import json
import time
from pathlib import Path
from openai import OpenAI
from dataclasses import dataclass
from tqdm.auto import tqdm
from prompt_template import PROMPT_WRAPPER_API
from typing import Any, Literal, cast
from prompt_template import PROMPT_WRAPPER
from eval.utils import (
    read_jsonl, 
    write_jsonl, 
    get_humaneval_raw_problems,
    get_mbpp_raw_problems,
    get_mbpp_pro_raw_problems, 
    get_humaneval_pro_raw_problems,
    map_humaneval_problem,
    map_mbpp_problem,
    map_humaneval_pro_problem,
    map_mbpp_pro_problem,
    map_humaneval_pro_problem_cot,
    map_mbpp_pro_problem_cot,
    map_humaneval_pro_problem_1shot,
    map_mbpp_pro_problem_1shot,
    get_bigcodebench_lite_pro_problems,
    map_bigcodebench_lite_pro_problem
)

from transformers import (
    HfArgumentParser
)


DATASET_MAPPING={
    "humaneval": (get_humaneval_raw_problems, map_humaneval_problem),
    "mbpp": (get_mbpp_raw_problems, map_mbpp_problem),
    "humaneval_pro":(get_humaneval_pro_raw_problems, map_humaneval_pro_problem),
    "mbpp_pro": (get_mbpp_pro_raw_problems, map_mbpp_pro_problem),
    "humaneval_pro_cot":(get_humaneval_pro_raw_problems, map_humaneval_pro_problem_cot),
    "mbpp_pro_cot": (get_mbpp_pro_raw_problems, map_mbpp_pro_problem_cot),
    "humaneval_pro_1shot":(get_humaneval_pro_raw_problems, map_humaneval_pro_problem_1shot),
    "mbpp_pro_1shot": (get_mbpp_pro_raw_problems, map_mbpp_pro_problem_1shot),
    "bigcodebench_lite_pro": (get_bigcodebench_lite_pro_problems, map_bigcodebench_lite_pro_problem),
}


def make_request(prompt, model, api_key, base_url):
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    print(response)
    return response.choices[0].message.content


@dataclass(frozen=True)
class Args:
    dataset: Literal["humaneval", "mbpp", "humaneval_pro", "mbpp_pro", "humaneval_pro_cot", "mbpp_pro_cot", "humaneval_pro_1shot", "mbpp_pro_1shot", "bigcodebench_lite_pro"]
    save_path: str
    api_key: str
    base_url: str 
    model_name: str 


def main():
    parser = HfArgumentParser(Args)
    args = cast(
        Args,
        parser.parse_args_into_dataclasses(),
    )[0]
    raw_problem_fn, map_problem_fn = DATASET_MAPPING[args.dataset]
              
    raw_problems = raw_problem_fn()
    problems = list(map(map_problem_fn, raw_problems))

    
    if Path(args.save_path).exists():
        with open(args.save_path) as f:
            pre_record = f.readlines()
        samples = [json.loads(d) for d in pre_record]
        start_id = len(pre_record)
    else:
        Path(args.save_path).write_text("")
        samples = []
        start_id = 0

    for problem in tqdm(problems[start_id:]):
        task_id = problem["id"]
        prompt = PROMPT_WRAPPER_API.format(
                instruction=problem["instruction"], response=problem["response_prefix"]
            )
        print("PROMPT")
        print(prompt)
        completion=make_request(prompt, args.model_name, args.api_key, args.base_url).split('```python', 1)[-1]
        print("COMPLETION")
        print(completion)

        samples += [
            dict(
                task_id=task_id,
                solution=completion[
                    : index
                    if (index := completion.find("```")) != -1
                    else len(completion)
                ],
            )
        ]
        # print(samples)
        write_jsonl(args.save_path, samples)


if __name__ == "__main__":
    main()