import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from datasets import load_dataset, Dataset
from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl


def get_mbpp_pro_raw_problems() -> list[dict]:
    problems = load_dataset('CodeEval-Pro/mbpp-pro',split='train')
    return list(problems)


def get_humaneval_pro_raw_problems() -> list[dict]:
    problems = load_dataset('CodeEval-Pro/humaneval-pro',split='train')
    return list(problems)

def map_swebench_problem(p: Dataset) -> Dict[str, Any]:
    id = p["instance_id"]
    prompt = p["text"]
    instruction = f"{prompt}"
    response_prefix = f"""<patch>
"""
    return dict(
        id=id, instruction=instruction, response_prefix=response_prefix
    )

def map_mbpp_pro_problem(p: dict) -> Dict[str, Any]:
    id = p["id"]
    prompt1 = p["raw_problem"].strip()
    prompt2 = p["new_problem"].strip()

    instruction = f"""Write a solution of python file to the following problems, the solution of the second problem requires single or multiple calls to the first solution.
```python
{prompt1}
{prompt2}
```"""
    response_prefix = f"""```python
{prompt1}
"""
    return dict(
        id=id, instruction=instruction, response_prefix=response_prefix
    )


def map_mbpp_pro_problem_cot(p: dict) -> Dict[str, Any]:
    id = p["id"]
    prompt1 = p["raw_problem"].strip()
    prompt2 = p["new_problem"].strip()

    instruction = f"""Write a solution of python file to the following problems, the solution of the second problem requires single or multiple calls to the first solution.
```python
{prompt1}
{prompt2}
```"""
    response_prefix = f"""Let's think step by step.
```python
{prompt1}
"""
    return dict(
        id=id, instruction=instruction, response_prefix=response_prefix
    )


def get_mbpp_raw_problems() -> list[dict]:
    problems = get_mbpp_plus()
    return list(problems.values())




def get_humaneval_raw_problems() -> list[dict]:
    problems = get_human_eval_plus()
    return list(problems.values())


def map_mbpp_problem(p: dict) -> Dict[str, Any]:
    id = p["task_id"]
    prompt = p["prompt"]
    start_index = prompt.index('"""')
    end_index = prompt.rindex('"""')
    prompt = prompt[start_index + 3 : end_index]
    assert_index = prompt.index("assert")
    instruction = prompt[:assert_index].strip()
    if not instruction.endswith("."):
        instruction += "."
    assertion = prompt[assert_index:].strip()
    instruction = f"""{instruction} Your code should satisfy the following assertion:
```python
{assertion}
```"""
    response_prefix = f"""```python"""
    return dict(
        id=id, instruction=instruction, response_prefix=response_prefix
    )


def map_humaneval_problem(p: dict) -> Dict[str, Any]:
    id = p["task_id"]
    prompt = p["prompt"]
    prompt = prompt.strip()
    # try:
    #     docstring_index = prompt.index('"""')
    # except ValueError:
    #     docstring_index = prompt.index("'''")
    # signature = prompt[:docstring_index].strip()
    # Instruction
    instruction = f"""Write a solution to the following problem:
```python
{prompt}
```"""
    response_prefix = f"""```python
{prompt}"""
    return dict(
        id=id, instruction=instruction, response_prefix=response_prefix
    )

def map_humaneval_pro_problem_cot(p: dict) -> Dict[str, Any]:
    id = p["id"]
    prompt1 = p["raw_problem"].strip()
    prompt2 = p["new_problem"].strip()

    instruction = f"""Write a solution of python file to the following problems, the solution of the second problem requires single or multiple calls to the first solution.
```python
{prompt1}
{prompt2}
```"""
    response_prefix = f"""Let's think step by step.
```python
{prompt1}
"""
    # response_prefix = ''
    return dict(
        id=id, instruction=instruction, response_prefix=response_prefix
    )


def map_humaneval_pro_problem(p: dict) -> Dict[str, Any]:
    id = p["id"]
    prompt1 = p["raw_problem"].strip()
    prompt2 = p["new_problem"].strip()

    instruction = f"""Write a solution of python file to the following problems, the solution of the second problem requires single or multiple calls to the first solution.
```python
{prompt1}
{prompt2}
```"""
    response_prefix = f"""```python
{prompt1}
"""
    # response_prefix = ''
    return dict(
        id=id, instruction=instruction, response_prefix=response_prefix
    )


def read_jsonl(path: str | Path):
    with Path(path).open("r") as f:
        return json.load(f)
        # return [json.loads(line) for line in f]

def map_humaneval_pro_problem_1shot(p: dict) -> Dict[str, Any]:
    id = p["id"]
    prompt1 = p["raw_problem"].strip()
    prompt2 = p["new_problem"].strip()

    example = p['test_code'].split('\n')[1]
    instruction = f"""Write a solution of python file to the following problems, the solution of the second problem requires single or multiple calls to the first solution.
```python
{prompt1}
{prompt2}

{example}
```"""
    response_prefix = f"""```python
{prompt1}
"""
    # response_prefix = ''
    return dict(
        id=id, instruction=instruction, response_prefix=response_prefix
    )

def map_mbpp_pro_problem_1shot(p: dict) -> Dict[str, Any]:
    id = p["id"]
    prompt1 = p["raw_problem"].strip()
    prompt2 = p["new_problem"].strip()
    example = p['test_code'].split('\n')[0]
    instruction = f"""Write a solution of python file to the following problems, the solution of the second problem requires single or multiple calls to the first solution.
```python
{prompt1}
{prompt2}

{example}
```"""
    response_prefix = f"""```python
{prompt1}
"""
    return dict(
        id=id, instruction=instruction, response_prefix=response_prefix
    )

def get_bigcodebench_lite_pro_problems() -> list[dict]:
    # problems = read_jsonl('dataset/refined_bigcodebench_lite_pro.json')
    problems = load_dataset('CodeEval-Pro/bigcodebench-lite-pro',split='train')
    return list(problems)

def map_bigcodebench_lite_pro_problem(p: dict) -> Dict[str, Any]:
    id = p["id"]
    prompt1 = p["raw_problem"].strip()
    prompt2 = p["new_problem"].strip()
    example = p['test_code'].split('\n')[0]
    instruction = f"""Write a solution of python file to the following problems, the solution of the second problem requires single or multiple calls to the first solution.
```python
{prompt1}
{prompt2}

```"""
    response_prefix = f"""```python
{prompt1}
"""
    return dict(
        id=id, instruction=instruction, response_prefix=response_prefix
    )


# def write_jsonl(path: str | Path, data: Sequence[Mapping]):
#     # cannot use `dict` here as it is invariant
#     with Path(path).open("w") as f:
#         json.dump(data, f, indent=4)
