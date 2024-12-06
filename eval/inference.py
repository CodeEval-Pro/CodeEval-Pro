import torch
import itertools
from pathlib import Path
from dataclasses import dataclass

from vllm import LLM, SamplingParams
from tqdm.auto import tqdm
from typing import Any, Literal, Dict, Mapping, Iterable, Sequence, TypeVar, cast
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import (
    HfArgumentParser,
    GenerationConfig,
    AutoModelForCausalLM, 
    AutoTokenizer
)

from eval.prompt_template import PROMPT_WRAPPER
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
)


_T = TypeVar("_T")
CUDA_NUM=torch.cuda.device_count()
EOS=[
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "<|EOT|>",
    # "\nif __name__",
    # "\ndef main(",
    # "\nprint(",
]

DATASET_MAPPING={
    "humaneval": (get_humaneval_raw_problems, map_humaneval_problem),
    "mbpp": (get_mbpp_raw_problems, map_mbpp_problem),
    "humaneval_pro":(get_humaneval_pro_raw_problems, map_humaneval_pro_problem),
    "mbpp_pro": (get_mbpp_pro_raw_problems, map_mbpp_pro_problem),
    "humaneval_pro_cot":(get_humaneval_pro_raw_problems, map_humaneval_pro_problem_cot),
    "mbpp_pro_cot": (get_mbpp_pro_raw_problems, map_mbpp_pro_problem_cot),
    "humaneval_pro_1shot":(get_humaneval_pro_raw_problems, map_humaneval_pro_problem_1shot),
    "mbpp_pro_1shot": (get_mbpp_pro_raw_problems, map_mbpp_pro_problem_1shot),
}


@dataclass(frozen=True)
class Args:
    dataset: Literal["humaneval", "mbpp", "humaneval_pro", "mbpp_pro", "humaneval_pro_cot", "mbpp_pro_cot", "humaneval_pro_1shot", "mbpp_pro_1shot"]
    save_path: str

    n_batches: int
    n_problems_per_batch: int
    n_samples_per_problem: int

    max_new_tokens: int
    top_p: float
    max_new_tokens: int
    num_return_sequences: int
    temperature: float
    do_sample: bool

    model_name_or_path: str | None = None
    use_flash_attention: bool = False
    is_use_vllm: bool = True


@dataclass(frozen=True)
class ModelContext:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer = None
    
    def complete(self, config: SamplingParams|GenerationConfig, prompts: list[str]) -> Dict:
        if self.tokenizer and isinstance(config, GenerationConfig):
            self.tokenizer.pad_token = self.tokenizer.eos_token
            input_ids = self.tokenizer(prompts, return_tensors="pt", padding=True)
            input_len = input_ids['input_ids'].shape[-1]
            input_ids = input_ids.to(self.model.device)
            output_ids = self.model.generate(**input_ids, generation_config=config)
            output_ids = output_ids[:, input_len:]
            output_strings = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        elif isinstance(config, SamplingParams):
            vllm_outputs = self.model.generate(prompts, config)
            output_strings =  [
                                [t.text for t in output.outputs]
                                for output in vllm_outputs
                               ]
            # input_ids = [output.outputs[0].prompt_token_ids for output in vllm_outputs]
            # output_ids = [output.outputs[0].token_ids for output in vllm_outputs]
        else:
            raise ValueError("Please check the generation configuration.")
        return dict(
            # raw_inputs=input_ids,
            # raw_outputs=output_ids,
            decoded_outputs=output_strings,
        )
    

def chunked(seq: Sequence[_T], n: int) -> Iterable[Sequence[_T]]:
    """Yield successive n-sized chunks from seq."""
    return (seq[i : i + n] for i in range(0, len(seq), n))


def main():
    parser = HfArgumentParser(Args)
    args = cast(
        Args,
        parser.parse_args_into_dataclasses(),
    )[0]

    raw_problem_fn, map_problem_fn = DATASET_MAPPING[args.dataset]
              
    raw_problems = raw_problem_fn()
    problems = list(map(map_problem_fn, raw_problems))

    other_kwargs: dict = {}
    other_kwargs["device_map"] = "auto"
    if args.use_flash_attention:
        other_kwargs["use_flash_attention_2"] = True
    if args.is_use_vllm:
        generation_config = SamplingParams(
            temperature=args.temperature, 
            top_p=args.top_p if args.do_sample else 1.0,
            n=args.num_return_sequences,
            max_tokens=args.max_new_tokens,
            stop=EOS
            )
        model = LLM(
            model=args.model_name_or_path, 
            max_model_len=8192, 
            download_dir='/data/zhuotaodeng/pretrained_models/',
            tensor_parallel_size=CUDA_NUM,
            trust_remote_code=True,
            gpu_memory_utilization=0.9
            )   
        tokenizer = None
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, 
            cache_dir='/data/zhuotaodeng/pretrained_models/', 
            trust_remote_code=True,
            **other_kwargs
            )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, 
            use_fast=True, 
            trust_remote_code=True,
            padding_side='left'
            )
        generation_config = GenerationConfig(
            do_sample=args.do_sample,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_return_sequences,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            )
        
    state = ModelContext(model, tokenizer)
    problems_chunked = list(chunked(list(problems), args.n_problems_per_batch))
    iter = itertools.product(problems_chunked, range(args.n_batches))
    n_total = len(problems_chunked) * args.n_batches

    samples = []
    Path(args.save_path).write_text("")
    for problems, batch_idx in tqdm(iter, total=n_total):
        task_ids = [problem["id"] for problem in problems]
        prompts = [
            PROMPT_WRAPPER.format(
                instruction=problem["instruction"], response=problem["response_prefix"]
            )
            for problem in problems
        ]
        
        print("PROMPT")
        print(prompts[-1])
        # all_prompts = prompts * args.n_samples_per_problem
        all_task_ids = task_ids * args.n_samples_per_problem
        response = state.complete(generation_config, prompts)
        completions = response['decoded_outputs']
        assert len(problems) <= args.n_problems_per_batch
        assert len(completions) == len(problems) * args.n_samples_per_problem
        print("COMPLETION")
        print(completions[-1])
        samples += [
            dict(
                task_id=task_id,
                completion=[
                        completion[
                        : index
                        if (index := completion.find("```")) != -1
                        else len(completion)
                                    ] 
                for completion in completion_batch
                ],
            )
            for task_id, completion_batch in zip(all_task_ids, completions)
        ]
        # print(samples)
        write_jsonl(args.save_path, samples)


if __name__ == "__main__":
    main()