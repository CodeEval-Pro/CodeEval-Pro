import json
import os
import re  
import subprocess  
import argparse

from pathlib import Path
from typing import List
from collections import defaultdict

from evaluate import load


code_metric = load("code_eval")
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def read_data(path):
    with open(path, 'r') as f:
        ds = f.readlines()
    data = [json.loads(d) for d in ds]
    return data


def generate_py_file(references, generated_code, save_path='./code_run'):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, reference in enumerate(references):
        code_candidate = generated_code[i]
        if not os.path.exists(f"{save_path}/case_{i}"):
            os.makedirs(f"{save_path}/case_{i}")
        for j, code in enumerate(code_candidate):
            with open(f"{save_path}/case_{i}/gen_{j}.py", 'w') as f:
                f.write(code.replace('\t','    '))
                f.write('\n' + reference)


def run_generated_py_file(references, generated_code, scripts_folder="./code_run/"):
    res = []
    statistic = {}
    error_type_dict = defaultdict(int)

    generate_py_file(references, generated_code, scripts_folder)

    file_dirs = os.listdir(scripts_folder)
    def extract_numbers(s):
        return int(''.join(re.findall(r'\d+', s)))

    for id,file_dir in enumerate(sorted(file_dirs,key = extract_numbers)):
        python_files = [f for f in os.listdir(f"{scripts_folder}{file_dir}") if f.endswith(".py")]

        for sid,file in enumerate(python_files): 
            status_result = []
            file_path = f"{scripts_folder}{file_dir}/{file}"
            print(f"file_path={file_path}")
            try:  
                subprocess.run(["python", file_path], check=True, stderr=subprocess.PIPE, universal_newlines=True, timeout = 30)
                status = 'Passed'
                error_type = 'Passed'
            except subprocess.CalledProcessError as e:  
                status = 'Failed'
                if 'AssertionError' in e.stderr:
                    error_type = 'AssertionError'
                else:
                    error_type = e.stderr.split('\n')[-2].split('Error: ')[0]+'Error'
            except subprocess.TimeoutExpired as e1:
                status = 'Failed'
                error_type = 'Timeout'
            finally:
                result = dict(
                    problem_id=id,
                    solution_id=sid,
                    status=status,
                    error_type=error_type
                )
                status_result.append(result)
                error_type_dict[error_type] += 1

        res.append(status_result)

    statistic['error_stats'] = error_type_dict
    statistic['analysis'] = res
    
    with open(Path(scripts_folder)/'basic_statistics.json', 'w') as f:
        json.dump(statistic, f, indent=4)

    return statistic

def evaluation(reference_path: str, gen_code_path: str):
    references = reference_path
    generated_code = gen_code_path
    results, _ = code_metric.compute(
            references=references,
            predictions=generated_code,
            k=[1, 5, 10],
            num_workers=8
        )

    return results


def main():
    """
    Main function to handle command-line arguments and run the evaluation process.
    """
    # Create a command-line argument parser
    parser = argparse.ArgumentParser(description='Evaluate generated code against reference test cases.')
    
    # Add command-line arguments
    parser.add_argument('--model_name', type=str, required=True, help='The model name')
    parser.add_argument('--task', type=str, required=True, help='The task name')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the problems JSON file.')
    parser.add_argument('--source_path', type=str, required=True, help='Path to the generated code JSONL file.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to the log directory.')
    parser.add_argument('--run_code',  action='store_true', help='Path to the save directory.')

    args = parser.parse_args()
    
    # Read the problems data
    problems = json.load(open((args.dataset_path)))

    # Read the generated code
    gt_code = [[d['raw_problem']+ d['raw_solution'] + d['new_problem'] + d['new_solution']] for id, d in enumerate(problems)]

    gen_code_file =read_data(Path(args.source_path)/'results.jsonl')
    santized_gen_code_file =read_data(Path(args.source_path)/'santized_results.jsonl')     

    if 'completion' in  gen_code_file[0].keys():
        if isinstance(gen_code_file[0]['completion'], str):
            gen_code_file = [{'task_id': d['task_id'], 'completion':[d['completion']]} for d in gen_code_file]
            santized_gen_code_file = [{'task_id': d['task_id'], 'completion':[d['completion']]} for d in santized_gen_code_file]
        gen_code = [
                        [problems[id]['raw_problem'] + completion for completion in d['completion']] 
                        for id, d in enumerate(gen_code_file)
                    ]
        santized_gen_code = [
                        [problems[id]['raw_problem'] + completion for completion in d['completion']] 
                        for id, d in enumerate(santized_gen_code_file)
                    ]

    elif 'solution' in  gen_code_file[0].keys():
        if isinstance(gen_code_file[0]['solution'], str):
            gen_code_file = [{'task_id': d['task_id'], 'solution':[d['solution']]} for d in gen_code_file]
            santized_gen_code_file = [{'task_id': d['task_id'], 'solution':[d['solution']]} for d in santized_gen_code_file]
        gen_code = [
                        [solution for solution in d['solution']] 
                        for id, d in enumerate(gen_code_file)
                    ]
        santized_gen_code = [
                        [solution for solution in d['solution']] 
                        for id, d in enumerate(santized_gen_code_file)
                    ]
    else:
        raise ValueError("Please check the result.jsonl file.")
    
    # Get the reference test codes
    reference = [d['test_code'] for d in problems] 
    


    if args.task == 'bigcodebench_lite_pro':
        gt_statistic = run_generated_py_file(reference, gt_code, args.save_path+'/log/gt_results/')
        gt_score = gt_statistic['error_stats']['Passed'] / len(gt_code)
        print(f"Result of Ground Truth : {gt_score}")

        statistic = run_generated_py_file(reference, gen_code, args.save_path+'/log/results/')
        gen_code_score = statistic['error_stats']['Passed'] / len(gt_code)
        print(f"Result of Your Outputs : {gen_code_score}")

        santized_statistic = run_generated_py_file(reference, santized_gen_code, args.save_path+'/log/santized_results/')
        santized_gen_code_score = santized_statistic['error_stats']['Passed'] / len(gt_code)
        print(f"Result of Your Santized Outputs : {santized_gen_code_score}")

    else:
        # Run the generated Python files and log the results
        if args.run_code:
            if (Path(args.save_path)/'log').exists():
                raise ValueError('Log file has exisied.')
            else:
                _ = run_generated_py_file(reference, gen_code, args.save_path+'/log/results/')
                _ = run_generated_py_file(reference, santized_gen_code, args.save_path+'/log/santized_results/')
        gt_score = evaluation(reference, gt_code)
        print(f"Result of Ground Truth : {gt_score}")
        
        gen_code_score = evaluation(reference, gen_code)
        print(f"Result of Your Outputs : {gen_code_score}")

        santized_gen_code_score = evaluation(reference, santized_gen_code)
        print(f"Result of Your Santized Outputs : {santized_gen_code_score}")


    # Print the evaluation results
    result = dict(
        pass_k_of_gt = gt_score,
        results = dict(
            model = args.model_name,
            pass_k_of_output = gen_code_score,
            pass_k_of_output_santized = santized_gen_code_score,
        )
    )

    with open(Path(args.save_path)/'result_of_pass_k.json', 'w') as f:
        json.dump(result, f, indent=4)

    
if __name__ == '__main__':
    main()