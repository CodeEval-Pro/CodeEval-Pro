import json
import argparse
from pathlib import Path


def post_process_humaneval(output):
    if 'if __name__' in output:
        output = output.split('if __name__' ,1)[0]
    elif '# Test' in output:
        output = output.split('# Test' ,1)[0]
    elif '# Example' in output:
        output = output.split('# Example ' ,1)[0]
    else:
        pass
    return output


if __name__ == '__main__':
    # Create a command-line argument parser
    parser = argparse.ArgumentParser()
    
    # Add command-line arguments
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--source_path', type=str, required=True, help='Path to the log directory.')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    with open(Path(args.source_path)/'results.jsonl' ,'r') as f:
        ds = f.readlines()
    ds = [json.loads(d) for d in ds]
    
    (Path(args.source_path)/'santized_results.jsonl').write_text("")

    with open(Path(args.source_path)/'santized_results.jsonl', 'a+') as f:
        for d in ds:
            if 'completion' in d.keys():
                d['completion'] = post_process_humaneval(d['completion'])
            elif 'solution' in d.keys():
                d['solution'] = post_process_humaneval(d['solution'])
            else:
                pass
            print(json.dumps(d), file=f, flush=True)
