import sys
import json
import os
import subprocess
import concurrent.futures

def process(input_file, output_file):
    if os.path.isfile(output_file):
        return 
    script_path = "./maldi_preprocess/end_to_end_preprocessing.R"
    result = subprocess.run(f"Rscript {script_path} {input_file} {output_file}",
                            stdout = subprocess.PIPE,
                            stderr = subprocess.STDOUT, shell = True,
                            universal_newlines = True # Python >= 3.7 also accepts "text=True"
                            ) 
    return "success!"

def main(args):
    # Use json.loads to parse the JSON string into a Python object
    print(args)
    parameters = json.loads(args[1])
    print(parameters)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = [executor.submit(process, input_file, output_file) for input_file, output_file in zip(parameters["input"], parameters["output"])]

if __name__ == "__main__":
    main(sys.argv)