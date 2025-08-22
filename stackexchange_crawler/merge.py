"""
Basic script to merge questions from different Stack Exchange sites after running `dataset_crawler.py`.
"""
import os
import glob
import json

output_dir = "outputs"
output_path = f"{output_dir}/dataset_candidates.jsonl"
input_pattern = f"{output_dir}/__crawler_*_questions.jsonl"

if __name__ == "__main__":
    print(f"Starting to merge dataset files matching pattern: {input_pattern}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be written to: {output_path}")
    with open(output_path, "w") as outfile:
        file_count = 0
        line_count = 0
        for filename in glob.glob(input_pattern):
            print(f"Processing file: {filename}")
            file_count += 1
            with open(filename, "r") as infile:
                for line in infile:
                    json_obj = json.loads(line)
                    outfile.write(json.dumps(json_obj) + "\n")
                    line_count += 1
        print(f"Merge complete. Processed {file_count} files with {line_count} total entries.")
