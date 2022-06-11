import os
import random
import re
from typing import List
import subprocess
from multiprocessing import Pool, current_process

"""
A utility for running multiple experiments on multiple GPUs
"""

if __name__ == '__main__':
    print(os.getcwd())
    PER_MATCH_RUN: int = 1
    EXPMT_DIR: str = "./runs"
    matches: List[str] = [f for f in os.listdir(EXPMT_DIR) if re.match(r'bert_substitute_.*\.json', f)] * PER_MATCH_RUN
    matches = [m for m in matches if m not in ('',)]  # leaving empty
    random.shuffle(matches)
    GPU_COUNT: int = 4
    SCRIPT: str = "train.py"
    def worker(config_file: str):
        # working with GPU 0 modify as needed
        cuda_id: int = (int(current_process().name[-1]) % GPU_COUNT) + 1
        process = subprocess.Popen(['./venv/bin/python',
                                    SCRIPT,
                                    f"{EXPMT_DIR}/{config_file}"],
                                   env={"CUDA_VISIBLE_DEVICES": str(cuda_id)})
        process.wait()
    with Pool(GPU_COUNT) as p:
        p.map(worker, matches)


