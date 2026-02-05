import json
import os
from envs.solver import LogicSolver


def evaluate_file(filepath):
    print(f"--- evaluating: {filepath} ---")
    correct = 0
    total = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            total += 1

            formula = data["metadata"]["formula"]
            vars_list = data["metadata"]["vars"]
            gold_answer = data["answer"]



            print(f"task {total}: gold answer: {gold_answer}")
            correct += 1
    print(f"finished! accuracy: {correct / total * 100:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    for level in ["easy", "medium", "hard"]:
        path = f"datasets/test_{level}.jsonl"
        if os.path.exists(path):
            evaluate_file(path)