import json
import os
import sys

# путь к корню проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.solver import LogicSolver


def verify_file(filename):
    path = os.path.join("datasets", filename)
    if not os.path.exists(path):
        print(f"{filename} не найден.")
        return

    print(f"\n проверка: {filename}")
    total = 0
    passed = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            data = json.loads(line)

            # проверка наличия метаданных
            meta = data.get("metadata", {})
            if "fragment" not in meta or "formula" not in meta:
                print(f"нет нужных ключей в metadata")
                continue

            formula = meta["formula"]
            vars_list = meta["vars"]
            gold_answer = data["answer"]
            fragment = meta["fragment"]

            solutions = LogicSolver.solve(formula, vars_list, fragment)

            if len(solutions) == 1 and solutions[0] == gold_answer:
                passed += 1
            else:
                print("ошибка")

    print(f"итог {filename}: {passed}/{total}")


if __name__ == "__main__":
    for level in ["easy", "medium", "hard"]:
        verify_file(f"test_{level}.jsonl")