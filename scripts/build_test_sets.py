import os
import json
from envs.ege_logic_env import EGELogicEnv


def save_dataset(filename, data_list):
    if not data_list:
        print(f"no data to save for {filename}")
        return

    os.makedirs("datasets", exist_ok=True)
    path = os.path.join("datasets", filename)
    with open(path, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item.to_json(), ensure_ascii=False) + "\n")
    print(f"successfully saved to {path}")


def main():
    env = EGELogicEnv()

    # сначала easy, потом medium, потом hard
    configs = [
        {"name": "test_easy.jsonl", "difficulty": 1, "count": 30},
        {"name": "test_medium.jsonl", "difficulty": 4, "count": 30},
        {"name": "test_hard.jsonl", "difficulty": 8, "count": 30},
    ]

    for config in configs:
        dataset = env.generate(
            num_of_questions=config["count"],
            difficulty=config["difficulty"]
        )
        save_dataset(config["name"], dataset)


if __name__ == "__main__":
    main()