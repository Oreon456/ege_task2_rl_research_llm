import json
import os
import sys
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# настройка путей, чтобы скрипт видел модули проекта
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from base.data import Data
from verifiers.ege_logic_verifier import EGELogicVerifier

# --- конфигурация ---
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "qwen_model")
SYSTEM_PROMPT = """Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>"""

# mps
device = torch.device("mps")


def evaluate_dataset(file_name, model, tokenizer, verifier):
    path_root = os.path.join(PROJECT_ROOT, "datasets", file_name)
    path_scripts = os.path.join(PROJECT_ROOT, "scripts", "datasets", file_name)

    if os.path.exists(path_root):
        path = path_root
    elif os.path.exists(path_scripts):
        path = path_scripts
    else:
        print(f"файл {file_name} не найден!")
        return None

    print(f"\nоценка датасета: {path}")
    correct = 0

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    pbar = tqdm(lines, desc=f"Level: {file_name}")

    for i, line in enumerate(pbar):
        data_dict = json.loads(line)
        data = Data(**data_dict)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": data.question},
        ]

        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer([input_text], return_tensors="pt").to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        if verifier.verify(data, response):
            correct += 1

        pbar.set_postfix({"acc": f"{(correct / (i + 1)) * 100:.1f}%"})

        if i == 0:
            tqdm.write(f"\npreview for {file_name}:")
            tqdm.write(f"response: {response[:150]}...")
            tqdm.write(f"gold: {data.answer}\n")

    accuracy = (correct / total) * 100
    return accuracy


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ошибка")
        return

    # загрузка токенизатора и модели
    print(f"загрузка модели")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="mps",
        local_files_only=True
    )
    model.eval()
    print("модель успешно загружена!")

    verifier = EGELogicVerifier()

    test_files = ["test_easy.jsonl", "test_medium.jsonl", "test_hard.jsonl"]
    final_stats = {}

    for file in test_files:
        acc = evaluate_dataset(file, model, tokenizer, verifier)
        if acc is not None:
            final_stats[file] = acc

    # итоговый отчет
    print("\n" + "=" * 40)
    print(f"{'dataset':<20} | {'accuracy':<10}")
    print("-" * 40)
    for file, acc in final_stats.items():
        print(f"{file:<20} | {acc:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()