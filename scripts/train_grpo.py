import json
import os
import sys
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from unsloth import FastLanguageModel


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from base.data import Data
from verifiers.ege_logic_verifier import EGELogicVerifier

# --- пути ---

BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "qwen_model")
FINETUNED_PATH = os.path.join(PROJECT_ROOT, "scripts", "models", "qwen-ege-logic-final")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "scripts", "datasets")

SYSTEM_PROMPT = """Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>"""

device = torch.device("mps")


def evaluate_dataset(file_name, model, tokenizer, verifier):
    path = os.path.join(DATASETS_DIR, file_name)

    if not os.path.exists(path):
        print(f"файл {file_name} не найден по пути: {path}")
        return None

    print(f"\n оценка датасета: {file_name}")
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

        # обновляем прогресс
        pbar.set_postfix({"acc": f"{(correct / (i + 1)) * 100:.1f}%"})

        # вывод первого ответа для визуального контроля
        if i == 0:
            tqdm.write(f"\n первый ответ для {file_name}:")
            tqdm.write(f"Response: {response[:200]}...")
            tqdm.write(f"Gold Answer: {data.answer}\n")

    accuracy = (correct / total) * 100
    return accuracy


def main():

    if not os.path.exists(BASE_MODEL_PATH) or not os.path.exists(FINETUNED_PATH):
        print(f"Base: {BASE_MODEL_PATH} ({os.path.exists(BASE_MODEL_PATH)})")
        print(f"Final: {FINETUNED_PATH} ({os.path.exists(FINETUNED_PATH)})")
        return

    print(f"загрузка оригинальной модели")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="mps",
        local_files_only=True
    )

    print(f"наложение обученного GRPO-адаптера")
    model = PeftModel.from_pretrained(
        base_model,
        FINETUNED_PATH,
        local_files_only=True
    ).to(device)

    model.eval()
    print("модель успешно готова к финальным тестам!")

    verifier = EGELogicVerifier()
    test_files = ["test_easy.jsonl", "test_medium.jsonl", "test_hard.jsonl"]
    final_results = {}

    for file in test_files:
        acc = evaluate_dataset(file, model, tokenizer, verifier)
        if acc is not None:
            final_results[file] = acc


    print("\n" + "=" * 45)
    print(f"{'DATASET (AFTER RL)':<25} | {'ACCURACY':<10}")
    print("-" * 45)
    for file, acc in final_results.items():
        print(f"{file:<25} | {acc:.2f}%")
    print("=" * 45)


if __name__ == "__main__":
    main()