import torch
import json
import os
import sys
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROJECT_ROOT = "/ege_task2_rl"
BASE_MODEL_PATH = "/ege_task2_rl/models/qwen_model"
FINETUNED_PATH = "/ege_task2_rl/scripts/models/qwen-ege-logic-final"
DATASET_PATH = "/ege_task2_rl/scripts/datasets/test_medium.jsonl"

device = torch.device("mps")


def get_response(model, tokenizer, prompt):
    messages = [
        {"role": "system",
         "content": "Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)


def main():
    print("загрузка базовой модели (Qwen 1.5B)")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="mps"
    )

    # берем одну задачу из датасета
    with open(DATASET_PATH, "r") as f:
        sample_task = json.loads(f.readlines()[2])

    print("\n" + "=" * 60)
    print("задача из датасета:")
    print(sample_task['question'])
    print(f'правильный ответ: {sample_task['answer']}')
    print("=" * 60)


    print("\n[1/2] запрос к оригинальной модели...")
    base_response = get_response(base_model, tokenizer, sample_task['question'])

    print("\n--- ответ модели без дообучения ---")
    print(base_response)

    # накладываем GRPO адаптер
    print("\n" + "-" * 30)
    print("🧠 Накладываю RL-адаптер (LoRA)...")
    rl_model = PeftModel.from_pretrained(base_model, FINETUNED_PATH).to(device)
    print("-" * 30)


    print("\n[2/2] запрос к дообученной модели (GRPO)...")
    rl_response = get_response(rl_model, tokenizer, sample_task['question'])

    print("\n--- ответ дообученной модели ---")
    print(rl_response)



if __name__ == "__main__":
    main()