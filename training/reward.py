import re


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    rewards = []
    for completion, gold in zip(completions, answer):
        # извлекаем текст внутри тегов <answer>...</answer>
        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        if match is None:
            rewards.append(0.0)  # модель не соблюла формат - награда 0
            continue

        pred = match.group(1).strip()

        # награда 1.0 за верный ответ, 0.0 за неверный
        if pred == gold.strip():
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


def format_reward_func(completions, **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        # проверяем наличие обоих блоков
        if "<think>" in completion and "</think>" in completion and \
                "<answer>" in completion and "</answer>" in completion:
            rewards.append(0.2)  # маленький бонус за дисциплину
        else:
            rewards.append(0.0)
    return rewards