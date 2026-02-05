import re


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:

    rewards = []

    for completion, gold in zip(completions, answer):
        # извлекаем текст из completion (может быть строкой или списком сообщений)
        if isinstance(completion, list):

            content = completion[0]["content"] if len(completion) > 0 else ""
        else:
            content = completion

        # ищем содержимое тега <answer>
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)

        if match is None:
            rewards.append(0.0)
            continue

        pred = match.group(1).strip().lower().replace(" ", "").replace("\n", "")
        gold_clean = str(gold).strip().lower().replace(" ", "").replace("\n", "")

        if pred == gold_clean:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


def format_reward_func(completions, **kwargs) -> list[float]:
    """
    награда за соблюдение формата <think>...</think> и <answer>...</answer>.
    """
    rewards = []

    for completion in completions:
        if isinstance(completion, list):
            content = completion[0]["content"] if len(completion) > 0 else ""
        else:
            content = completion
        has_think = "<think>" in content and "</think>" in content
        has_answer = "<answer>" in content and "</answer>" in content

        if has_think and has_answer:
            # проверяем что <think> идет перед <answer>
            if content.find("<think>") < content.find("<answer>"):
                rewards.append(0.2)
            else:
                rewards.append(0.1)
        else:
            rewards.append(0.0)

    return rewards