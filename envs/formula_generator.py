import random

# cписок операций: и, или, исключающее или, импликация, эквивалентность
OPS = ["and", "or", "xor", "imp", "eq"]


def random_atom(vars):
    v = random.choice(vars)
    return v if random.random() < 0.7 else f"(not {v})"


def generate_formula(vars, depth):
    if depth <= 1:
        return random_atom(vars)

    if random.random() < 0.2:
        return andom_atom(vars)

    left = generate_formula(vars, depth - 1)
    right = generate_formula(vars, depth - 1)
    op = random.choice(OPS)


    if op == "and": return f"({left} and {right})" # обычный and
    if op == "or":  return f"({left} or {right})" # обычный or
    if op == "xor": return f"({left} ^ {right})" # обычный xor
    if op == "imp": return f"((not {left}) or {right})"  # импликация: A -> B это (not A) or B
    if op == "eq":  return f"({left} == {right})"  # эквивалентность
    return f"({left} and {right})"


def eval_formula(expr, values: dict):
    # превращаем x, y, z в bool
    return bool(eval(expr, {"__builtins__": {}}, values))