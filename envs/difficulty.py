def difficulty_to_params(difficulty: int):
    num_vars = min(2 + difficulty // 3, 5)
    max_possible_rows = 2 ** num_vars
    requested_rows = 3 + difficulty

    return {
        "num_vars": num_vars,
        "depth": min(2 + difficulty // 2, 5),
        "num_rows": min(requested_rows, max_possible_rows - 1)  # оставляем хотя бы 1 строку скрытой
    }