import itertools

# подсчет сколько решений подойдет под наш фрагмент таблицы истинности
# целимся в 1 решение
def count_solutions(formula, vars_list, fragment, eval_fn):
    if not fragment: return 0

    col_names = list(fragment[0][0].keys())
    valid_count = 0

    for perm in itertools.permutations(col_names):
        mapping = dict(zip(vars_list, perm))
        is_correct = True
        for row, f_gold in fragment:
            var_values = {v: row[mapping[v]] for v in vars_list}
            if eval_fn(formula, var_values) != bool(f_gold):
                is_correct = False
                break
        if is_correct:
            valid_count += 1
    return valid_count


def build_informative_fragment(full_table, vars_list, formula, eval_fn, min_rows):
    import random
    rows = full_table[:]
    random.shuffle(rows)

    fragment = []


    for row, f in rows:
        fragment.append((row, f))

        if len(fragment) >= min_rows:
            # считаем количество решений для текущего набора строк
            solutions_count = count_solutions(formula, vars_list, fragment, eval_fn)

            if solutions_count == 1:
                return fragment  # одно решение - возвращаем фрагмента

            if solutions_count == 0:
                return None  # что-то пошло не так (математически невозможно)

    return None  # если перебрали все строки и решений всё равно > 1 (симметричная формула)