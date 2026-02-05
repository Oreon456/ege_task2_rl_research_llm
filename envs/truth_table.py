import itertools

def build_truth_table(formula, vars, eval_fn):
    table = []
    for bits in itertools.product([0, 1], repeat=len(vars)):
        row = dict(zip(vars, bits))
        res = eval_fn(formula, row)
        table.append((row, int(res)))
    return table


def depends_on_all_vars(table, vars):
    base = [f for _, f in table]
    for v in vars:
        changed = False
        for i, (row, _) in enumerate(table):
            flipped = row.copy()
            flipped[v] ^= 1
            for j, (row2, f2) in enumerate(table):
                if all(flipped[k] == row2[k] for k in vars):
                    if f2 != base[i]:
                        changed = True
                        break
            if changed:
                break
        if not changed:
            return False
    return True
