import itertools

def count_valid_mappings(fragment, vars):
    count = 0
    solutions = []

    for perm in itertools.permutations(vars):
        ok = True
        mapping = dict(zip(vars, perm))

        for row, _ in fragment:
            restored = {v: row[mapping[v]] for v in vars}
            if restored != restored:
                ok = False
                break

        if ok:
            count += 1
            solutions.append(mapping)

    return count, solutions
