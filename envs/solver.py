import itertools
from envs.formula_generator import eval_formula


class LogicSolver:
    @staticmethod
    def solve(formula, vars_list, fragment):
        # fragment приходит как список словарей [{'row': {}, 'f': 0}, ...]
        if not fragment: return []

        col_names = list(fragment[0]['row'].keys())
        valid_solutions = []

        for perm in itertools.permutations(col_names):
            mapping = dict(zip(vars_list, perm))
            is_correct = True

            for item in fragment:
                row_data = item['row']
                f_gold = item['f']

                var_values = {v: row_data[mapping[v]] for v in vars_list}

                if eval_formula(formula, var_values) != bool(f_gold):
                    is_correct = False
                    break

            if is_correct:
                ans_parts = [f"{v}={mapping[v]}" for v in sorted(vars_list)]
                valid_solutions.append(", ".join(ans_parts))


        return valid_solutions