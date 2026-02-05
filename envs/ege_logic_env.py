import random
from base.env import Env
from base.data import Data
from envs.difficulty import difficulty_to_params
from envs.formula_generator import generate_formula, eval_formula
from envs.truth_table import build_truth_table, depends_on_all_vars
from envs.fragment_builder import build_informative_fragment
from envs.prompt_builder import build_prompt
from verifiers.ege_logic_verifier import EGELogicVerifier


class EGELogicEnv(Env):

    def __init__(self):
        super().__init__("EGE_LOGIC", EGELogicVerifier)

    def extract_answer(self, text: str):
        return self.verifier.extract_answer(text)

    def generate(self, num_of_questions=10, difficulty=1, max_attempts=5000):

        dataset = []
        params = difficulty_to_params(difficulty)


        vars_needed = ["x", "y", "z", "w", "u"][:params["num_vars"]]

        attempts = 0
        print(f"generating {num_of_questions} tasks for difficulty {difficulty}")

        while len(dataset) < num_of_questions and attempts < max_attempts:
            attempts += 1

            formula = generate_formula(vars_needed, params["depth"])

            # построение полной таблицы истинности
            table = build_truth_table(formula, vars_needed, eval_formula)

            #отсекаем функции-константы где все 1 или все 0
            f_values = [row[1] for row in table]
            if len(set(f_values)) < 2:
                continue

            # проверяем что формула зависит от каждой заявленной переменной
            if not depends_on_all_vars(table, vars_needed):
                continue

            # поиск фрагмента таблицы истинности чтобы решениебыло одно
            fragment = build_informative_fragment(
                table,
                vars_needed,
                formula,
                eval_formula,
                params["num_rows"]
            )

            if fragment is None:
                continue

            # перемешивание столбцов
            perm = vars_needed[:]
            random.shuffle(perm)
            mapping = dict(zip(vars_needed, perm))

            # подготовка данных для промпта и для сохранения в метаданные
            shuffled_tuples = []
            fragment_to_save = []

            for row, f in fragment:
                # переименовываем столбцы согласно случайной перестановке
                new_row = {mapping[v]: row[v] for v in vars_needed}
                shuffled_tuples.append((new_row, f))
                # сохраняем в формате списка словарей
                fragment_to_save.append({"row": new_row, "f": f})

            # формирование эталонного ответа (например: x=z, y=w, z=y)
            ans_parts = [f"{v}={mapping[v]}" for v in sorted(vars_needed)]
            answer = ", ".join(ans_parts)

            # текст промпта
            prompt = build_prompt(formula, vars_needed, shuffled_tuples, perm)

            # сохранение задачи в датасет
            dataset.append(Data(
                question=prompt,
                answer=answer,
                difficulty=difficulty,
                metadata={
                    "formula": formula,
                    "vars": vars_needed,
                    "mapping": mapping,
                    "fragment": fragment_to_save  # фрагмент задачи для брутфорса и верификации
                }
            ))

            if len(dataset) % 10 == 0:
                print(f"   progress: {len(dataset)}/{num_of_questions}")

        print(f"ready: {len(dataset)}")
        return dataset