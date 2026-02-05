from envs.ege_logic_env import EGELogicEnv

env = EGELogicEnv()

data = env.generate(5, difficulty=8)

for d in data:
    print("=" * 60)
    print(d.question)
    print("answer:", d.answer)
