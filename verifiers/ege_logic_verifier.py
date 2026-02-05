import re
from base.verifier import Verifier

class EGELogicVerifier(Verifier):

    def extract_answer(self, text: str) -> str:
        if "<answer>" in text:
            text = text.split("<answer>")[1].split("</answer>")[0]

        text = text.lower().replace(" ", "")
        pairs = re.findall(r"([a-z])=([a-z])", text)
        pairs = sorted(pairs)
        return ";".join(f"{a}={b}" for a, b in pairs)

    def verify(self, data, test_answer: str) -> bool:
        pred = self.extract_answer(test_answer)
        gold = self.extract_answer(data.answer)
        return pred == gold
