from fire import Fire
from typing import List
from pydantic import BaseModel
import evaluate
from sklearn.metrics import accuracy_score, f1_score

from data_loading import ArgumentData

class AMScorer(BaseModel):
    @staticmethod
    def run(predictions: List[str], targets: List[str]) -> float:
        accuracy = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average="weighted")
        return dict(accuracy=accuracy, f1=f1)
    
class AGScorer(BaseModel):
    @staticmethod
    def run(predictions: List[str], references: List[str]) -> dict:
        # bert score
        bertscore = evaluate.load("bertscore")
        results = bertscore.compute(predictions=predictions, references=references, lang="en")
        bertscore = sum(results["f1"])/len(results["f1"])

        # rouge score
        rouge = evaluate.load('rouge')
        results = rouge.compute(predictions=predictions,references=references)
        rouge1 = results["rouge1"]
        rouge2 = results["rouge2"]
        rougeL = results["rougeL"]

        # meteor
        meteor = evaluate.load('meteor')
        results = meteor.compute(predictions=predictions, references=references)
        meteor = results["meteor"]

        return dict(bertscore=bertscore, rouge1=rouge1, rouge2=rouge2, rougeL=rougeL, meteor=meteor)


def select_scorer(task: str):
    if task in ["claim_detection", "evidence_detection", "stance_detection", "evidence_classification"]:
        return AMScorer()
    elif task in ["counter_arg_gen", "conclugen", "debatesum"]:
        return AGScorer()
    else:
        raise NotImplementedError
    

def test_scorer(task: str, output_path: str):
    data = ArgumentData.load_outputs(output_path)
    predictions = [sample.pred for sample in data.samples]
    targets = [sample.tgt for sample in data.samples]

    scorer = select_scorer(task)
    scores = scorer.run(predictions, targets)
    print(scores)


if __name__ == "__main__":
    Fire()
