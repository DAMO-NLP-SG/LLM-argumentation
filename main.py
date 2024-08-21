import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import TextIO

from fire import Fire
from tqdm import tqdm

from data_loading import ArgumentSample, ArgumentData
from modeling import select_model, EvalModel
from prompting import select_prompter, Prompter
from scoring import select_scorer


def inference(
    model: EvalModel,
    data_train: ArgumentData,
    data_test: ArgumentData,
    prompter: Prompter,
    file: TextIO,
):

    progress = tqdm(data_test.samples)
    sample: ArgumentSample

    targets = []
    predictions = []
    for _, sample in enumerate(progress):
        k = int(len(data_train.samples))
        prompt = prompter.run(data_train, sample)
        # handle prompt length
        while not model.check_valid_length(prompt) and k > 0:
            k -= 1
            data_train.samples = data_train.samples[:k]
            prompt = prompter.run(data_train, sample)

        if not model.check_valid_length(prompt):
            prompt = model.truncate_input(prompt)

        # predict
        sample.prompt = prompt
        sample.raw = model.run(prompt)
        sample.pred = prompter.get_answer(sample.raw)
        print(sample.model_dump_json(), file=file)

        targets.append(sample.tgt)
        predictions.append(sample.pred)

    return predictions, targets

def main(
    task: str = "conclugen",
    data_name: str = "base",
    num_train: int = 5,
    seed: int = 0,
    **kwargs
):
    # load model
    model = select_model(**kwargs)
    print(locals())

    # select prompter
    prompter = select_prompter(task, data_name)

    # load data
    data_train, data_test = ArgumentData.load(task, data_name, num_train, seed)

    # set path
    output_folder = f"output/{task}/{data_name}/{num_train}_shot/seed_{seed}"
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    model_name = Path(model.path_model).stem
    output_path = f"{output_folder}/{model_name}.json"

    # infer
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as file:
        targets, predictions = inference(model, data_train, data_test, prompter, file)

    # score
    scorer = select_scorer(task)
    scores = scorer.run(predictions, targets)
    print(scores)


if __name__ == "__main__":
    Fire(main)
