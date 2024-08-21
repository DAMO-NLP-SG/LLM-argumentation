import json
import os

from typing import List
from fire import Fire

from pydantic import BaseModel


class ArgumentSample(BaseModel):
    src: str = ""
    tgt: str = ""
    prompt: str = ""
    raw: str = ""
    pred: str = ""

    @classmethod
    def default_format(cls, src: str, tgt: str):
        src = src.strip()
        tgt = tgt.strip()
        src = src.replace('\n', ' ')
        tgt = tgt.replace('\n', ' ')

        return cls(src=src, tgt=tgt)
    

class ArgumentData(BaseModel):
    samples: List[ArgumentSample]

    @classmethod
    def load_from_paths(cls, src_path: str, tgt_path: str):
        with open(src_path) as f:
            raw_src = [line for line in f]
        with open(tgt_path) as f:
            raw_tgt = [line for line in f]

        assert len(raw_src) == len(raw_tgt)

        return cls(samples=[ArgumentSample.default_format(line_src, line_tgt) for line_src, line_tgt in zip(raw_src, raw_tgt)])
    
    @classmethod
    def load_train(cls, task: str, data_name: str, num_train: int, seed: int):
        train_folder = f"sampled_data/{task}/{data_name}/train/{num_train}_shot/seed_{seed}"
        if not os.path.isdir(train_folder):
            return cls(samples=[])
        else:
            train_src_path = f"{train_folder}/source.txt"
            train_tgt_path = f"{train_folder}/target.txt"
            return cls.load_from_paths(train_src_path, train_tgt_path)
        
    @classmethod
    def load_test(cls, task: str, data_name: str):
        test_folder = f"sampled_data/{task}/{data_name}/test"
        test_src_path = f"{test_folder}/source.txt"
        test_tgt_path = f"{test_folder}/target.txt"
        return cls.load_from_paths(test_src_path, test_tgt_path)

    @classmethod
    def load(cls, task: str, data_name: str, num_train: int, seed: int):
        data_train = cls.load_train(task, data_name, num_train, seed)
        data_test = cls.load_test(task, data_name)
        return data_train, data_test
    
    @classmethod
    def load_outputs(cls, output_path: str):
        samples = []
        with open(output_path) as f:
            for line in f:
                samples.append(ArgumentSample(**json.loads(line.strip())))
        return cls(samples=samples)
    

def test_data(task: str, data_name: str, num_train: int, seed: int):
    data_train, data_test = ArgumentData.load(task, data_name, num_train, seed)
    print(data_test.samples[0])
    print("num train: ", len(data_train.samples))
    print("num test: ", len(data_test.samples))


if __name__ == "__main__":
    Fire()
