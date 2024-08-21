from typing import Optional, List

from fire import Fire
from pydantic import BaseModel

from data_loading import ArgumentSample, ArgumentData


class Prompter(BaseModel):

    def run(self, data_train: ArgumentSample, sample_test: ArgumentSample) -> str:
        raise NotImplementedError

    def get_answer(self, text: str) -> str:
        raise NotImplementedError
    

class ConclugenBasePrompter(Prompter):
    def run(self, data_train: ArgumentData, sample_test: ArgumentSample) -> str:
        prompt = "Based on the evidence presented, what is the most logical and justifiable stance to take on the issue at hand?\n"
        for sample in data_train.samples:
            prompt += f"Argument: {sample.src}\n"
            prompt += f"Conclusion: {sample.tgt}\n\n"

        prompt += f"Argument: {sample_test.src}\n"
        prompt += "Conclusion: "
        return prompt

    def get_answer(self, text: str) -> str:
        text = text.split('\n\n')[0]
        return text.strip()
    

class ConclugenTopicPrompter(Prompter):
    def run(self, data_train: ArgumentData, sample_test: ArgumentSample) -> str:
        prompt = "Based on the evidence presented, what is the most logical and justifiable stance to take on the issue at hand?\n"
        for sample in data_train.samples:
            topic = sample.src.split('<|TOPIC|>')[1].split('<|ARGUMENT|>')[0]
            argument = sample.src.split('<|ARGUMENT|>')[1].split('<|CONCLUSION|>')[0]
            prompt += f"Topic: {topic}\n"
            prompt += f"Argument: {argument}\n"
            prompt += f"Conclusion: {sample.tgt}\n\n"

        topic = sample_test.src.split('<|TOPIC|>')[1].split('<|ARGUMENT|>')[0]
        argument = sample_test.src.split('<|ARGUMENT|>')[1].split('<|CONCLUSION|>')[0]
        prompt += f"Topic: {topic}\n"
        prompt += f"Argument: {argument}\n"
        prompt += "Conclusion: "
        return prompt
    
    def get_answer(self, text: str) -> str:
        text = text.split('\n\n')[0]
        return text.strip()
    

class ConclugenAspectsPrompter(Prompter):
    def run(self, data_train: ArgumentData, sample_test: ArgumentSample) -> str:
        prompt = "Based on the evidence presented, what is the most logical and justifiable stance to take on the issue at hand?\n"
        for sample in data_train.samples:
            topic = sample.src.split('<|TOPIC|>')[1].split('<|ARGUMENT|>')[0]
            argument = sample.src.split('<|ARGUMENT|>')[1].split('<|ASPECTS|>')[0]
            aspects = sample.src.split('<|ASPECTS|>')[1].split('<|CONCLUSION|>')[0]
            prompt += f"Topic: {topic}\n"
            prompt += f"Argument: {argument}\n"
            prompt += f"Aspects: {aspects}\n"
            prompt += f"Conclusion: {sample.tgt}\n\n"

        topic = sample_test.src.split('<|TOPIC|>')[1].split('<|ARGUMENT|>')[0]
        argument = sample_test.src.split('<|ARGUMENT|>')[1].split('<|ASPECTS|>')[0]
        aspects = sample_test.src.split('<|ASPECTS|>')[1].split('<|CONCLUSION|>')[0]
        prompt += f"Topic: {topic}\n"
        prompt += f"Argument: {argument}\n"
        prompt += f"Aspects: {aspects}\n"
        prompt += "Conclusion: "
        return prompt
    
    def get_answer(self, text: str) -> str:
        text = text.split('\n\n')[0]
        return text.strip()
    

class ConclugenTargetsPrompter(Prompter):
    def run(self, data_train: ArgumentData, sample_test: ArgumentSample) -> str:
        prompt = "Based on the evidence presented, what is the most logical and justifiable stance to take on the issue at hand?\n"
        for sample in data_train.samples:
            topic = sample.src.split('<|TOPIC|>')[1].split('<|ARGUMENT|>')[0]
            argument = sample.src.split('<|ARGUMENT|>')[1].split('<|TARGETS|>')[0]
            targets = sample.src.split('<|TARGETS|>')[1].split('<|CONCLUSION|>')[0]
            prompt += f"Topic: {topic}\n"
            prompt += f"Argument: {argument}\n"
            prompt += f"Targets: {targets}\n"
            prompt += f"Conclusion: {sample.tgt}\n\n"

        topic = sample_test.src.split('<|TOPIC|>')[1].split('<|ARGUMENT|>')[0]
        argument = sample_test.src.split('<|ARGUMENT|>')[1].split('<|TARGETS|>')[0]
        targets = sample_test.src.split('<|TARGETS|>')[1].split('<|CONCLUSION|>')[0]
        prompt += f"Topic: {topic}\n"
        prompt += f"Argument: {argument}\n"
        prompt += f"Targets: {targets}\n"
        prompt += "Conclusion: "
        return prompt
    
    def get_answer(self, text: str) -> str:
        text = text.split('\n\n')[0]
        return text.strip()
    

class DebatesumAbstractivePrompter(Prompter):
    def run(self, data_train: ArgumentData, sample_test: ArgumentSample) -> str:
        prompt = "What is the main idea or argument presented in the document?\n"
        for sample in data_train.samples:
            prompt += f"Document: {sample.src}\n"
            prompt += f"Abstractive Summary: {sample.tgt}\n\n"

        prompt += f"Document: {sample_test.src}\n"
        prompt += "Abstractive Summary: "
        return prompt

    def get_answer(self, text: str) -> str:
        text = text.split('\n\n')[0]
        return text.strip()
    

class DebatesumExtractivePrompter(Prompter):
    def run(self, data_train: ArgumentData, sample_test: ArgumentSample) -> str:
        prompt = "Identify the main points and supporting evidence in the document that support the argument being made.\n"
        for sample in data_train.samples:
            prompt += f"Document: {sample.src}\n"
            prompt += f"Extractive Summary: {sample.tgt}\n\n"

        prompt += f"Document: {sample_test.src}\n"
        prompt += "Extractive Summary: "
        return prompt

    def get_answer(self, text: str) -> str:
        text = text.split('\n\n')[0]
        return text.strip()
    

class CounterargumentPremisesPrompter(Prompter):
    def run(self, data_train: ArgumentData, sample_test: ArgumentSample) -> str:
        prompt = "Identify a premise for the claim and come up with a counter-argument that challenges the validity of that premise.\n"
        for sample in data_train.samples:
            claim, premises = sample.src.split('\t')
            prompt += f"Claim: {claim}\n"
            prompt += f"Premises: {premises}\n"
            prompt += f"Counter Argument: {sample.tgt}\n\n"

        claim, premises = sample_test.src.split('\t')
        prompt += f"Claim: {claim}\n"
        prompt += f"Premises: {premises}\n"
        prompt += "Counter Argument: "
        return prompt

    def get_answer(self, text: str) -> str:
        text = text.split('\n\n')[0]
        return text.strip()
    

class CounterargumentWeakPremisesPrompter(Prompter):
    def run(self, data_train: ArgumentData, sample_test: ArgumentSample) -> str:
        prompt = "Identify a weak premise for the claim and formulate a counter-argument to challenge it.\n"
        for sample in data_train.samples:
            claim, premises = sample.src.split('\t')
            prompt += f"Claim: {claim}\n"
            prompt += f"Weak Premises: {premises}\n"
            prompt += f"Counter Argument: {sample.tgt}\n\n"

        claim, premises = sample_test.src.split('\t')
        prompt += f"Claim: {claim}\n"
        prompt += f"Weak Premises: {premises}\n"
        prompt += "Counter Argument: "
        return prompt

    def get_answer(self, text: str) -> str:
        text = text.split('\n\n')[0]
        return text.strip()
    

class ClaimDetectionPrompter(Prompter):
    def run(self, data_train: ArgumentData, sample_test: ArgumentSample) -> str:
        prompt = "Identify whether the given sentence is a claim towards the given topic. Choose from 'claim' or 'non claim'.\n"
        for sample in data_train.samples:
            topic, sentence = sample.src.split('\t')
            prompt += f"Topic: {topic}\n"
            prompt += f"Sentence: {sentence}\n"
            prompt += f"Label: {sample.tgt}\n\n"

        topic, sentence = sample_test.src.split('\t')
        prompt += f"Topic: {topic}\n"
        prompt += f"Sentence: {sentence}\n"
        prompt += "Label: "
        return prompt

    def get_answer(self, text: str) -> str:
        text = text.split('\n')[0]
        text = text.replace('-', ' ')
        return text.strip().lower()
    

class ArgumentDetectionPrompter(Prompter):
    def run(self, data_train: ArgumentData, sample_test: ArgumentSample) -> str:
        prompt = "Identify whether the given sentence is an argument towards the given topic. Choose from 'argument' or 'non argument'.\n"
        for sample in data_train.samples:
            topic, sentence = sample.src.split('\t')
            prompt += f"Topic: {topic}\n"
            prompt += f"Sentence: {sentence}\n"
            prompt += f"Label: {sample.tgt}\n\n"

        topic, sentence = sample_test.src.split('\t')
        prompt += f"Topic: {topic}\n"
        prompt += f"Sentence: {sentence}\n"
        prompt += "Label: "
        return prompt

    def get_answer(self, text: str) -> str:
        text = text.split('\n')[0]
        text = text.replace('-', ' ')
        return text.strip().lower()
    

class EvidenceDetectionIAMPrompter(Prompter):
    def run(self, data_train: ArgumentData, sample_test: ArgumentSample) -> str:
        prompt = "Identify whether the given sentence is a piece of evidence towards the given claim. Choose from 'evidence' or 'non evidence'.\n"
        for sample in data_train.samples:
            claim, sentence = sample.src.split('\t')
            prompt += f"Claim: {claim}\n"
            prompt += f"Sentence: {sentence}\n"
            prompt += f"Label: {sample.tgt}\n\n"

        claim, sentence = sample_test.src.split('\t')
        prompt += f"Claim: {claim}\n"
        prompt += f"Sentence: {sentence}\n"
        prompt += "Label: "
        return prompt

    def get_answer(self, text: str) -> str:
        text = text.split('\n')[0]
        text = text.replace('-', ' ')
        return text.strip().lower()
    

class EvidenceDetectionIBMPrompter(Prompter):
    def run(self, data_train: ArgumentData, sample_test: ArgumentSample) -> str:
        prompt = "Identify whether the given sentence is a piece of evidence towards the given topic. Choose from 'evidence' or 'non evidence'.\n"
        for sample in data_train.samples:
            topic, sentence = sample.src.split('\t')
            prompt += f"Topic: {topic}\n"
            prompt += f"Sentence: {sentence}\n"
            prompt += f"Label: {sample.tgt}\n\n"

        topic, sentence = sample_test.src.split('\t')
        prompt += f"Topic: {topic}\n"
        prompt += f"Sentence: {sentence}\n"
        prompt += "Label: "
        return prompt

    def get_answer(self, text: str) -> str:
        text = text.split('\n')[0]
        text = text.replace('-', ' ')
        return text.strip().lower()
    

class StanceDetectionPrompter(Prompter):
    def run(self, data_train: ArgumentData, sample_test: ArgumentSample) -> str:
        prompt = "Identify the stance of the given sentence towards the given topic. Choose from 'support' or 'attack'.\n"
        for sample in data_train.samples:
            topic, sentence = sample.src.split('\t')
            prompt += f"Topic: {topic}\n"
            prompt += f"Sentence: {sentence}\n"
            prompt += f"Label: {sample.tgt}\n\n"

        topic, sentence = sample_test.src.split('\t')
        prompt += f"Topic: {topic}\n"
        prompt += f"Sentence: {sentence}\n"
        prompt += "Label: "
        return prompt

    def get_answer(self, text: str) -> str:
        text = text.split('\n')[0]
        return text.strip().lower()


class StanceDetectionFeverPrompter(Prompter):
    def run(self, data_train: ArgumentData, sample_test: ArgumentSample) -> str:
        prompt = "Identify the stance of the given sentence. Choose from 'support', 'attack', or 'neutral'.\n"
        for sample in data_train.samples:
            prompt += f"Sentence: {sample.src}\n"
            prompt += f"Label: {sample.tgt}\n\n"

        prompt += f"Sentence: {sample_test.src}\n"
        prompt += "Label: "
        return prompt

    def get_answer(self, text: str) -> str:
        text = text.split('\n')[0]
        return text.strip().lower()
    

class StanceDetectionMTSDPrompter(Prompter):
    def run(self, data_train: ArgumentData, sample_test: ArgumentSample) -> str:
        prompt = "Identify the stance of the given sentence towards each given target. Choose from 'support', 'attack', or 'neutral' for each target in the target pair. Format the output as a label pair: label1, label2.\n"
        for sample in data_train.samples:
            sentence, target1, target2 = sample.src.split('\t')
            label1, label2 = sample.tgt.split('\t')
            prompt += f"Sentence: {sentence}\n"
            prompt += f"Target Pair: {target1}, {target2}\n"
            prompt += f"Label Pair: {label1}, {label2}\n\n"

        sentence, target1, target2 = sample_test.src.split('\t')
        prompt += f"Sentence: {sentence}\n"
        prompt += f"Target Pair: {target1}, {target2}\n"
        prompt += "Label Pair: "
        return prompt

    def get_answer(self, text: str) -> str:
        text = text.split('\n')[0]
        text = text.replace(', ', '\t')
        return text.strip().lower()
    

class EvidenceClassificationIBMPrompter(Prompter):
    def run(self, data_train: ArgumentData, sample_test: ArgumentSample) -> str:
        prompt = "Identify the evidence type of the given sentence. Choose from 'study', 'anecdotal' or 'expert'.\n"
        for sample in data_train.samples:
            prompt += f"Sentence: {sample.src}\n"
            prompt += f"Evidence Type: {sample.tgt}\n\n"

        prompt += f"Sentence: {sample_test.src}\n"
        prompt += "Evidence Type: "
        return prompt

    def get_answer(self, text: str) -> str:
        text = text.split('\n')[0]
        return text.strip().lower()
    

class EvidenceClassificationAQEPrompter(Prompter):
    def run(self, data_train: ArgumentData, sample_test: ArgumentSample) -> str:
        prompt = "Identify the evidence type of the given sentence. Choose from 'research', 'case', 'expert', 'explanation' or 'others'.\n"
        for sample in data_train.samples:
            prompt += f"Sentence: {sample.src}\n"
            prompt += f"Evidence Type: {sample.tgt}\n\n"

        prompt += f"Sentence: {sample_test.src}\n"
        prompt += "Evidence Type: "
        return prompt
    

    def get_answer(self, text: str) -> str:
        text = text.split('\n')[0]
        return text.strip().lower()


def select_prompter(task: str, data_name: str) -> Prompter:
    if task == "conclugen":
        if data_name == "base":
            return ConclugenBasePrompter()
        elif data_name == "aspects":
            return ConclugenAspectsPrompter()
        elif data_name == "targets":
            return ConclugenTargetsPrompter()
        elif data_name == "topic":
            return ConclugenTopicPrompter()
        else:
            raise ValueError(f"Invalid data name: {data_name}")
    elif task == "debatesum":
        if data_name == "abstract":
            return DebatesumAbstractivePrompter()
        elif data_name == "extract":
            return DebatesumExtractivePrompter()
        else:
            raise ValueError(f"Invalid data name: {data_name}")
    elif task == "counter_arg_gen":
        if data_name == "weak_premises":
            return CounterargumentWeakPremisesPrompter()
        elif data_name == "premises":
            return CounterargumentPremisesPrompter()
        else:
            raise ValueError(f"Invalid data name: {data_name}")
    elif task == "claim_detection":
        if data_name == "ibm_argument":
            return ArgumentDetectionPrompter()
        elif data_name in ["iam_claims", "ibm_claims"]:
            return ClaimDetectionPrompter()
        else:
            raise ValueError(f"Invalid data name: {data_name}")
    elif task == "evidence_detection":
        if data_name == "iam_evidence":
            return EvidenceDetectionIAMPrompter()
        elif data_name == "ibm_evidence":
            return EvidenceDetectionIBMPrompter()
        else:
            raise ValueError(f"Invalid data name: {data_name}")
    elif task == "stance_detection":
        if data_name == "fever":
            return StanceDetectionFeverPrompter()
        elif data_name == "mtsd":
            return StanceDetectionMTSDPrompter()
        elif data_name in ["ibm_stance", "iam_stance"]:
            return StanceDetectionPrompter()
        else:
            raise ValueError(f"Invalid data name: {data_name}")
    elif task == "evidence_classification":
        if data_name == "aqe_type":
            return EvidenceClassificationAQEPrompter()
        elif data_name == "ibm_type":
            return EvidenceClassificationIBMPrompter()
        else:
            raise ValueError(f"Invalid data name: {data_name}")
    else:
        raise ValueError(f"Invalid task: {task}")
    

def test_prompt(task: str, data_name: str, num_train: int, seed: int):
    data_train, data_test = ArgumentData.load(task, data_name, num_train, seed)
    prompter = select_prompter(task, data_name)
    prompt = prompter.run(data_train, data_test.samples[0])
    print(prompt)


if __name__ == "__main__":
    Fire()