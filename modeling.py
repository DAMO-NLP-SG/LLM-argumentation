import time
import json
from typing import Optional

import torch
import openai
import tiktoken
from fire import Fire
from pydantic import BaseModel
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


class DummyImport:
    LLM = None
    SamplingParams = None


try:
    import vllm
    from vllm.lora.request import LoRARequest
except ImportError:
    print("vLLM not installed")
    vllm = DummyImport()
    LoRARequest = lambda *args: args


class EvalModel(BaseModel, arbitrary_types_allowed=True):
    path_model: str
    max_input_length: int = 512
    max_output_length: int = 512

    def run(self, prompt: str) -> str:
        raise NotImplementedError


class VLLMModel(EvalModel):
    path_model: str
    model: vllm.LLM = None
    quantization: Optional[str] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    tensor_parallel_size: int = 1

    def load(self):
        if self.model is None:
            self.model = vllm.LLM(
                model=self.path_model,
                trust_remote_code=True,
                quantization=self.quantization,
                tensor_parallel_size=self.tensor_parallel_size,
            )
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.path_model)

    def format_prompt(self, prompt: str) -> str:
        self.load()
        prompt = prompt.rstrip(" ")
        return prompt

    def make_kwargs(self, do_sample: bool, **kwargs) -> dict:
        params = vllm.SamplingParams(
            temperature=0.5 if do_sample else 0.0,
            max_tokens=self.max_output_length,
            **kwargs
        )
        outputs = dict(sampling_params=params, use_tqdm=False)
        return outputs

    def run(self, prompt: str) -> str:
        prompt = self.format_prompt(prompt)
        outputs = self.model.generate([prompt], **self.make_kwargs(do_sample=False))
        pred = outputs[0].outputs[0].text
        pred = pred.split("<|endoftext|>")[0]
        return pred
    
    def check_valid_length(self, text: str) -> bool:
        self.load()
        inputs = self.tokenizer(text)
        return len(inputs.input_ids) <= self.max_input_length

    def truncate_input(self, input) -> str:
        return self.tokenizer.decode(self.tokenizer(input).input_ids[:self.max_input_length])
    

class SeqToSeqModel(EvalModel):
    path_model: str
    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    device: str = "cuda"
    load_8bit: bool = False
    fp16: bool = False

    def load(self):
        if "flan-ul2" in self.path_model.lower():
            self.max_input_length = 2048
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            elif self.fp16:
                args.update(device_map="auto", torch_dtype=torch.float16)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.path_model, **args)
            if self.fp16 or self.load_8bit:
                self.model.eval()
            else:
                self.model.to(self.device)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.path_model)

    def run(self, prompt: str) -> str:
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_output_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def check_valid_length(self, text: str) -> bool:
        self.load()
        inputs = self.tokenizer(text)
        return len(inputs.input_ids) <= self.max_input_length

    def truncate_input(self, input) -> str:
        return self.tokenizer.decode(self.tokenizer(input).input_ids[:self.max_input_length])


class OpenAIModel(EvalModel):
    path_model: str
    tokenizer: Optional[tiktoken.Encoding]
    temperature: float = 0.0
    max_input_length: int = 3996 # to allow 100 tokens for response

    def load(self):
        if self.tokenizer is None:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # chatgpt/gpt-4

        with open(self.path_model) as f:
            info = json.load(f)
            openai.api_key = info["key"]
            self.model = info["model"]

    def run(self, prompt: str) -> str:
        self.load()
        while True:
            try:
                messages = [{"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature, 
                )
                output = response.choices[0].message["content"]
                break
            except Exception as e:
                print(e)
                time.sleep(5)
                continue
        return output

    def check_valid_length(self, prompt: str) -> bool:
        self.load()
        tokens_per_message = 4 
        tokens_per_name = -1  
        messages = [{"role": "user", "content": prompt}]
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(self.tokenizer.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3

        return num_tokens <= self.max_input_length

    def truncate_input(self, input) -> str:
        return self.tokenizer.decode(self.tokenizer.encode(input)[:self.max_input_length-8])


def select_model(model_name: str, **kwargs) -> EvalModel:
    model_map = dict(
        flan_t5_xl=SeqToSeqModel,
        flan_t5_xxl = SeqToSeqModel,
        flan_ul2 = SeqToSeqModel,
        openai=OpenAIModel,     
        llama2_7b=VLLMModel,
        llama2_13b=VLLMModel,
    )
    model_class = model_map.get(model_name)
    if model_class is None:
        raise ValueError(f"{model_name}. Choose from {list(model_map.keys())}")
    return model_class(**kwargs)


def test_model(
    prompt: str = "Identify the stance of the given sentence. Choose from 'support', 'attack', or 'neutral'.\nSentence: Menace II Society is a motion picture.\nLabel: ",
    model_name: str = "flan_t5_xl",
    path_model: str = "google/flan-t5-xl",
    **kwargs,
):
    
    model = select_model(model_name, path_model=path_model, **kwargs)
    print(locals())
    print(model.check_valid_length(prompt))
    if not model.check_valid_length(prompt):
        prompt = model.truncate_input(prompt)
        print(f"Truncated prompt: {prompt}\n Length:{model.max_input_length}")

    print(model.run(prompt))


if __name__ == "__main__":
    Fire()
