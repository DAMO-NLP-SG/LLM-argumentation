# Exploring the Potential of Large Language Models in Computational Argumentation

This repo contains the data and codes for our paper ["Exploring the Potential of Large Language Models in Computational Argumentation"](https://aclanthology.org/2024.acl-long.126/) in ACL 2024.

### Abstract

Computational argumentation has become an essential tool in various domains, including law, public policy, and artificial intelligence. It is an emerging research field in natural language processing that attracts increasing attention. Research on computational argumentation mainly involves two types of tasks: argument mining and argument generation. As large language models (LLMs) have demonstrated impressive capabilities in understanding context and generating natural language, it is worthwhile to evaluate the performance of LLMs on diverse computational argumentation tasks. This work aims to embark on an assessment of LLMs, such as ChatGPT, Flan models, and LLaMA2 models, in both zero-shot and few-shot settings. We organize existing tasks into six main categories and standardize the format of fourteen openly available datasets. In addition, we present a new benchmark dataset on counter speech generation that aims to holistically evaluate the end-to-end performance of LLMs on argument mining and argument generation. Extensive experiments show that LLMs exhibit commendable performance across most of the datasets, demonstrating their capabilities in the field of argumentation. Our analysis offers valuable suggestions for evaluating computational argumentation and its integration with LLMs in future research endeavors.

### Setup

To install dependencies:

```
conda create -n llm-am python=3.9 -y
conda activate llm-am
pip install -r requirements.txt
```

To run OpenAI models, insert your [OpenAI key](https://platform.openai.com/account/api-keys) and model version in [openai_info.json](openai_info.json):

```
{
  "engine": "gpt-3.5-turbo-0301",
  "key": "YOUR API KEY"
}
```

### Example Usage

To run Flan-T5-XL on the ibm_claims dataset using 5-shot demonstrations:

```
python main.py \
--model_name flan_t5_xl \
--path_model google/flan-t5-xl \
--task claim_detection \
--data_name ibm_claims \
--num_train 5
```

The results will be printed as

```
{'accuracy': 0.74, 'f1': 0.7909496513561132}
```

(Note that some variance is possible)


Run using your own prompts by modifying the prompts in [prompting.py](prompting.py).

### Citation
```
@inproceedings{chen-etal-2024-exploring-potential,
    title = "Exploring the Potential of Large Language Models in Computational Argumentation",
    author = "Chen, Guizhen  and
      Cheng, Liying  and
      Luu, Anh Tuan  and
      Bing, Lidong",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.126",
    pages = "2309--2330",
    abstract = "Computational argumentation has become an essential tool in various domains, including law, public policy, and artificial intelligence. It is an emerging research field in natural language processing that attracts increasing attention. Research on computational argumentation mainly involves two types of tasks: argument mining and argument generation. As large language models (LLMs) have demonstrated impressive capabilities in understanding context and generating natural language, it is worthwhile to evaluate the performance of LLMs on diverse computational argumentation tasks. This work aims to embark on an assessment of LLMs, such as ChatGPT, Flan models, and LLaMA2 models, in both zero-shot and few-shot settings. We organize existing tasks into six main categories and standardize the format of fourteen openly available datasets. In addition, we present a new benchmark dataset on counter speech generation that aims to holistically evaluate the end-to-end performance of LLMs on argument mining and argument generation. Extensive experiments show that LLMs exhibit commendable performance across most of the datasets, demonstrating their capabilities in the field of argumentation. Our analysis offers valuable suggestions for evaluating computational argumentation and its integration with LLMs in future research endeavors.",
}
```