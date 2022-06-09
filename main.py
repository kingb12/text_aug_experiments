# Adapting from this tutorial: https://huggingface.co/docs/transformers/training
import json
import sys
from collections import Counter
from typing import List, Any, Dict

import torch
import wandb as wandb
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizer, BatchEncoding

# this is just a place holder type for a single example in the dataset, which at initial load has fields 'text' and
# 'label'
Example = Dict[str, Any]


def tokenize_function(examples_batched: Dict[str, List[Any]], tokenizer: PreTrainedTokenizer) -> BatchEncoding:
    """
    Tokenizes the text field of the data examples

    :param examples_batched: examples in {k: [v, v, v]} form
    :param tokenizer: tokenizer to tokenize with
    :return: a batch encoding (dict) that gets added to each example. See datasets.map
    """
    return tokenizer(examples_batched["text"], padding="max_length", truncation=True,
                     max_length=tokenizer.max_len_single_sentence)


def main(model_name: str = "bert-base-cased", dataset_name: str = "Brendan/yahoo_answers",
         num_examples_per_class: int = 100, **kwargs):
    dataset: DatasetDict = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # filter to just 10 per class
    for split in dataset:
        example_class_counter: Counter[int] = Counter()

        def only_n_per_class(example) -> bool:
            example_class_counter[example['label']] += 1
            return example_class_counter[example['label']] <= num_examples_per_class

        dataset[split] = dataset[split].shuffle(seed=42).filter(only_n_per_class)
    dataset


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        with open(sys.argv[1]) as f:
            kwargs = json.loads(f.read())
    else:
        # some defaults
        kwargs = {
            "eval_steps": 1,
            "save_steps": 500,
            "logging_steps": 100,
        }
    gpus: int = torch.cuda.device_count()
    per_device_train_batch_size: int = kwargs.get('per_device_train_batch_size', 32)
    kwargs['per_device_train_batch_size'] = per_device_train_batch_size
    gradient_accumulation_steps: int = 64 // (per_device_train_batch_size * gpus)
    run = wandb.init(project="text-aug-experiments", entity="kingb12", name=kwargs.get("run_name", "test"),
                     notes=kwargs.get("run_notes", "test"), group=kwargs.get("run_group", "test"))
    kwargs['push_to_hub_model_id'] = run.name
    kwargs['gradient_accumulation_steps'] = gradient_accumulation_steps
    wandb.log(kwargs)
    main(**kwargs)
