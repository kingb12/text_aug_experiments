# Adapting the BERT training from this tutorial: https://huggingface.co/docs/transformers/training
import json
import os
import sys
from collections import Counter
from typing import List, Any, Dict

import numpy as np
import torch.cuda
import wandb as wandb
from datasets import load_dataset, DatasetDict, load_metric, Metric, concatenate_datasets
import nlpaug.flow as naf
import nlpaug.augmenter.word as naw
from nlpaug import Augmenter
from transformers import AutoTokenizer, PreTrainedTokenizer, BatchEncoding, PreTrainedModel, \
    AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction, EarlyStoppingCallback

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


def compute_metrics(eval_pred: EvalPrediction, acc_metric: Metric, f1_metric: Metric) -> Dict[str, Any]:
    """
    Compute accuracy and other metrics

    :param eval_pred: an EvalPrediction (has logits and labels?) from tutorial
    :param acc_metric: accuracy metric
        (see https://huggingface.co/docs/datasets/v2.2.1/en/package_reference/main_classes#datasets.Metric)
    :param f1_metric: F1 metric
        (see https://huggingface.co/spaces/evaluate-metric/f1)
    :return: dict of computed metrics
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    result: Dict[str, Any] = acc_metric.compute(predictions=predictions, references=labels)
    result.update(f1_metric.compute(predictions=predictions, references=labels, average='macro'))
    return result


def augment_dataset(dataset_name: str, augmentation_strategy: str, dataset: DatasetDict, num_aug_per_instance: int) -> DatasetDict:
    if augmentation_strategy == 'bert_substitute':
        aug: Augmenter = naw.ContextualWordEmbsAug(model_path='bert-base-cased', action='substitute',
                                                   device=torch.cuda.is_available() and 'cuda:0' or 'cpu',
                                                   batch_size=128)
    else:
        raise NotImplementedError(f"{augmentation_strategy} not supported")
    augmentations: List[DatasetDict] = []
    for _ in range(num_aug_per_instance):
        # create a unique augmentation: dataset.map requires equal length in/out, hence add to list + concatenate
        augmentations.append(dataset.map(lambda example_batch: {"text": aug.augment(example_batch['text'])},
                                         batched=True, batch_size=128))
    # merge to one dataset and return
    for split in dataset:
        dataset[split] = concatenate_datasets([dataset[split]] + [a[split] for a in augmentations])
    dataset = dataset.shuffle(seed=42)

    # save this for later use TODO: make correct use of this
    dataset.save_to_disk(f"{os.getcwd()}/data/{dataset_name}/length_{len(dataset['train'])}/num_aug_{num_aug_per_instance}")
    return dataset


def main(model_name: str = "bert-base-cased", dataset_name: str = "Brendan/yahoo_answers",
         run_name: str = "test", run_group: str = "test", num_examples_per_class: int = 100,
         augmentation_strategy: str = None, num_aug_per_instance: int = 5, **kwargs):
    dataset: DatasetDict = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # filter to just 10 per class
    for split in dataset:
        example_class_counter: Counter[int] = Counter()

        def only_n_per_class(example) -> bool:
            example_class_counter[example['label']] += 1
            return example_class_counter[example['label']] <= num_examples_per_class

        dataset[split] = dataset[split].shuffle(seed=42).filter(only_n_per_class)

    # apply the augmentation strategy, if specified
    if augmentation_strategy:
        dataset = augment_dataset(dataset_name, augmentation_strategy, dataset, num_aug_per_instance)

    # tokenize all the texts
    dataset = dataset.map(lambda example_batch: tokenize_function(example_batch, tokenizer))

    # validate tokenization
    for key in ['label', 'text', 'input_ids', 'token_type_ids', 'attention_mask']:
        assert key in dataset['train'][0]
    decoded: str = tokenizer.decode(dataset['train'][0]['input_ids'])
    assert decoded.startswith("[CLS] ") and "[SEP]" in decoded

    # initialize the model
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained("bert-base-cased",
                                                                                # should be 10 for Yahoo answers
                                                                                num_labels=len(dataset['train']
                                                                                               .unique("label")))

    # set up a huggingface trainer and accuracy computer
    training_args: TrainingArguments = TrainingArguments(
        output_dir=f"./checkpoints/{run_group}/{run_name}",
        eval_steps=100,
        evaluation_strategy="steps",
        save_steps=100,
        num_train_epochs=20,
        logging_steps=200,
        save_total_limit=8,
        max_grad_norm=1.0,
        metric_for_best_model="accuracy",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=256,
        load_best_model_at_end=True,
        dataloader_num_workers=4,
        fp16=True
    )

    # overwrite training arguments with anything from our key-word arguments (may come from config file)
    for parameter, value in kwargs.items():
        training_args.__setattr__(parameter, value)

    acc_metric: Metric = load_metric("accuracy")
    f1_metric: Metric = load_metric("f1")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
        # for accuracy, we need no arguments but for F1 we need to specify 'macro' average:
            # ‘macro’: Calculate metrics for each label, and find their unweighted mean.
            # This does not take label imbalance into account (ok for us, we balanced the classes)
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred=eval_pred, acc_metric=acc_metric, f1_metric=f1_metric),
    )

    trainer.evaluate()
    trainer.train()
    trainer.evaluate(dataset['train'], metric_key_prefix="final_train")
    trainer.evaluate(dataset['test'], metric_key_prefix="final_test")

    trainer.save_model(f"./final_models/{run_group}/{run_name}")


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        with open(sys.argv[1]) as f:
            kwargs = json.loads(f.read())
    else:
        # some defaults
        kwargs = {
            "eval_steps": 10,
            "save_steps": 10,
            "logging_steps": 10,
            "augmentation_strategy": "bert_substitute"
        }
    run = wandb.init(project="text-aug-experiments", entity="kingb12", name=kwargs.get("run_name", "test"),
                     notes=kwargs.get("run_notes", "test"), group=kwargs.get("run_group", "test"))
    kwargs['push_to_hub_model_id'] = run.name
    wandb.log(kwargs)
    main(**kwargs)
