import json
import os
import sys

import wandb
from datasets import load_from_disk, DatasetDict, load_dataset, load_metric, Metric
from transformers import PreTrainedModel, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer

from train import augment_dataset, compute_metrics, tokenize_function, filter_to_n_per_class


def main(path_to_clean_model: str = "bert-base-cased",  path_to_aug_model: str = "bert-base-cased",
         dataset_name: str = "Brendan/yahoo_answers", run_name: str = "test", run_group: str = "test",
         num_examples_per_class: int = 100, augmentation_strategy: str = None, dataset_full_length: int = 6000,
         num_aug_per_instance: int = 5, **kwargs):

    # load the datasets
    clean_dataset: DatasetDict = load_dataset(dataset_name)

    # filter to just N per class
    clean_dataset = filter_to_n_per_class(clean_dataset, num_examples_per_class)

    try:
        aug_dataset: DatasetDict = load_from_disk(f"{os.getcwd()}/data/{dataset_name}/{augmentation_strategy}/"
                                              f"length_{dataset_full_length}/num_aug_{num_aug_per_instance}")
    except BaseException as e:
        # NOTE: we are heavily reliant on the consistency of shuffling with random seeds here, though it should work
        print("WARNING: not expecting to need to re-create dataset, make sure to verify its consistent with training")
        # filter to just 10 per class
        aug_dataset: DatasetDict = clean_dataset
        # apply the augmentation strategy, if specified
        if augmentation_strategy:
            aug_dataset = augment_dataset(dataset_name, augmentation_strategy, aug_dataset, num_aug_per_instance)

    # tokenize each dataset
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    clean_dataset = clean_dataset.map(lambda example_batch: tokenize_function(example_batch, tokenizer))
    aug_dataset = aug_dataset.map(lambda example_batch: tokenize_function(example_batch, tokenizer))

    # load each model
    clean_model: PreTrainedModel = AutoModelForSequenceClassification\
        .from_pretrained(path_to_clean_model, num_labels=len(clean_dataset['train'] .unique("label")))
    aug_model: PreTrainedModel = AutoModelForSequenceClassification\
        .from_pretrained(path_to_aug_model, num_labels=len(clean_dataset['train'] .unique("label")))

    # set up a huggingface trainer and accuracy computer (this may not be necessary since we don't train but it works)
    training_args: TrainingArguments = TrainingArguments(
        output_dir=f"./checkpoints/{run_group}/{run_name}",
        per_device_eval_batch_size=512
    )
    acc_metric: Metric = load_metric("accuracy")
    f1_metric: Metric = load_metric("f1")
    clean_trainer = Trainer(
        model=clean_model,
        args=training_args,
        train_dataset=clean_dataset['train'],
        eval_dataset=clean_dataset['test'],
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred=eval_pred, acc_metric=acc_metric,
                                                          f1_metric=f1_metric),
    )
    
    aug_trainer = Trainer(
        model=aug_model,
        args=training_args,
        train_dataset=aug_dataset['train'],
        eval_dataset=aug_dataset['test'],
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred=eval_pred, acc_metric=acc_metric,
                                                          f1_metric=f1_metric),
    )

    # we'll just compute everything, and figure out diversity, etc. later
    aug_trainer.evaluate(clean_dataset['train'], metric_key_prefix="aug_on_clean_train")
    aug_trainer.evaluate(clean_dataset['test'], metric_key_prefix="aug_on_clean_test")
    aug_trainer.evaluate(aug_dataset['train'], metric_key_prefix="aug_on_aug_train")
    aug_trainer.evaluate(aug_dataset['test'], metric_key_prefix="aug_on_aug_test")
    clean_trainer.evaluate(aug_dataset['train'], metric_key_prefix="clean_on_aug_train")
    clean_trainer.evaluate(aug_dataset['test'], metric_key_prefix="clean_on_aug_test")
    clean_trainer.evaluate(clean_dataset['train'], metric_key_prefix="clean_on_clean_train")
    clean_trainer.evaluate(clean_dataset['test'], metric_key_prefix="clean_on_clean_test")


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        with open(sys.argv[1]) as f:
            kwargs = json.loads(f.read())
    else:
        # some defaults
        kwargs = {
            "path_to_aug_model": "final_models/bert_substitute_train/bert_substitute_100",
            "path_to_clean_model": "final_models/clean_train/clean_100",
            "augmentation_strategy": "bert_substitute"
        }
    run = wandb.init(project="text-aug-experiments", entity="kingb12", name=kwargs.get("run_name", "test"),
                     notes=kwargs.get("run_notes", "test"), group=kwargs.get("run_group", "test"))
    wandb.log(kwargs)
    main(**kwargs)