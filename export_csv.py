import pandas as pd
import wandb

from eval_affinity_diversity import calc_clean_on_aug

if __name__ == '__main__':
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs("kingb12/text-aug-experiments")

    summary_list, config_list, name_list, train_acc, test_acc, train_loss, test_loss, aug_prob = [], [], [], [], [], [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        if 'run_group' not in run.summary._json_dict or run.summary._json_dict['run_group'] == 'eval' or \
            run.name == 'test':
            continue
        summary_list.append(run.summary._json_dict)
        
        train_loss.append(run.summary._json_dict['train/final_train_loss'])
        test_loss.append(run.summary._json_dict['train/final_test_loss'])
        train_acc.append(run.summary._json_dict['train/final_train_accuracy'])
        test_acc.append(run.summary._json_dict['train/final_test_accuracy'])
        aug_prob.append(run.summary._json_dict.get('augment_probability', 0.0))
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
              if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame({
        "name": name_list,
        #"summary": summary_list,
        #"config": config_list,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "aug_prob": aug_prob
        })

    # this now holds clean on clean, aug on aug, but we need clean on aug for affinity. Take the clean model and run
    # it against each augmented dataset. Technically, these are new augmentations, but its also test accuracy, so we can
    # assume the differences are i.i.d.
    clean_on_aug_test_acc = []
    for r in runs_df.iterrows():
        p = r[1]['aug_prob']
        if p == 0.0:
            clean_on_aug_test_acc.append(r[1]['test_acc'])
        else:
            result = calc_clean_on_aug(path_to_clean_model="final_models/clean_train/clean_100", augmentation_strategy="bert_substitute", augment_probability=p)
            clean_on_aug_test_acc.append(result['clean_on_aug_test_accuracy'])
            print(result)
    runs_df['clean_on_aug_test_acc'] = clean_on_aug_test_acc
    runs_df.to_csv("project.csv")