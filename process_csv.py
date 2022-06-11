import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == '__main__':
    runs_df = pd.read_csv("project.csv")
    runs_df['affinity'] = runs_df['clean_on_aug_test_acc'] / runs_df['test_acc']
    clean_train_loss = float(runs_df[runs_df['name'] == 'clean_100']['train_loss'])
    runs_df['diversity'] = runs_df['train_loss'] / clean_train_loss
    for r in runs_df.sort_values('aug_prob').iterrows():
        data = r[1]
        plt.scatter([data['affinity']], [data['diversity']],
                    label=f"$P(aug) = {data['aug_prob']}$",
                    c=data['aug_prob'],
                    vmin=-.2, vmax=1,
                    cmap='Greens')
    # plot hash lines to show (1, 1) == zero augmentation
    plt.plot(np.linspace(0.97, 1.01, 20), [1.0] * 20, color='lightgrey', linestyle='dashed')
    plt.plot([1.0] * 20, np.linspace(0.0, 1.42, 20), color='lightgrey', linestyle='dashed')
    plt.xlabel("Affinity")
    plt.ylabel("Diversity")
    plt.title("Affinity & Diversity for choices of word-augment probability")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig("affinity_vs_diversity_bert_substitute.png")