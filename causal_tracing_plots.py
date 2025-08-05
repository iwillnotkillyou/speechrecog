import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
from typing import Dict, List
import json
from scipy import stats
import argparse

class Feature:
    def __init__(self, name):
        self.name = name
        self.d = []

    def get_name(self):
        return self.name

    def to_array(self):
        return np.array(self.d)

    def add(self, v):
        self.d.append(v)

    def avg(self):
        np_array = np.array(self.d)
        return np.mean(np_array[~np.isnan(np_array)])

    def std(self):
        np_array = np.array(self.d)
        return np.std(np_array[~np.isnan(np_array)])

    def __len__(self):
        return len(self.d)

    def get(self, i):
        return self.d[i]

def read_json(json_path: str) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def group_results(facts, bucket):
    labels = ["subj-first", "subj-middle", "subj-last", "cont-first", "cont-middle", "cont-last"]

    corrupted_probs = Feature("corr")
    clean_probs = Feature("clean")
    results = {kind: {labels[i]: Feature(labels[i]) for i in range(6)} for kind in ["hidden", "mlp", "attn"]}

    target_token = f"{bucket}_token"

    for processed_fact in facts:

        processed_fact = processed_fact["results"]
        corrupted_score = processed_fact["corrupted"][target_token]["probs"]
        clean_score = processed_fact["clean"][target_token]["probs"]

        # If there is a zero interval, skip the fact
        interval_to_explain = max(clean_score - corrupted_score, 0)
        if interval_to_explain == 0:
            continue

        corrupted_probs.add(corrupted_score)
        clean_probs.add(clean_score)

        for kind in ["hidden", "mlp", "attn"]:
            (
                avg_first_subject,
                avg_middle_subject,
                avg_last_subject,
                avg_first_after,
                avg_middle_after,
                avg_last_after,
            ) = results[kind].values()

            tokens = processed_fact["tokens"]
            started_subject = False
            finished_subject = False
            temp_mid = 0.0
            count_mid = 0

            for token in tokens:
                interval_explained_average = 0
                for layer in token[kind]:
                    interval_explained_average += max(token[kind][layer][target_token]["probs"] - corrupted_score,
                                                      0) / len(token[kind])
                token_effect = min(interval_explained_average / interval_to_explain, 1)

                if "subject_pos" in token:
                    if not started_subject:
                        avg_first_subject.add(token_effect)
                        started_subject = True

                        if token["subject_pos"] == -1:
                            avg_last_subject.add(token_effect)
                    else:
                        subject_pos = token["subject_pos"]
                        if subject_pos == -1:
                            avg_last_subject.add(token_effect)
                        else:
                            temp_mid += token_effect
                            count_mid += 1
                else:
                    if not finished_subject:
                        # Process all subject middle tokens
                        if count_mid > 0:
                            avg_middle_subject.add(temp_mid / count_mid)
                            temp_mid = 0.0
                            count_mid = 0
                        else:
                            avg_middle_subject.add(0.0)
                        avg_first_after.add(token_effect)
                        finished_subject = True

                        if token["pos"] == -1:
                            avg_last_after.add(token_effect)
                    else:
                        token_pos = token["pos"]
                        if token_pos == -1:
                            avg_last_after.add(token_effect)
                        else:
                            temp_mid += token_effect
                            count_mid += 1

            if count_mid > 0:
                avg_middle_after.add(temp_mid / count_mid)
            else:
                avg_middle_after.add(0.0)

    return results, corrupted_probs, clean_probs

import numpy as np
from matplotlib import pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--causal_traces_dir", type=str, default="./causal_traces"
    )
    parser.add_argument("--output_dir", type=str, default="./plots")

    return parser.parse_args()


def plot(dataset_name, model_name, grounded_results, unfaithful_results, save_path):
    titles = {
        "hidden": "Hidden activations",
        "mlp": "MLPs",
        "attn": "Attention heads"
    }
    model_names_conversion = {
        "llama2": "Llama2-7B",
        "llama": "LLaMA-7B",
        "gpt2": "GPT2-XL",
    }
    dataset_names_conversion = {
        "base_fakepedia": "Fakepedia-base",
        "multihop_fakepedia": "FakePedia-MH"
    }

    labels = [feature.get_name() for feature in next(iter(grounded_results[0].values())).values()]
    width = 0.8
    x = np.arange(len(labels))
    colors = {"grounded": "#FFC75F", "ungrounded": "#5390D9"}
    z_score = 1.96
    error_bar_props = {"capsize": 5, "capthick": 2, "elinewidth": 2}

    plt.rcParams.update({
        "font.size": 24,
        "font.family": "serif",
    })

    # Make a subplot for each bucket
    fig, axs = plt.subplots(1, 3, figsize=(30, 8))

    for i, kind in enumerate(["hidden", "mlp", "attn"]):
        for j, (bucket, results) in enumerate([("grounded", grounded_results), ("ungrounded", unfaithful_results)]):
            ax = axs[i]

            effects, corrupted_probs, clean_probs = results

            # Plot the three kind bars for each token
            for t, label in enumerate(labels):
                bar = ax.bar(
                    x[t] + (width / 4) * (["grounded", "ungrounded"].index(bucket) * 2 - 1),
                    effects[kind][label].avg() * 100,
                    width / 2,
                    yerr=effects[kind][label].std() * z_score / np.sqrt(len(effects[kind][label])) * 100,
                    color=colors[bucket],
                    error_kw=error_bar_props,
                    label=bucket if t == 0 else "",  # Label only the first bar for legend
                )

            # Perform a statistical test (t-test) to compare grounded and ungrounded results
            p_values = [stats.ttest_ind(
                grounded_results[0][kind][label].to_array(),
                unfaithful_results[0][kind][label].to_array()
            ).pvalue for label in labels]

            # Color-code x-axis labels based on p-values
            label_colors = []
            for p_value in p_values:
                if p_value < 0.1:
                    label_colors.append('red')  # Significant difference
                else:
                    label_colors.append('black')  # Not significant

            # Set the ticks and labels
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=28)

            for xtick, color in zip(ax.get_xticklabels(), label_colors):
                xtick.set_color(color)

            # Set the limits for the y-axis to be the same for both subplots
            ax.set_ylim([0, 100])

            # Set title for each subplot
            ax.set_title(titles[kind], fontsize=35, pad=20)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            for v in [20, 40, 60, 80]:
                ax.axhline(y=v, linestyle='--', color='gray', linewidth=1, alpha=0.5)

            # Add legend
            if i == 0:  # Add legend in the first subplot only
                ax.set_ylabel('MGCT effect', fontsize=25)

            if i == 1:
                handles_leg, labels_leg = [], []
                for label_leg, color_leg in colors.items():
                    handles_leg.append(plt.Rectangle((0, 0), width, width, color=color_leg))
                    labels_leg.append(label_leg)

                ax.legend(handles_leg, labels_leg, loc='upper center', ncol=2, frameon=False)

    fig.suptitle(f"{model_names_conversion[model_name] if model_name in model_names_conversion else model_name} ({dataset_names_conversion[dataset_name] if dataset_name in dataset_names_conversion else dataset_name})", fontsize=45)

    plt.tight_layout()
    plt.subplots_adjust(left=0.10, right=0.90)

    # Add number of grounded and ungrounded facts represented
    save_path = save_path.replace(
        ".pdf", f"_grounded={len(grounded_results[1])}_ungrounded={len(unfaithful_results[1])}.pdf"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", format='pdf')
    plt.close()


def plot_main(args):
    # List all dir names (each dir is a dataset)
    dataset_names = os.listdir(args.causal_traces_dir)

    for dataset_name in dataset_names:
        if "unfiltered" in dataset_name:
            continue

        # List all models (each model is a dir)
        model_names = os.listdir(os.path.join(args.causal_traces_dir, dataset_name))
        print(model_names)
        for model_name in model_names:

            buckets = ["grounded", "unfaithful"]
            buckets_paths = [
                os.path.join(args.causal_traces_dir, dataset_name, model_name, f"{bucket}.json") for bucket in buckets
            ]

            if not all([os.path.exists(bucket_path) for bucket_path in buckets_paths]):
                continue

            results = []
            for bucket, bucket_path in zip(buckets, buckets_paths):
                results.append(group_results(read_json(bucket_path), bucket))

            plot_path = os.path.join(args.output_dir, dataset_name, f"{model_name}.pdf")
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            print(plot_path)
            plot(dataset_name, model_name, results[0], results[1], plot_path)


def main():
    plot_main(get_args())


if __name__ == "__main__":
    main()