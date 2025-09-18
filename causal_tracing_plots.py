import os
import sys
import numpy as np
from matplotlib import pyplot as plt
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

def process_facts2(target_token, facts, class_map, results, corrupted_probs, clean_probs):
    for processed_fact in facts:
        clas = class_map(processed_fact)
        corrupted_score = processed_fact["results"]["corrupted"][target_token]["probs"]
        clean_score = processed_fact["results"]["clean"][target_token]["probs"]

        # If there is a zero interval, skip the fact
        interval_to_explain = max(clean_score - corrupted_score, 0)

        corrupted_probs[clas].add(corrupted_score)
        clean_probs[clas].add(clean_score)

        for kind in ["hidden", "mlp", "attn"]:
            (
                avg_first_subject,
                avg_middle_subject,
                avg_last_subject,
                avg_first_after,
                avg_middle_after,
                avg_last_after,
            ) = results[clas][kind].values()

            tokens = processed_fact["results"]["tokens"]
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


def get_interval_to_explain(processed_fact, target_token):
    corrupted_score = processed_fact["results"]["corrupted"][target_token]["probs"]
    clean_score = processed_fact["results"]["clean"][target_token]["probs"]

    interval_to_explain = max(clean_score - corrupted_score, 0)
    return interval_to_explain


def group_results2(facts_grounded, facts_unfaithful, args):
    labels = ["subj-first", "subj-middle", "subj-last", "cont-first", "cont-middle", "cont-last"]
    num_classes = args.num_classes
    corrupted_probs = [Feature("corr") for _ in range(num_classes)]
    clean_probs = [Feature("clean") for _ in range(num_classes)]
    # Here results is a list of dictionaries, each dictionary contains the results for one class (i.e. label i.e. bucket), the bucket is decided in the process_facts2 function
    results = [
        {kind: {labels[i]: Feature(labels[i]) for i in range(6)} for kind in ["hidden", "mlp", "attn"]}
        for _ in range(num_classes)]
    if args.balance:
        # Some rel_lemma are not present in all sets.
        lemmauf = set(x["fact"]["rel_lemma"] for x in facts_unfaithful)
        lemmaf = set(x["fact"]["rel_lemma"] for x in facts_grounded)
        print("in facts_grounded not in facts_unfaithful", lemmaf.difference(lemmauf))
        print("in facts_unfaithful not in facts_grounded", lemmauf.difference(lemmaf))
        tempsuf = set(x["fact"]["subject"] for x in facts_unfaithful)
        tempsf = set(x["fact"]["subject"] for x in facts_grounded)
        # Templates not uniformly distributed, half of in unfaithful never appear in grounded
        print("num unique templates facts_unfaithful",
              len(tempsuf))
        print("num unique templates facts_grounded",
              len(tempsf))
        print("num templates in facts_grounded not in facts_unfaithful",
              len(tempsf.difference(tempsuf)))
        print("num templates in facts_unfaithful not in facts_grounded",
              len(tempsuf.difference(tempsf)))
        # Remove trivial samples
        trivial = tempsf.symmetric_difference(tempsuf)
        facts_unfaithful = [x for x in facts_unfaithful if x["fact"]["subject"] not in trivial]
        facts_grounded = [x for x in facts_grounded if x["fact"]["subject"] not in trivial]

    print("lens", len(facts_grounded), len(facts_unfaithful))
    facts_grounded = [x for x in facts_grounded if not get_interval_to_explain(x, "grounded_token") == 0]
    facts_unfaithful = [x for x in facts_unfaithful if not get_interval_to_explain(x, "unfaithful_token") == 0]
    print("lens after", len(facts_unfaithful), len(facts_grounded))
    process_facts2("unfaithful_token", facts_unfaithful,
                   lambda x: 0, results, corrupted_probs, clean_probs)
    ps = [x["results"]["clean"]["grounded_token"]["probs"] for x in facts_grounded]
    ls = np.linspace(0, 0.5, num_classes)
    if False:
        class_boundaries = np.quantile(ps, ls)[1:-1]
        print(ps, ls, class_boundaries)

        def classify(fact):
            return (1 if args.separate else 0) + np.digitize(fact["results"]["clean"]["grounded_token"]["probs"],
                                   class_boundaries) if num_classes > 2 else 1

    process_facts2("grounded_token", facts_grounded,
                   lambda x: 1, results, corrupted_probs, clean_probs)
    # print("corrupted_probs, clean_probs", [x.d for x in corrupted_probs], [x.d for x in clean_probs])
    vs = list(
        (x, i, len(x[0]["hidden"]["subj-first"])) for i, x in enumerate(zip(results, corrupted_probs, clean_probs)))
    print("class_c", [x[2] for x in vs])
    vs = [(x, i) for x, i, l in vs if l >= args.min_count]
    # in next experiment try ["grounded", "confidently grounded"]
    return [x for x, i in vs], [f"p:{i}" for x, i in vs]


def get_args(default = False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--causal_traces_dir", type=str, default="./causal_traces"
    )
    parser.add_argument(
        "--num_classes", type=int, default=2
    )

    parser.add_argument(
        "--balance", default=False, action="store_true",
    )
    parser.add_argument(
        "--separate", default=False, action="store_true",
    )
    parser.add_argument(
        "--min_count", type=int, default=0
    )
    parser.add_argument("--output_dir", type=str, default="./plots")

    return parser.parse_args([]) if default else parser.parse_args()


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
                if p_value < 0.05:
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
    os.makedirs(args.causal_traces_dir + "/f/f", exist_ok=True)
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

            results, labels = group_results2(read_json(buckets_paths[0]), read_json(buckets_paths[1]), args)
            print(results[0][0])

            plot_path = os.path.join(args.output_dir, dataset_name, f"{model_name}.pdf")
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            print(plot_path)
            plot(dataset_name, model_name, results[1], results[0], plot_path)


def main():
    plot_main(get_args(True))


if __name__ == "__main__":
    main()