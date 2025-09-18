import os

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, recall_score, f1_score, fbeta_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
from causal_tracing_whisper import *
from causal_tracing_plots import plot
def load_metrics(save_dir):
    with open(os.path.join(save_dir, "results.json"), "r") as file:
        results = json.load(file)
    return results


def save_metrics(results, feature_names, save_dir):
    with open(os.path.join(save_dir, "results.json"), "w") as file:
        json.dump(results, file, indent=4)

    if "feature_importances" in results:
        importances = results["feature_importances"]
        print("importances", importances)
        indices = np.argsort(importances)

        # Logic to determine the kind for colors
        colors = {"hidden": "grey", "mlp": "blue", "attn": "orange", "corr": "grey", "clean": "grey"}

        def determine_kind(verbose_name):
            for kind, color in colors.items():
                if kind in verbose_name.lower():
                    return color
            print(f"Unmatched feature: {verbose_name}")
            raise ValueError("Unknown feature kind.")

        bar_colors = [determine_kind(name) for name in feature_names]

        # Make the font size larger
        plt.rcParams.update({"font.size": 21})

        # Change the font family
        plt.rcParams["font.family"] = "serif"

        plt.figure(figsize=(15, 15))
        plt.barh(
            range(len(indices)),
            [importances[i] for i in indices],
            align="center",
            color=[bar_colors[i] for i in indices],
            edgecolor="white",
        )
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "feature_importances.png"))
        plt.close()


def save_decision_tree_plot(tree, feature_names, class_names, save_dir):
    plt.figure(figsize=(20, 10))
    plot_tree(tree, feature_names=feature_names, class_names=class_names, filled=True)
    plt.savefig(os.path.join(save_dir, "decision_tree.png"))
    plt.close()


def plot_metrics_comparison(metrics_by_model, save_dir):
    """
    metrics_by_model: dict, keys are model names (like 'Logistic Regression', 'DecisionTree', 'XGBoost') and values are
                      dictionaries of metrics (keys are metric names, values are metric values)
    save_dir: directory where plots will be saved
    """
    model_colors = {"LogisticRegression": "grey", "DecisionTree": "orange", "DecisionTreeSmall": "green",
                    "XGBoost": "blue"}

    # Validate that all models in metrics_by_model are known
    for model in metrics_by_model:
        if model not in model_colors:
            raise Exception(f"Unknown model: {model}")

    n_models = len(metrics_by_model)
    n_metrics = len(metrics_by_model[next(iter(metrics_by_model))])

    # Set bar width, distance between bars in a group, and positions
    bar_width = 0.2
    distance = 0.05  # distance between bars in a group
    r1 = np.arange(n_metrics)  # positions for first model
    r2 = [x + bar_width + distance for x in r1]  # positions for second model
    r3 = [x + bar_width + distance for x in r2]  # positions for third model

    # Make the font size larger
    plt.rcParams.update({"font.size": 21})

    # Change the font family
    plt.rcParams["font.family"] = "serif"

    plt.figure(figsize=(15, 10))

    # Plotting bars for each model
    all_metric_values = []
    for idx, (model, metrics) in enumerate(metrics_by_model.items()):
        metric_values = [metrics[metric] for metric in metrics]
        all_metric_values.extend(metric_values)
        positions = [r1, r2, r3][idx]
        plt.bar(positions, metric_values, color=model_colors[model], width=bar_width, edgecolor="white", label=model)

    # Adjust y-axis limit
    plt.ylim(bottom=min(all_metric_values) * 0.9)

    plt.xlabel("Metrics")
    plt.ylabel("Score")
    xtick_positions = [r2[i] for i in range(n_metrics)]  # Averages of r1 and r2 positions
    plt.xticks(xtick_positions, list(metrics_by_model[next(iter(metrics_by_model))]))

    # Place the legend outside the plot on the right
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "all_metrics_comparison.png"), bbox_inches="tight")
    plt.close()

def train_and_save(models, train_data, test_data, feature_names, class_names, grounded_results, unfaithful_results, args, seed, replot_only=False, target_acc = 0.6):
    save_dir = get_output_dir()
    def score_min_tree(estimator, X_test, y_test, **score_params):
        return -np.abs(target_acc-estimator.score(X_test, y_test))
    plt.rcParams["font.size"] = max(1, plt.rcParams["font.size"])

    metrics_by_model = {}

    for model_name, model_info in models.items():
        model_save_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)

        if not replot_only:
            X_train, y_train, inds_train = train_data
            X_test, y_test, inds_test = test_data

            if "random_state" in model_info["model"].get_params():
                model_info["model"].set_params(random_state=seed)

            best_clas = None
            best_f = None

            all_feat_removed = set()
            for x in range(1000):
                all_feat_removed.add(tuple(np.random.choice(np.arange(X_train.shape[1]), size=max(1, X_train.shape[1] - 6), replace=False).tolist()))

            for feat_removed in list(all_feat_removed)[:100]:
                X_train1 = X_train.copy()
                X_train1[:, feat_removed] = np.nan
                clf = GridSearchCV(model_info["model"], model_info["param_grid"], cv=3, verbose=0, scoring=lambda estimator, X_test, y_test : -np.abs(target_acc-fbeta_score(y_test, estimator.predict(X_test), beta=2)))
                clf.fit(X_train1[:-10], y_train[:-10])
                y_pred = clf.predict(X_train[-10:])
                f = fbeta_score(y_train[-10:], y_pred, beta=2, average='weighted')
                if best_f is None or f > best_f:
                    best_f = f
                    best_clas = clf
                    best_feat = [feature_names[x] for x in sorted(set(range(len(feature_names))).difference(set(feat_removed)))]

            clf = best_clas
            y_pred = clf.predict(X_test)
            y_train_pred = clf.predict(X_train)
            y_train_proba = clf.predict_proba(X_train)[:, 1]
            with open(f"{model_name}_confusion_matrix.json", "w") as f:
                json.dump(confusion_matrix(y_test, y_pred).tolist(), f)
            results = {
                "train": {
                    "accuracy": accuracy_score(y_train, y_train_pred),
                    "precision": precision_score(y_train, y_train_pred, average='weighted'),
                    "recall": recall_score(y_train, y_train_pred, average='weighted'),
                    "f1_score": f1_score(y_train, y_train_pred, average='weighted'),
                    # "roc_auc": roc_auc_score(y_train, y_train_proba,
                    #                         multi_class = "ovr", average='weighted'),
                },
                "test": {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average='weighted'),
                    "recall": recall_score(y_test, y_pred, average='weighted'),
                    "f1_score": f1_score(y_test, y_pred, average='weighted'),
                    # "roc_auc": roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1],
                    #                         multi_class = "ovr", average='weighted'),
                },
                "best_hyperparameters": clf.best_params_,
                "used_features": best_feat,
                "train_size": len(y_train),
                "test_size": len(y_test)
            }

            if hasattr(clf.best_estimator_, "feature_importances_"):
                # If there is an importance type attribute, print it
                if hasattr(clf.best_estimator_, "importance_type"):
                    print(f"Feature importances: {clf.best_estimator_.importance_type}")
                results["feature_importances"] = list(clf.best_estimator_.feature_importances_)
                results["feature_importances"] = [float(val) for val in results["feature_importances"]]

            if isinstance(clf.best_estimator_, LogisticRegression):
                # Taking the absolute values of the coefficients
                results["feature_importances"] = [float(abs(val)) for val in clf.best_estimator_.coef_.flatten()]
            try:
                save_metrics(results, feature_names, model_save_dir)
            except:
                pass
            if isinstance(model_info["model"], (DecisionTreeClassifier,)):
                save_decision_tree_plot(clf.best_estimator_, feature_names, class_names, model_save_dir)
                tree = clf.best_estimator_.tree_
                goes_left = X_test[:, tree.feature[0]] <= tree.threshold[0]
                goes_left_train = X_train[:, tree.feature[0]] <= tree.threshold[0]
                def double_apply(d, func):
                    return {k: {k2: func(v2) for k2, v2 in v.items()} for k,v in d.items()}

                plot("first_left_train", "l",
                     double_apply(grounded_results, lambda x: x.indexer(inds_train[goes_left_train & (y_train > 0)])),
                     double_apply(unfaithful_results, lambda x: x.indexer(inds_train[goes_left_train & (y_train == 0)])),
                     model_save_dir + "/first_left_train.pdf", args)
                plot("first_right_train", "l",
                     double_apply(grounded_results, lambda x: x.indexer(inds_train[~goes_left_train & (y_train > 0)])),
                     double_apply(unfaithful_results, lambda x: x.indexer(inds_train[~goes_left_train & (y_train == 0)])),
                     model_save_dir + "/first_right_train.pdf", args)
                plot("first_left", "l", double_apply(grounded_results, lambda x: x.indexer(inds_test[goes_left & (y_test > 0)])), double_apply(unfaithful_results, lambda x: x.indexer(inds_test[goes_left & (y_test == 0)])), model_save_dir+"/first_left.pdf", args)
                plot("first_right", "l", double_apply(grounded_results, lambda x: x.indexer(inds_test[~goes_left & (y_test > 0)])), double_apply(unfaithful_results, lambda x: x.indexer(inds_test[~goes_left & (y_test == 0)])), model_save_dir+"/first_right.pdf", args)
        else:
            results = load_metrics(model_save_dir)
            save_metrics(results, feature_names, model_save_dir)

        metrics_by_model[model_name] = results["test"]

    plot_metrics_comparison(metrics_by_model, save_dir)


def process_facts2(target_token, facts, class_map, results, corrupted_probs, clean_probs, tokenizer):
    for processed_fact in facts:
        clas = class_map(processed_fact)
        corrupted_score = processed_fact["results"]["corrupted"][target_token]["probs"]
        clean_score = processed_fact["results"]["clean"][target_token]["probs"]

        # If there is a zero interval, skip the fact
        interval_to_explain = max(clean_score - corrupted_score, 0)

        corrupted_probs[clas].add(corrupted_score)
        clean_probs[clas].add(clean_score)

        for kind in ["hidden", "mlp", "attn"]:
            d = results[clas][kind]

            tokens = processed_fact["results"]["tokens"]
            tokens_sorted = sorted(tokens, key=lambda x: x["pos"])
            token_effects_subj = []
            token_effects_cont = []
            for token in tokens_sorted:
                interval_explained_average = 0
                for layer in token[kind]:
                    interval_explained_average += max(token[kind][layer][target_token]["probs"] - corrupted_score,
                                                      0) / len(token[kind])
                token_effect = min(interval_explained_average / interval_to_explain, 1)
                if "subject_pos" in token:
                    token_effects_subj.append((token["pos"], token_effect, token["val"]))
                else:
                    token_effects_cont.append((token["pos"], token_effect, token["val"]))

            while token_effects_subj[-1][-1] == " ":
                token_effects_subj = token_effects_subj[:-1]

            while token_effects_cont[-1][-1] == " ":
                token_effects_cont = token_effects_cont[:-1]
            def set_one(k, v):
                if k not in d:
                    d[k] = Feature(k)
                d[k].add(v)

            def set_else_0(c,k,f):
                if c:
                    set_one(k, f())
                else:
                    set_one(k, 0.0)

            if True:
                feats = [(len(token_effects_subj) > 1, "subj-first", lambda : token_effects_subj[0][1]),
                (len(token_effects_subj) > 3, "subj-second-first", lambda : token_effects_subj[1][1]),
                (len(token_effects_subj) > 4, "subj-middle", lambda : float(np.mean([x[1] for x in token_effects_subj[2:-2]]))),
                (len(token_effects_subj) > 2, "subj-second-last", lambda : token_effects_subj[-2][1]),
                (len(token_effects_subj) > 0, "subj-last", lambda : token_effects_subj[-1][1]),
                (len(token_effects_cont) > 1, "cont-first", lambda : token_effects_cont[0][1]),
                (len(token_effects_subj) > 3, "cont-second-first", lambda : token_effects_subj[1][1]),
                (len(token_effects_cont) > 4, "cont-middle", lambda : float(np.mean([x[1] for x in token_effects_cont[2:-2]]))),
                (len(token_effects_cont) > 2, "cont-second-last", lambda : token_effects_cont[-2][1]),
                (len(token_effects_cont) > 0, "cont-last", lambda : token_effects_cont[-1][1])]
            else:
                feats = [(len(token_effects_subj) > 0, "subj-last", lambda: token_effects_subj[-1][1]),
                (len(token_effects_cont) > 0, "cont-first", lambda: token_effects_cont[0][1]),
                (len(token_effects_cont) > 0, "cont-middle",
                 lambda: float(np.mean([x[1] for x in token_effects_cont[1:-1]]))),
                (len(token_effects_cont) > 0, "cont-last", lambda: token_effects_cont[-1][1])]
            assert len(feats) == len(set(x[1] for x in feats))
            for c, k, f in feats:
                set_else_0(c, k, f)
            assert len(d) > 0


def filter_facts(processed_fact, target_token):
    corrupted_score = processed_fact["results"]["corrupted"][target_token]["probs"]
    clean_score = processed_fact["results"]["clean"][target_token]["probs"]

    interval_to_explain = max(clean_score - corrupted_score, 0)
    return interval_to_explain == 0


def group_results2(facts_grounded, facts_unfaithful, tokenizer, args):
    num_classes = args.num_classes
    corrupted_probs = [Feature("corr") for _ in range(num_classes)]
    clean_probs = [Feature("clean") for _ in range(num_classes)]
    # Here results is a list of dictionaries, each dictionary contains the results for one class (i.e. label i.e. bucket), the bucket is decided in the process_facts2 function
    results = [
        {kind: {} for kind in ["hidden", "mlp", "attn"]}
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

    facts_grounded = [x for x in facts_grounded if not filter_facts(x, "grounded_token")]
    facts_unfaithful = [x for x in facts_unfaithful if not filter_facts(x, "unfaithful_token")]
    print("fact_lengths", len(facts_unfaithful), len(facts_grounded))
    process_facts2("unfaithful_token", facts_unfaithful,
                   lambda x: 0, results, corrupted_probs, clean_probs, tokenizer)
    print([x["fact"]["object"] for x in facts_grounded if filter_facts(x, "grounded_token")])
    ps = [x["results"]["clean"]["grounded_token"]["probs"] for x in facts_grounded]
    ls = np.linspace(0, 0.5, num_classes)
    class_boundaries = np.quantile(ps, ls)[1:-1]
    #print(ps, ls, class_boundaries)

    def classify(fact):
        return (1 if args.separate else 0) + np.digitize(fact["results"]["clean"]["grounded_token"]["probs"],
                                                         class_boundaries) if num_classes > 2 else 1

    process_facts2("grounded_token", facts_grounded, lambda x: 1 if True else classify(x)
                   , results, corrupted_probs, clean_probs, tokenizer)
    # print("corrupted_probs, clean_probs", [x.d for x in corrupted_probs], [x.d for x in clean_probs])
    vs = list(
        (x, i, len(next(iter(x[0]["hidden"].values())))) for i, x in enumerate(zip(results, corrupted_probs, clean_probs)) if len(x[0]["hidden"]) > 0)
    print("class_c", [x[2] for x in vs])
    vs = [(x, i) for x, i, l in vs if l >= args.min_count]
    # in next experiment try ["grounded", "confidently grounded"]
    return [x for x, i in vs], [f"p:{i}" for x, i in vs]


def generate_datasets2(buckets,
                       train_ratio=0.8
                       ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray], Any]:
    logger = get_logger()
    feature_names = [
        f"{kind}-{feature}" for kind, features in buckets[0][0].items() for feature in features.keys()
    ]

    logger.info(f"Feature names: {feature_names}")

    all_samples = []
    all_labels = []
    all_inds = []

    all_label_samples = []
    for label, bucket_results in enumerate(buckets):
        all_label_samples.append([])
        kinds_results, corr_probs, clean_probs = bucket_results
        # This is the correct number of samples the feature adding is done in a wierd way but there is always
        # exactly one value in each feature for each sample
        num_samples = len(corr_probs) - 1

        logger.info("Number of samples: {}".format(num_samples))
        print(label, [[(kind_name, feature_name, len(feature_results),
                        feature_results.avg(), feature_results.std())
                       for feature_name, feature_results in kind_results.items()] for kind_name, kind_results in
                      kinds_results.items()])

        for i in range(num_samples):
            candidate_example = tuple(
                feature_results.get(i)
                for kind_results in kinds_results.values()
                for feature_results in kind_results.values()
            )

            if any([feature is None for feature in candidate_example]):
                continue

            all_label_samples[-1].append((candidate_example, label, i))
    assert len(set([tuple(sorted(x)) for x in all_label_samples])) == len(all_label_samples)
    print("all_label_samples", len(all_label_samples[0]), len(all_label_samples[1]))
    minl = min(len(x) for x in all_label_samples)
    for current_label_samples in all_label_samples:
        np.random.shuffle(current_label_samples)
        all_samples.extend([sample[0] for sample in current_label_samples[:minl]])
        all_labels.extend([sample[1] for sample in current_label_samples[:minl]])
        all_inds.extend([sample[2] for sample in current_label_samples[:minl]])

    # Convert all_samples and all_labels to np arrays
    all_samples_array = np.array(all_samples)
    all_labels_array = np.array(all_labels)
    all_inds_array = np.array(all_inds)

    # Calculate lengths for each split
    total_size = len(all_samples_array)
    train_size = int(total_size * train_ratio)

    # Shuffle and split the dataset
    indices = np.arange(total_size)
    np.random.shuffle(indices)

    train_dataset = (all_samples_array[indices[:train_size]], all_labels_array[indices[:train_size]], all_inds_array[indices[:train_size]])
    test_dataset = (all_samples_array[indices[train_size:]], all_labels_array[indices[train_size:]], all_inds_array[indices[train_size:]])
    print(test_dataset[0].shape, test_dataset[0].shape, feature_names)
    #print(list(zip(test_dataset[0][:, -1], test_dataset[1])))
    return train_dataset, test_dataset, feature_names


"""
"param_grid": {
                "max_depth": [None, 5, 10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
"param_grid": {
                "max_depth": [3, 4],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
"""


def train_detector(args, models):
    buckets = ["grounded", "unfaithful"]
    buckets_paths = [
        os.path.join(args.causal_traces_dir, args.dataset_name, args.model_name, f"{bucket}.json") for bucket in buckets
    ]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai/whisper-base.en", force_download=False,
                                              add_bos_token=True)
    print("read_lengths", [len(read_json(x)) for x in buckets_paths])
    results, buckets = group_results2(read_json(buckets_paths[0]),
                                      read_json(buckets_paths[1]),
                                      tokenizer, args)

    # If we are only including certain kinds, filter the kinds
    if args.kinds_to_include is not None:
        results = [
            (
                {kind: bucket_results[0][kind] for kind in bucket_results[0] if kind in args.kinds_to_include},
                bucket_results[1],
                bucket_results[2],
            )
            for bucket_results in results
        ]

    # If we are only including certain features, filter the features
    if args.features_to_include is not None:
        results = [
            (
                {
                    kind: {
                        feature: bucket_results[0][kind][feature]
                        for feature in bucket_results[0][kind]
                        if feature in args.features_to_include
                    }
                    for kind in bucket_results[0]
                },
                bucket_results[1],
                bucket_results[2],
            )
            for bucket_results in results
        ]
    #print([[[len(z) for z in y] for y in x[0].values()] for x in results])
    # Generate the datasets
    train_data, test_data, feature_names = generate_datasets2(
        results,
        train_ratio=args.train_ratio
    )
    print(len(train_data), len(train_data[0]), train_data[1])
    print(len(test_data), len(test_data[0]), test_data[1])
    # Train the models and save the results
    train_and_save(models, train_data, test_data, feature_names, buckets, results[1][0], results[0][0], args, seed=args.seed, target_acc=args.target_acc)


def plot(dataset_name, model_name, grounded_results, unfaithful_results, save_path, args):
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

    labels = [feature.get_name() for feature in next(iter(grounded_results.values())).values()]
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

    for i, kind in enumerate(args.kinds_to_include):
        for j, (bucket, results) in enumerate([("grounded", grounded_results), ("ungrounded", unfaithful_results)]):
            ax = axs[i]

            effects = results

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
                grounded_results[kind][label].to_array(),
                unfaithful_results[kind][label].to_array()
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

    fig.suptitle(
        f"{model_names_conversion[model_name] if model_name in model_names_conversion else model_name} ({dataset_names_conversion[dataset_name] if dataset_name in dataset_names_conversion else dataset_name})",
        fontsize=45)

    plt.tight_layout()
    plt.subplots_adjust(left=0.10, right=0.90)

    # Add number of grounded and ungrounded facts represented
    save_path = save_path.replace(
        ".pdf", f"_grounded={len(next(iter(next(iter(grounded_results.values())).values())))}_ungrounded={len(next(iter(next(iter(unfaithful_results.values())).values())))}.pdf"
    )
    print(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", format='pdf')
    plt.close()


class Namespace2:
    def __init__(self):
        self.causal_traces_dir = "./causal_traces"
        self.dataset_name = "transl"
        self.model_name = "LLaMa"
        self.output_dir = "out"
        self.balance = False
        features = "all"
        if features == "all":
            self.features_to_include = None
        elif features == "cont_last_removed":
            self.features_to_include = ["subj-first", "subj-middle", "subj-last", "cont-first", "cont-middle"]
        elif features == "cont_middle_last_removed":
            self.features_to_include = ["subj-first", "subj-middle", "subj-last", "cont-first"]
        elif features == "only_important":
            self.features_to_include = ["subj-last", "cont-first",
                                        "cont-last"]
        elif features == "only_important_no_cont_last":
            self.features_to_include = ["subj-first", "subj-middle", "subj-last", "cont-first"]
        self.kinds_to_include = ["hidden", "mlp"]
        self.train_ratio = 0.8
        self.ablation_only_clean = False
        self.ablation_include_corrupted = False
        self.seed = 2
        self.num_classes = 2
        self.min_count = 5
        self.separate = False
        self.target_acc = 0.755


def main2(models):
    sp = "./specific_runs/run4"
    p = './causal_traces/transl/LLaMa'
    names = ['grounded.json', 'unfaithful.json']
    for name in names:
        ds = []
        for x in os.listdir(sp):
            filep = os.path.join(sp, x, name)
            if os.path.exists(filep):
                with open(filep) as f:
                    ds.extend(json.load(f))
        with open(os.path.join(p, name), "w") as f:
            json.dump(ds, f, indent=4)
    os.makedirs(p, exist_ok=True)
    args = Namespace2()
    freeze_args(args)
    set_seed_everywhere(args.seed)
    train_detector(args, models)


if __name__ == "__main__":
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000),
            "param_grid": {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
        } if False else {},
        "DecisionTreeSmall": {
            "model": DecisionTreeClassifier(),
            "param_grid": {
                "max_depth": [2, 3],
                "min_samples_split": [2, 4, 8],
                "min_samples_leaf": [2, 4, 8],
                #"ccp_alpha": [0, 0.005, 0.02, 0.03, 0.05, 0.08]
            },
        },
    }
    if False:
        models["DecisionTree"] = {
            "model": DecisionTreeClassifier(),
            "param_grid": {
                "max_depth": [4, 5, 6, 8, 10],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "ccp_alpha": [0.08, 0.1, 0.15]
            },
        }
    models = dict([x for x in models.items() if len(x[1]) > 0])
    xgboostmode = "None"
    if xgboostmode == "full":
        models["XGBoost"] = {
            "model": xgb.XGBClassifier(
                use_label_encoder=False, eval_metric="logloss", device="cuda", importance_type="total_gain"
            ),
            "param_grid": {
                "learning_rate": [0.01, 0.05, 0.1],
                "n_estimators": [100, 200, 500],
                "max_depth": [3, 5, 7, 10],
                "subsample": [0.8, 0.9, 1.0],
            },
        }
    elif xgboostmode == "small":
        models["XGBoost"] = {
            "model": xgb.XGBClassifier(
                use_label_encoder=False, eval_metric="logloss", device="cuda", importance_type="total_gain"
            ),
            "param_grid": {
                "learning_rate": [0.01, 0.05, 0.1],
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "subsample": [0.8],
            },
        }
    else:
        pass

    main2(models)
    # !rm -r LLama

    for model_name in models:
        with open(f"{model_name}_confusion_matrix.json") as f:
            confm = json.load(f)
        disp = ConfusionMatrixDisplay(confusion_matrix=np.array(confm))
        disp.plot()
        plt.savefig(f"{model_name}_confusion_matrix.png")
