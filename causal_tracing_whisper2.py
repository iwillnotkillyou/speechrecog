from itertools import chain
import xgboost as xgb
import logging
from itertools import chain
from itertools import product

import scipy
import sklearn
import xgboost as xgb
from sklearn.metrics import ConfusionMatrixDisplay, fbeta_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from causal_tracing_whisper import *
from causal_tracing_whisper1 import make_model, Namespace1


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
                    "XGBoost": "blue", "KNN": "blue"}

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

def run_LLM(clf, end_to_end_test_data):
    ns = Namespace1()
    language_model, tokenizer = make_model(ns)
    resultsl = []
    np.random.seed(42)

    def flatten(data):
        return list(zip([tuple(feature_results.get(i)
                               for kind_results in data[0].values()
                               for feature_results in kind_results.values()) for i in range(len(data[3]))],
                        data[3]))

    def longest_match_any_startf(cands, ref):
        best_c = 0
        best_l = 0
        print(cands, ref)
        words_ref = ref.strip(". ").split(" ")
        for cand in range(len(cands)):
            words_cand = cands[cand].strip(". ").split(" ")
            for i in range(min(len(words_ref), len(words_cand))):
                for j in range(min(len(words_ref), len(words_cand))):
                    if words_ref[i:i + len(words_cand) - j] == words_cand[0:len(words_cand) - j]:
                        if len(words_cand) - j > best_l:
                            best_l = len(words_cand) - j
                            best_c = cand
        print(best_c)
        return best_c

    def longest_match_any_start(cands, ref):
        print(cands, ref)
        best_c = 10
        for i in range(1, len(ref)):
            l = [x for x in range(len(cands)) if
                 ref[:-i].strip(". ").lower().endswith(cands[x].strip(". ").lower())]
            if len(l) > 0:
                best_c = l[0]
                break
        print(best_c)
        return best_c

    def extract_object(x):
        if "unfaithful_token" in x[1]["results"]["corrupted"]:
            return x[1]["fact"]["fact_parent"]["object"]
        else:
            return x[1]["fact"]["object"]


    l = list(chain.from_iterable([flatten(x) for x in end_to_end_test_data]))
    oldlenl = len(l)
    # l = [x for x in l if longest_match_any_start([y[1] for y in x[1]["fact"]["rel_lemma"]], x[1]["fact"]["query"] + extract_object(x)) == 0]
    print("len(l)", len(l), oldlenl)
    np.random.shuffle(l)
    ids, inds = np.unique([int(y[1]["fact"]["group_id"]) for y in l], return_index=True)
    ids = ids[np.argsort(inds)]
    for x in ids[:len(ids) // 2]:
        x = [y for y in l if int(y[1]["fact"]["group_id"]) == x]
        for y in x[:1]:
            fact = y[1]["fact"]
            context = fact["fact_paragraph"]
            answers_list = fact["rel_lemma"]

            def get_prob(string, model, tokenizer, device):
                encoder_input_text, decoder_input_text = string.split("<pad>")
                encoder_text = tokenizer.apply_chat_template([{"role": "user", "content": encoder_input_text}],
                                                             tokenize=False, add_generation_prompt=True)
                print(encoder_text)
                from transformers import Text2TextGenerationPipeline
                pipe = Text2TextGenerationPipeline(model, tokenizer,
                                                   dtype=torch.bfloat16,
                                                   device_map=device,
                                                   )
                encoder_ids = pipe.preprocess(encoder_text)
                encoder_output = model.model.encoder(
                    input_ids=encoder_ids["input_ids"].to(device)
                )
                labels = torch.tensor([tokenizer.encode(decoder_input_text, add_special_tokens=False)],
                                      device=device)
                output_dict = model.forward(labels=labels, encoder_outputs=encoder_output)
                prob = np.prod([torch.softmax(output_dict.logits[0][x], -1)[labels[0][x]].item() for x in
                                range(1, len(output_dict.logits[0]))])
                print(prob, [{"t": x[0], "p": x[1]} for x in zip(tokenizer.convert_ids_to_tokens(labels[0]), [
                    list(zip(tokenizer.convert_ids_to_tokens(y), x.tolist())) for x, y in
                    zip(*torch.softmax(output_dict.logits[0], -1).topk(2, -1))])])
                return float(prob)

            order = np.random.permutation(len(answers_list))
            alphabet = [chr(x) for x in range(ord('a'), ord('a') + len(answers_list))]
            optionsl = [f"**{alphabet[ind]}**) {answers_list[i][1]}" for ind, i in enumerate(order)]
            options = "\n".join(optionsl)
            contextl = context.split("\\n")
            context = f"{contextl[0]}\n\nThe most likely answer is **letter**.\n\n{options}\n\n{contextl[1]}"
            w_probs = [(answers_list[i][0], answers_list[i][1],
                        get_prob(context + "<pad>**" + alphabet[ind], language_model, tokenizer, "cuda")) for ind, i in
                       enumerate(order)]
            w_probs = sorted(w_probs, key=lambda x: x[2], reverse=True)
            assert all([x[2] > 0.001 for x in w_probs])
        result_tracing = clf.predict(np.array([y[0] for y in x])).tolist()

        def improve(result_tracing, tracing_order, original_probs):
            matching = [
                longest_match_any_start([y[1] for y in original_probs], y[1]["fact"]["query"] + extract_object(y)) for y
                in tracing_order]
            print("matching", (
            matching, result_tracing, [y[1]["fact"]["query"] + extract_object(y) for y in tracing_order],
            [y[1] for y in original_probs]))
            possible_improves = [x for x in
                                 range(len(original_probs)) if
                                 x not in matching or result_tracing[matching.index(x)] != 0]
            if len(possible_improves) == 0:
                r = original_probs[0][0]
            else:
                r = original_probs[possible_improves[0]][0]
            print("possible_improves", possible_improves, r)
            return float(r)

        improved_label = improve(result_tracing, x, w_probs)
        small_model_label_improved_label = improve(result_tracing, x, answers_list)

        resultsl.append(
            {"gold_label": 0, "large_model_label": w_probs[0][0], "improved_label": improved_label,
             "small_model_label": answers_list[0][0],
             "small_model_label_improved_label": small_model_label_improved_label, "large_probs": w_probs,
             "fact": fact})

    for x in l[:0]:
        fact = x[1]["fact"]
        answers_list = fact["rel_lemma"]
        result_tracing = clf.predict(np.array([x[0]]))
        print((answers_list[0][1], fact["query"] + extract_object(x)), result_tracing[0],
              "unfaithful_token" in x[1]["results"]["corrupted"])
        if result_tracing[0] == 0:
            small_model_label_improved_label = fact["rel_lemma"][1][0]
        else:
            continue
        resultsl.append(
            {"gold_label": 0, "small_model_label": fact["rel_lemma"][0][0],
             "small_model_label_improved_label": small_model_label_improved_label, "fact": fact})
    assert len(resultsl) > 0
    return resultsl

def train_and_save(models, dataset, generate_wrapper, class_names, grounded_results, unfaithful_results, args, seed,
                   replot_only=False, target_acc=0.6):
    save_dir = get_output_dir()

    def score_min_tree(estimator, X_test, y_test, **score_params):
        return -np.abs(target_acc - estimator.score(X_test, y_test))

    plt.rcParams["font.size"] = max(1, plt.rcParams["font.size"])

    metrics_by_model = {}

    def filter(da, func):
        data = {key: {k: zip(v[k].d, da[1].d, da[2].d, da[3]) for k in v} for key, v in da[0].items()}
        data = {key: {k: list(zip(*[x for x in v[k] if func(x)])) for k in v} for key, v in data.items()}
        data =  {key: {k: Feature.from_list(v[k][0], k) for k in v} for key, v in data.items()}, Feature.from_list(
            list(list(data.values())[0].values())[0][1], da[1].name), Feature.from_list(
            list(list(data.values())[0].values())[0][2], da[2].name), list(list(data.values())[0].values())[0][3]
        return data

    print("len(dataset)", len(dataset[0][1].d))
    try:
        end_to_end_test_data = [filter(x, lambda x: x[3]["id"] is not None and int(x[3]["id"]) >= args.id_for_end_to_end) for x in dataset]
    except IndexError as e:
        logging.warning(f"IndexError when filtering end-to-end test data: {e}")
        end_to_end_test_data = None
    dataset = [filter(x, lambda x: x[3]["id"] is None or int(x[3]["id"]) < args.id_for_end_to_end) for x in dataset]
    print("len(end_to_end_test_data)", len(dataset[0][1].d), len(end_to_end_test_data[0][1].d) if end_to_end_test_data is not None else 0)
    train_data, test_data, holdout_data, feature_names = generate_wrapper(dataset, 0.8, 0, None)
    X_train, y_train, inds_train, weights_train = train_data
    X_test, y_test, inds_test, weights_test = test_data
    for model_name, model_info in models.items():
        model_save_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)

        if "random_state" in model_info["model"].get_params():
            model_info["model"].set_params(random_state=seed)

        best_clas = None
        best_f = None
        best_sets = None
        best_params = None
        best_feat = None

        all_feat_removed = set()
        for x in range(args.num_feat_tries*5):
            if args.random_feat_tries:
                all_feat_removed.add(tuple(sorted(
                np.random.choice(np.arange(X_train.shape[1]), size=np.random.randint(1, X_train.shape[1]//2) if X_train.shape[1] > 3 else 1, replace=False).tolist())) if X_train.shape[1] > 2 else tuple())
            else:
                all_feat_removed.add(tuple(range(np.random.randint(3, X_train.shape[1]), X_train.shape[1])))
        print("all_feat_removed", all_feat_removed)
        all_feat_removed = list(all_feat_removed)[:args.num_feat_tries] if args.num_feat_tries > 1 else [tuple()]
        np.random.shuffle(all_feat_removed)
        params = product(all_feat_removed,
                         np.linspace((args.tree_nodes[0], args.tree_nodes[0]*2), (args.tree_nodes[1], args.tree_nodes[1]*2), args.num_tree_nodes, endpoint=True) if args.num_tree_nodes > 1 else [args.tree_nodes],
                         np.linspace(args.train_ratio / 2, args.train_ratio, args.num_train_sizes, endpoint=True) if args.num_train_sizes > 1 else [args.train_ratio],
                         np.linspace(0, 1, 10, endpoint=True) if args.additional_weight > 0 else [0],
                         np.linspace(0, 0.5, 5, endpoint=True) if args.PCA_quantile_threshold is not None else [0])
        # params = product(list(all_feat_removed)[:10], [0.5], [0], [0])
        param_scores = {}
        for feat_removed, tree_nodes, train_ratio, additional_weight, PCA_quantile_threshold in params:
            r = generate_wrapper(dataset, train_ratio, additional_weight,
                                 PCA_quantile_threshold if PCA_quantile_threshold > 0 else None)
            if r is None:
                logging.warning(f"No dataset.")
                continue
            train_data, test_data, holdout_data, feature_names = r
            X_train, y_train, inds_train, weights_train = train_data
            X_test, y_test, inds_test, weights_test = test_data
            X_holdout, y_holdout, inds_holdout, weights_holdout = holdout_data
            X_train1 = X_train.copy()
            X_train1[:, feat_removed] = 0
            beta = 1
            best_score = -10000
            best_estimator = None
            l = X_holdout.shape[0]
            def modify_estimator(estimator, X_t, y_t):
                if isinstance(model_info["model"], (DecisionTreeClassifier,)):
                    path = estimator.cost_complexity_pruning_path(X_holdout[:l], y_holdout[:l], weights_holdout[:l])
                    best_feature = np.argmax(estimator.feature_importances_)
                    ccp_alphas, impurities = path.ccp_alphas, path.impurities
                    counts = []
                    for ccp_alpha in ccp_alphas:
                        counts.append(estimator.tree_.node_count)
                        if estimator.tree_.node_count < tree_nodes[1]:
                            break
                        estimator = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
                        estimator.fit(X_t, y_t)
                        if estimator.tree_.node_count < tree_nodes[1]:
                            break
                    print("ccp_alphas", list(zip(counts, ccp_alphas)))

                return estimator

            def calc_score(estimator, X_t, y_t):
                nonlocal best_score, best_estimator
                estimator = modify_estimator(estimator, X_t, y_t)
                if isinstance(model_info["model"], (DecisionTreeClassifier,)) and estimator.tree_.node_count < tree_nodes[0]:
                    logging.info(f"Skipping estimator with too few nodes: {estimator.tree_.node_count} < {tree_nodes[0]}")
                    return -10000
                score = (-np.abs(target_acc - fbeta_score(y_t, estimator.predict(X_t), beta=beta, average='weighted',
                                                            zero_division=0.0)))
                if score > best_score:
                    best_score = score
                    best_estimator = estimator
                return score

            clf = GridSearchCV(model_info["model"], model_info["param_grid"], cv=2, verbose=0,
                               scoring=calc_score)
            try:
                if args.balance_w:
                    clf.fit(X_train1[weights_train > 0], y_train[weights_train > 0],
                        sample_weight=weights_train[weights_train > 0])
                else:
                    clf.fit(X_train1[weights_train > 0], y_train[weights_train > 0])
            except ValueError as e:
                logging.warning(f"ValueError during model fitting for model {model_name}: {e}")
                continue
            y_holdout_pred = clf.predict(X_holdout)
            fscore = fbeta_score(y_holdout, y_holdout_pred, beta=beta, average='weighted', zero_division=0.0,
                                 sample_weight=weights_holdout if args.balance_w else None)

            if (best_f is None or fscore > best_f) and fscore < args.holdout_score_limit and best_score > -10000 and best_estimator is not None:
                best_f = fscore
                clf.best_estimator_ = best_estimator
                best_clas = clf
                best_feat = [feature_names[x] for x in
                             sorted(set(range(len(feature_names))).difference(set(feat_removed)))]
                best_sets = (train_data, test_data, holdout_data, feature_names)
                best_params = {"train_ratio": train_ratio, "additional_weight": additional_weight,
                               "PCA_quantile_threshold": PCA_quantile_threshold}
        if best_sets is None:
            logging.warning(f"No valid model found for {model_name}.")
            metrics_by_model[model_name] = None
            continue
        train_data, test_data, holdout_data, feature_names = best_sets
        X_train, y_train, inds_train, weights_train = train_data
        X_test, y_test, inds_test, weights_test = test_data
        X_holdout, y_holdout, inds_holdout, weights_holdout = holdout_data
        clf = best_clas
        y_pred = clf.predict(X_test)
        y_train_nonholdout = y_train
        y_train_holdout = y_holdout

        def get_stats1(true, preds):
            return {"accuracy": accuracy_score(true, preds),
                    "precision": precision_score(true, preds, average='weighted', zero_division=0.0),
                    "recall": recall_score(true, preds, average='weighted', zero_division=0.0),
                    "f1_score": f1_score(true, preds, average='weighted', zero_division=0.0),
                    # "roc_auc": roc_auc_score(true, clf.predict_proba(pred)[:, 1],
                    #                         multi_class = "ovr", average='weighted'),
                    }

        def get_stats(true, data):
            return {"accuracy": accuracy_score(true, clf.predict(data)),
                    "precision": precision_score(true, clf.predict(data), average='weighted', zero_division=0.0),
                    "recall": recall_score(true, clf.predict(data), average='weighted', zero_division=0.0),
                    "f1_score": f1_score(true, clf.predict(data), average='weighted', zero_division=0.0),
                    # "roc_auc": roc_auc_score(true, clf.predict_proba(pred)[:, 1],
                    #                         multi_class = "ovr", average='weighted'),
                    }

        with open(f"{model_name}_confusion_matrix.json", "w") as f:
            json.dump(confusion_matrix(y_test, y_pred).tolist(), f)
        results = {
            "train": get_stats(y_train_nonholdout, X_train),
            "holdout": get_stats(y_train_holdout, X_holdout),
            "test": get_stats(y_test, X_test),
            "best_hyperparameters": clf.best_params_,
            "used_features": best_feat,
            "train_size": len(y_train),
            "test_size": len(y_test),
            "test_label0_frac": int(np.sum(y_test == 0)) / len(y_test),
            "best_params": best_params
        }

        if hasattr(clf.best_estimator_, "feature_importances_"):
            if hasattr(clf.best_estimator_, "importance_type"):
                print(f"Feature importances: {clf.best_estimator_.importance_type}")
            results["feature_importances"] = list(clf.best_estimator_.feature_importances_)
            results["feature_importances"] = [float(val) for val in results["feature_importances"]]

        if isinstance(clf.best_estimator_, LogisticRegression):
            # Taking the absolute values of the coefficients
            results["feature_importances"] = [float(abs(val)) for val in clf.best_estimator_.coef_.flatten()]



        def invert_permutation(p):
            """Return an array s with which np.array_equal(arr[p][s], arr) is True.
            The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.
            """
            p = np.asanyarray(p)  # in case p is a tuple, etc.
            s = np.empty_like(p)
            s[p] = np.arange(p.size)
            return list(s)

        if args.run_LLM and end_to_end_test_data is not None:
            resultsl = run_LLM(clf, end_to_end_test_data)
            with open(os.path.join(model_save_dir, "end_to_end_results.json"), "w") as f:
                json.dump(resultsl, f, indent=4)
            results["end_to_end_results"] = {x: get_stats1([y["gold_label"] for y in resultsl], [y[x] for y in resultsl])
                                             for x in resultsl[0].keys() if x not in ["gold_label", "fact", "large_probs"]}

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
                return {k: {k2: func(v2) for k2, v2 in v.items()} for k, v in d.items()}

            plot("first_left_train", "l",
                 double_apply(grounded_results, lambda x: x.indexer(inds_train[goes_left_train & (y_train > 0)])),
                 double_apply(unfaithful_results, lambda x: x.indexer(inds_train[goes_left_train & (y_train == 0)])),
                 model_save_dir + "/first_left_train.pdf", args)
            plot("first_right_train", "l",
                 double_apply(grounded_results, lambda x: x.indexer(inds_train[~goes_left_train & (y_train > 0)])),
                 double_apply(unfaithful_results, lambda x: x.indexer(inds_train[~goes_left_train & (y_train == 0)])),
                 model_save_dir + "/first_right_train.pdf", args)
            plot("first_left", "l",
                 double_apply(grounded_results, lambda x: x.indexer(inds_test[goes_left & (y_test > 0)])),
                 double_apply(unfaithful_results, lambda x: x.indexer(inds_test[goes_left & (y_test == 0)])),
                 model_save_dir + "/first_left.pdf", args)
            plot("first_right", "l",
                 double_apply(grounded_results, lambda x: x.indexer(inds_test[~goes_left & (y_test > 0)])),
                 double_apply(unfaithful_results, lambda x: x.indexer(inds_test[~goes_left & (y_test == 0)])),
                 model_save_dir + "/first_right.pdf", args)
            # else:
            #    results = load_metrics(model_save_dir)
            #    save_metrics(results, feature_names, model_save_dir)

        metrics_by_model[model_name] = results["test"]

    try:
        plot_metrics_comparison(metrics_by_model, save_dir)
    except Exception as e:
        logging.warning(f"Could not plot metrics comparison: {e}")

    return {x: metrics_by_model[x]["accuracy"] if metrics_by_model[x] is not None else None for x in metrics_by_model}


def process_facts2(target_token, facts, class_map, results, corrupted_probs, clean_probs, ids, tokenizer=None):
    for processed_fact in facts:
        clas = class_map(processed_fact)
        corrupted_score = processed_fact["results"]["corrupted"][target_token]["probs"]
        clean_score = processed_fact["results"]["clean"][target_token]["probs"]

        # If there is a zero interval, skip the fact
        interval_to_explain = max(clean_score - corrupted_score, 0)

        corrupted_probs[clas].add(corrupted_score)
        clean_probs[clas].add(clean_score)
        info_dict = {"is_additional": processed_fact["is_additional"] if "is_additional" in processed_fact else 1,
                     "id": int(processed_fact["fact"]["group_id"]) if "group_id" in processed_fact["fact"] and processed_fact["fact"]["group_id"] is not None and len(
                         processed_fact["fact"]["group_id"]) > 0 else None, "fact": processed_fact["fact"],
                     "results": processed_fact["results"]}
        ids[clas].append(info_dict)

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

            if len(token_effects_cont) > 0:
                while token_effects_cont[-1][-1] == " ":
                    token_effects_cont = token_effects_cont[:-1]

            def set_one(k, v):
                if k not in d:
                    d[k] = Feature(k)
                d[k].add(v)

            def set_else_0(c, k, f):
                if c:
                    set_one(k, f())
                else:
                    set_one(k, 0.0)

            if True:
                feats = [(len(token_effects_subj) > 2, "subj-first", lambda: token_effects_subj[0][1]),
                         (len(token_effects_subj) > 3, "subj-middle",
                          lambda: float(np.mean([x[1] for x in token_effects_subj[1:-2]]))),
                         (len(token_effects_subj) > 1, "subj-second-last", lambda: token_effects_subj[-2][1]),
                         (len(token_effects_subj) > 0, "subj-last", lambda: token_effects_subj[-1][1]),
                         (len(token_effects_cont) > 2, "cont-first", lambda: token_effects_cont[0][1]),
                         # (len(token_effects_cont) > 3, "cont-middle", lambda : float(np.mean([x[1] for x in token_effects_cont[1:-2]]))),
                         (len(token_effects_cont) > 1, "cont-second-last", lambda: token_effects_cont[-2][1]),
                         # (len(token_effects_cont) > 0, "cont-last", lambda : token_effects_cont[-1][1])
                         ]
            elif False:
                feats = [(len(token_effects_subj) > 1, "subj-first", lambda: token_effects_subj[0][1]),
                         (len(token_effects_subj) > 3, "subj-second-first", lambda: token_effects_subj[1][1]),
                         (len(token_effects_subj) > 4, "subj-middle",
                          lambda: float(np.mean([x[1] for x in token_effects_subj[2:-2]]))),
                         (len(token_effects_subj) > 2, "subj-second-last", lambda: token_effects_subj[-2][1]),
                         (len(token_effects_subj) > 0, "subj-last", lambda: token_effects_subj[-1][1]),
                         (len(token_effects_cont) > 1, "cont-first", lambda: token_effects_cont[0][1]),
                         (len(token_effects_subj) > 3, "cont-second-first", lambda: token_effects_subj[1][1]),
                         (len(token_effects_cont) > 4, "cont-middle",
                          lambda: float(np.mean([x[1] for x in token_effects_cont[2:-2]]))),
                         (len(token_effects_cont) > 2, "cont-second-last", lambda: token_effects_cont[-2][1]),
                         (len(token_effects_cont) > 0, "cont-last", lambda: token_effects_cont[-1][1])]
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
    print(corrupted_score, clean_score)
    interval_to_explain = max(clean_score - corrupted_score, 0)
    if "tokens" not in processed_fact["results"]:
        logging.warning('"tokens" not in processed_fact["results"]')
    return interval_to_explain == 0 or "tokens" not in processed_fact["results"]


def group_results2(facts_grounded, facts_unfaithful, tokenizer, args):
    num_classes = args.num_classes
    corrupted_probs = [Feature("corr") for _ in range(num_classes)]
    clean_probs = [Feature("clean") for _ in range(num_classes)]
    ids = [[] for _ in range(num_classes)]
    # Here results is a list of dictionaries, each dictionary contains the results for one class (i.e. label i.e. bucket), the bucket is decided in the process_facts2 function
    results = [
        {kind: {} for kind in ["hidden", "mlp", "attn"]}
        for _ in range(num_classes)]
    if False:
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
                   lambda x: 0, results, corrupted_probs, clean_probs, ids, tokenizer)
    print([x["fact"]["object"] for x in facts_grounded if filter_facts(x, "grounded_token")])
    ps = [x["results"]["clean"]["grounded_token"]["probs"] for x in facts_grounded]
    ls = np.linspace(0, 0.5, num_classes)
    if num_classes > 2:
        class_boundaries = np.quantile(ps, ls)[1:-1]

    # print(ps, ls, class_boundaries)

    def classify(fact):
        return (1 if args.separate else 0) + np.digitize(fact["results"]["clean"]["grounded_token"]["probs"],
                                                         class_boundaries) if num_classes > 2 else 1

    process_facts2("grounded_token", facts_grounded, lambda x: 1 if True else classify(x)
                   , results, corrupted_probs, clean_probs, ids, tokenizer)

    # print("corrupted_probs, clean_probs", [x.d for x in corrupted_probs], [x.d for x in clean_probs])
    def group(idsx, resultsx, corrupted_probsx, clean_probsx):
        for k, v in list(resultsx.items()):
            for k2, v2 in list(v.items()):
                resultsx[k][k2 + "_mean"] = v[k2].copy(k2 + "_mean")
                resultsx[k][k2 + "_std"] = v[k2].copy(k2 + "_std")
                resultsx[k][k2 + "_max"] = v[k2].copy(k2 + "_max")
                resultsx[k].pop(k2)
        for x in np.unique(idsx):
            inds = np.where(np.array(idsx) == x)[0]
            print(inds)
            for i in inds[1:]:
                idsx[i] = None
            for k, v in resultsx.items():
                for k2, v2 in v.items():
                    if k2.endswith("_mean"):
                        v2.d[inds[0]] = np.mean([v2.d[i] for i in inds])
                    elif k2.endswith("_std"):
                        v2.d[inds[0]] = np.std([v2.d[i] for i in inds])
                    else:
                        v2.d[inds[0]] = np.max([v2.d[i] for i in inds])
        indskeep = [x for x in range(len(idsx)) if idsx[x] is not None]
        for k, v in resultsx.items():
            for k2, v2 in v.items():
                v2.d = [v2.d[x] for x in indskeep]
        corrupted_probsx.d = [corrupted_probsx.d[x] for x in indskeep]
        clean_probsx.d = [clean_probsx.d[x] for x in indskeep]

    if args.group and not (any(x["id"] is None for x in ids[0]) or any(x["id"] is None for x in ids[1])):
        group([x["id"] for x in ids[0]], results[0], corrupted_probs[0], clean_probs[0])
        group([x["id"] for x in ids[1]], results[1], corrupted_probs[1], clean_probs[1])
    vs = list(
        (x, i, len(next(iter(x[0]["hidden"].values())))) for i, x in
        enumerate(zip(results, corrupted_probs, clean_probs, ids)) if len(x[0]["hidden"]) > 0)
    print("class_c", [x[2] for x in vs])
    vs = [(x, i) for x, i, l in vs if l >= args.min_count]
    # in next experiment try ["grounded", "confidently grounded"]
    print(results)
    return [x for x, i in vs], [f"p:{i}" for x, i in vs]


def generate_datasets2(buckets
                       , args,
                       train_ratio=0.8, balance=False, balance_test=False, balance_w=True, additional_weight=0, check_additional=False,
                       PCA_quantile_threshold=0, holdout_ratio = 0.2):
    logger = get_logger()
    features_to_include = list(args.features_to_include if args.features_to_include is not None else list(buckets[0][0].values())[0].keys())
    feature_names = [
        f"{kind}-{feature}" for feature in features_to_include for kind in args.kinds_to_include
    ]
    logger.info(f"Feature names: {feature_names}")

    all_label_samples = []
    for label, bucket_results in enumerate(buckets):
        all_label_samples.append([])
        kinds_results, corr_probs, clean_probs, info = bucket_results
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
                kinds_results[k1][k2].get(i)
                for k1, k2 in [(kind, feature) for feature in features_to_include for kind in args.kinds_to_include]
            )
            print(tuple((kind_results,feature_results)
                for kind_results in args.kinds_to_include
                for feature_results in args.features_to_include))

            # if any([feature is None for feature in candidate_example]):
            #    continue

            all_label_samples[-1].append((candidate_example, label, i, info[i]["is_additional"]))
    assert len(set([tuple(sorted(x)) for x in all_label_samples])) == len(all_label_samples)
    print("all_label_samples", len(all_label_samples[0]), len(all_label_samples[1]))
    minl = min(len([y for y in x if y[3] > 0]) for x in all_label_samples) if balance else max(
        len([y for y in x if y[3] > 0]) for x in all_label_samples)

    if PCA_quantile_threshold is not None:
        all_samples_unsup = []
        for current_label_samples in all_label_samples:
            all_samples_unsup.extend([sample[0] for sample in current_label_samples])
        PCA = sklearn.decomposition.PCA(n_components=2, whiten=True)
        PCA.fit(np.array(all_samples_unsup))
        pca_importances = np.sum(np.abs(PCA.explained_variance_ratio_[:, None] * PCA.components_), axis=0)
        print("PCA importances", pca_importances)
        print([np.where(pca_importances < x)[0] for x in np.quantile(pca_importances, [0.05, 0.1, 0.25, 0.5, 0.75])])
        print([[feature_names[y] for y in np.where(pca_importances < x)[0]] for x in
               np.quantile(pca_importances, [0.05, 0.1, 0.25, 0.5, 0.75])])
        features_to_remove = np.where(pca_importances < np.quantile(pca_importances, PCA_quantile_threshold))

    all_samples = []
    all_labels = []
    all_inds = []
    all_weights = []
    all_samples1 = []
    all_labels1 = []
    all_inds1 = []
    all_weights1 = []
    for current_label_samples in all_label_samples:
        current_label_samples0 = [y for y in current_label_samples if y[3] == 0]
        current_label_samples1 = [y for y in current_label_samples if y[3] > 0]
        np.random.shuffle(current_label_samples1)
        np.random.shuffle(current_label_samples0)
        all_samples.extend([sample[0] for sample in current_label_samples1[:minl]])
        all_labels.extend([sample[1] for sample in current_label_samples1[:minl]])
        all_inds.extend([sample[2] for sample in current_label_samples1[:minl]])
        all_weights.extend([sample[3] for sample in current_label_samples1[:minl]])
        all_samples1.extend([sample[0] for sample in current_label_samples1[minl:] + current_label_samples0])
        all_labels1.extend([sample[1] for sample in current_label_samples1[minl:] + current_label_samples0])
        all_inds1.extend([sample[2] for sample in current_label_samples1[minl:] + current_label_samples0])
        all_weights1.extend([sample[3] for sample in current_label_samples1[minl:] + current_label_samples0])

    # Convert all_samples and all_labels to np arrays
    all_samples = np.array(all_samples)
    if PCA_quantile_threshold is not None:
        all_samples[:, features_to_remove] = 0
    all_labels = np.array(all_labels)
    all_inds = np.array(all_inds)
    all_weights = np.array(all_weights)
    assert np.sum(all_weights == 0) == 0
    print("number of classes", np.unique(all_labels, return_counts=True))
    # Calculate lengths for each split
    total_size = len(all_samples[all_weights > 0])
    train_size = int(total_size * train_ratio)

    train_samples_before, test_samples, train_labels, test_labels, train_inds, test_inds, train_weights, _ = sklearn.model_selection.train_test_split(
        all_samples, all_labels, all_inds, all_weights, train_size=train_size, stratify=all_labels, random_state=np.random.randint(0, 10000))
    try:
        train_samples_before, holdout_samples, train_labels, holdout_labels, train_inds, holdout_inds, train_weights, holdout_weights = sklearn.model_selection.train_test_split(
            train_samples_before, train_labels, train_inds, train_weights,
            test_size=max(int(len(train_labels) * holdout_ratio), 5), stratify=train_labels, random_state=np.random.randint(0, 10000))
    except Exception as e:
        logging.exception("Trainset likely too small.", exc_info=e)
        return None

    all_weights1 = np.array(all_weights1)
    all_samples1 = np.array(all_samples1)
    if len(all_samples1) > 0:
        if PCA_quantile_threshold is not None:
            all_samples1[:, features_to_remove] = 0
        all_labels1 = np.array(all_labels1)
        all_inds1 = np.array(all_inds1)
        train_samples_before = np.concatenate([train_samples_before, all_samples1[np.where(all_weights1 > 0)[0]]],
                                              axis=0)
        train_labels = np.concatenate([train_labels, all_labels1[all_weights1 > 0]], axis=0)
        train_inds = np.concatenate([train_inds, all_inds1[all_weights1 > 0]], axis=0)
        train_weights = np.concatenate([train_weights, all_weights1[all_weights1 > 0]], axis=0)

    if balance_test:
        dif = np.sum(test_labels == 0) - np.sum(test_labels == 1)
        assert np.abs(dif) < np.sum(test_labels == 0), f"{dif} {np.sum(test_labels == 0)}"
        assert np.abs(dif) < np.sum(test_labels == 1), f"{dif} {np.sum(test_labels == 1)}"
        if dif > 0:
            to_remove = np.where(test_labels == 0)[0]
            test_labels = np.delete(test_labels, to_remove[:dif], axis=0)
            test_inds = np.delete(test_inds, to_remove[:dif], axis=0)
            test_samples = np.delete(test_samples, to_remove[:dif], axis=0)
        if dif < 0:
            to_remove = np.where(test_labels == 1)[0]
            test_labels = np.delete(test_labels, to_remove[:dif], axis=0)
            test_inds = np.delete(test_inds, to_remove[:dif], axis=0)
            test_samples = np.delete(test_samples, to_remove[:dif], axis=0)
        difa = np.sum(test_labels == 0) - np.sum(test_labels == 1)
        print("difs", dif, difa)

    train_samples = train_samples_before
    if balance_w:
        stl1 = np.sum(train_labels == 1)
        stl0 = np.sum(train_labels == 0)
        train_weights = np.array([1.0 if x == 1 else (stl1 / stl0) for x in train_labels])
        shl1 = np.sum(holdout_labels == 1)
        shl0 = np.sum(holdout_labels == 0)
        holdout_weights = np.array([1.0 if x == 1 else (shl1 / shl0) for x in holdout_labels])
    if np.sum(all_weights1 == 0) > 0:
        sumw = np.sum(train_weights)
        train_weights = np.concatenate([train_weights, np.zeros(len(all_samples1[all_weights1 == 0]))], axis=0)
        train_samples = np.concatenate([train_samples_before, all_samples1[all_weights1 == 0]], axis=0)
        train_labels = np.concatenate([train_labels, all_labels1[all_weights1 == 0]], axis=0)
        train_inds = np.concatenate([train_inds, all_inds1[all_weights1 == 0]], axis=0)
        train_weights[len(train_samples_before):] = additional_weight * (
                    sumw / (len(train_weights) - len(train_samples_before)))
        tla = train_labels[len(train_samples_before):]
        stl10 = np.sum(tla == 1)
        stl00 = np.sum(tla == 0)
        train_weights[len(train_samples_before):] = np.array(
            [y if x == 1 else y * (stl10 / stl00) for y, x in zip(train_weights[len(train_samples_before):], tla)])
    elif check_additional:
        raise Exception()
    print("train_weights", np.unique(train_weights, return_counts=True), train_weights.tolist())
    # print(test_dataset[0].shape, test_dataset[0].shape, feature_names)
    # print(list(zip(test_dataset[0][:, -1], test_dataset[1])))
    return ((train_samples, train_labels, train_inds, train_weights),
            (test_samples, test_labels, test_inds, None),
            (holdout_samples, holdout_labels, holdout_inds, holdout_weights),
            feature_names,)


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
    additional_values = []
    print([x for x in args.other_dataset_name if x != args.dataset_name])
    for x in [x for x in args.other_dataset_name if
              x != args.dataset_name] if args.other_dataset_name is not None else []:
        additional_values.append([
            [dict(x, is_additional=0) for x in
             read_json(os.path.join(args.causal_traces_dir, x, args.model_name, f"{bucket}.json"))] for bucket in
            buckets
        ])
        print(type(additional_values[-1][0][0]))
        assert isinstance(additional_values[-1][0][0], dict), f"{type(additional_values[-1][0])}"
    for x in [x for x in args.other_dataset_name_unsup if
              x != args.dataset_name] if args.other_dataset_name is not None else []:
        additional_values.append([
            [dict(x, is_additional=-1) for x in
             read_json(os.path.join(args.causal_traces_dir, x, args.model_name, f"{bucket}.json"))] for bucket in
            buckets
        ])
        print(type(additional_values[-1][0][0]))
        assert isinstance(additional_values[-1][0][0], dict), f"{type(additional_values[-1][0])}"

    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("openai/whisper-base.en", force_download=False,
    #                                          add_bos_token=True)
    print("read_lengths", [len(read_json(x)) for x in buckets_paths])
    data0 = read_json(buckets_paths[0]) + list(chain(*[x[0] for x in additional_values]))
    data1 = read_json(buckets_paths[1]) + list(chain(*[x[1] for x in additional_values]))
    print(np.unique([str(type(x)) for x in list(chain(*[x[0] for x in additional_values]))], return_index=True))
    results, buckets = group_results2(data0,
                                      data1,
                                      None, args)

    # If we are only including certain kinds, filter the kinds
    if args.kinds_to_include is not None:
        results = [
            (
                {kind: bucket_results[0][kind] for kind in bucket_results[0] if kind in args.kinds_to_include},
            ) + tuple(bucket_results[1:])
            for bucket_results in results
        ]

    # If we are only including certain features, filter the features
    if args.features_to_include is not None:
        print("resultsl", [len(x) for x in results])
        results = [
            (
                {
                    kind: {
                        feature: bucket_results[0][kind][feature]
                        for feature in args.features_to_include
                        if feature in bucket_results[0][kind]
                    }
                    for kind in bucket_results[0]
                },
            ) + tuple(bucket_results[1:])
            for bucket_results in results
        ]
    # print([[[len(z) for z in y] for y in x[0].values()] for x in results])
    # Generate the datasets
    if False:
        train_data, test_data, holdout_data, feature_names = generate_datasets2(
            results, args,
            train_ratio=args.train_ratio, balance=args.balance, balance_w=args.balance_w,
            additional_weight=args.additional_weight,
            check_additional=args.other_dataset_name is not None and len(args.other_dataset_name) > 0,
            PCA_quantile_threshold=args.PCA_quantile_threshold
        )
        print("train_data", [len(train_data), len(train_data[0]), train_data[1]])
        print("test_data", [len(test_data), len(test_data[0]), test_data[1]])

    def generate_wrapper(results, train_ratio, additional_weight, PCA_quantile_threshold):
        return generate_datasets2(
            results, args,
            train_ratio=train_ratio, balance=args.balance, balance_test=args.balance_test, balance_w=args.balance_w,
            additional_weight=additional_weight,
            check_additional=args.other_dataset_name is not None and len(args.other_dataset_name) > 0,
            PCA_quantile_threshold=PCA_quantile_threshold,
            holdout_ratio = args.holdout_ratio
        )

    # Train the models and save the results
    return train_and_save(models, results, generate_wrapper, buckets, results[1][0], results[0][0], args, seed=args.seed,
                   target_acc=args.target_acc)


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
            ).pvalue if len(grounded_results[kind][label]) > 2 and len(unfaithful_results[kind][label]) > 2 else 1 for
                        label in labels]

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
        ".pdf",
        f"_grounded={len(next(iter(next(iter(grounded_results.values())).values())))}_ungrounded={len(next(iter(next(iter(unfaithful_results.values())).values())))}.pdf"
    )
    print(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", format='pdf')
    plt.close()


class Namespace2:
    def __init__(self):
        self.causal_traces_dir = "./causal_traces"
        self.dataset_name = "fakepedia"
        self.model_name = "t5gfinetune"
        self.output_dir = "out"
        features = "two"
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
        elif features == "only_important_no_cont":
            self.features_to_include = ["subj-first", "subj-second-first", "subj-middle", "subj-second-last", "subj-last"]
        elif features == "two":
            self.features_to_include = ["subj-last", "subj-middle", "subj-first"]
        self.kinds_to_include = ["mlp"] + ([] if False else ["hidden"])
        self.train_ratio = 0.7
        self.ablation_only_clean = False
        self.ablation_include_corrupted = False
        self.seed = 50
        self.num_classes = 2
        self.min_count = 5
        self.separate = False
        self.group = False
        self.target_acc = 1
        self.additional_weight = 0
        self.PCA_quantile_threshold = None
        self.other_dataset_name = ["simple", "transl"] if False else []
        self.other_dataset_name_unsup = ["med"] if False else []
        self.balance = False
        self.balance_w = False
        self.balance_test = False
        self.num_feat_tries = 10
        self.num_train_sizes = 1
        self.holdout_score_limit = 1
        self.run_LLM = True
        self.tree_nodes = (3, 15)
        self.num_tree_nodes = 1
        self.holdout_ratio = 0.05
        self.random_feat_tries = False
        self.id_for_end_to_end = 200000


def main2(models, x):
    if False:
        sp = "./specific_runs/run4"
        p = './causal_traces/f/f'
        os.makedirs(p, exist_ok=True)
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
    args = Namespace2()
    args.seed += x
    freeze_args(args)
    set_seed_everywhere(args.seed)
    return train_detector(args, models)


import traceback
import warnings
import sys

if __name__ == "__main__":
    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
        log = file if hasattr(file, 'write') else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))


    warnings.showwarning = warn_with_traceback
    np.seterr(all='raise')
    scipy.special.seterr(all='raise')
    models = {}
    dataset_size_muli = 1
    if True:
        models["DecisionTreeSmall"] = {
            "model": DecisionTreeClassifier(),
            "param_grid": {
                "max_depth": [2, 3],
            },
        }
    if True:
        models["LogisticRegression"] = {
            "model": LogisticRegression(max_iter=1000, solver="liblinear"),
            "param_grid": {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000], "penalty": ["l1", "l2"]},
        }
    if True:
        models["KNN"] = {
            "model": KNeighborsClassifier(),
            "param_grid": {
                "n_neighbors": np.arange(1, 5, 1).tolist(),
                "p": np.arange(2, 10, 1).tolist(),
                "n_jobs": [-1],
                "leaf_size": [2]
            },
        }
    if True:
        models["DecisionTree"] = {
            "model": DecisionTreeClassifier(),
            "param_grid": {
                "max_depth": np.arange(4, 15, 1).tolist(),
            },
        }
    models = dict([x for x in models.items() if len(x[1]) > 0])
    xgboostmode = "None" if True else "small"
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
                eval_metric="logloss", device="cuda", importance_type="total_gain"
            ),
            "param_grid": {
                "learning_rate": [0.01, 0.05, 0.1],
                "n_estimators": [20, 50, 75, 100],
                "max_depth": [2, 4],
                "subsample": [0.8],
            },
        }
    else:
        pass

    accs = []
    n = 1
    for x in range(n):
        acc = main2(models, x)
        accs.append(acc)
        print(f"acc{x}:{acc}")
    for k in accs[0].keys():
        data = [x[k] for x in accs if x[k] is not None]
        print((k) + ((data, np.mean(data), np.std(data)) if len(data) > 0 else ("No data",)))
    # !rm -r LLama

    for model_name in models:
        with open(f"{model_name}_confusion_matrix.json") as f:
            confm = json.load(f)
        disp = ConfusionMatrixDisplay(confusion_matrix=np.array(confm))
        disp.plot()
        plt.savefig(f"{model_name}_confusion_matrix.png")
