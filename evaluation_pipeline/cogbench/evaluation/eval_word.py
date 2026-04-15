import os
import glob
import json
import numpy as np
import scipy.io as sio
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from scipy.stats import pearsonr


def standardize_matrix(matrix):
    row_means = np.mean(matrix, axis=1, keepdims=True)
    row_stds = np.std(matrix, axis=1, keepdims=True)
    row_stds[row_stds == 0] = 1.0  # avoid divide by zero
    return (matrix - row_means) / row_stds


def ridge_prediction(X_train, X_test, y_train):
    model = Ridge()
    alphas = np.logspace(-4, 4, 10)  # tunable alpha search range
    param_grid = {"alpha": alphas}
    kf = KFold(n_splits=5, shuffle=False)

    X_train = np.nan_to_num(X_train, 0.0)
    X_test = np.nan_to_num(X_test, 0.0)

    grid = GridSearchCV(
        model,
        param_grid,
        scoring="neg_mean_squared_error",
        cv=kf,
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    return grid.predict(X_test)


def run_prediction(feature, fmri, save_path):
    """
    feature: (n_trials, n_features)
    fmri:    (n_trials, n_voxels_selected)
    """
    X = standardize_matrix(np.asarray(feature))
    Y = np.asarray(fmri)

    kf = KFold(n_splits=5, shuffle=False)
    all_corrs = []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]

        Y_pred = ridge_prediction(X_tr, X_te, Y_tr)

        # remove all-zero columns if there are any
        valid_cols = ~np.all(Y_te == 0, axis=0)
        if not np.any(valid_cols):
            all_corrs.append(0.0)
            continue

        Y_pred = Y_pred[:, valid_cols]
        Y_te = Y_te[:, valid_cols]

        # pearson r for each trial
        trial_corrs = np.array([
            pearsonr(Y_pred[i], Y_te[i])[0] for i in range(Y_te.shape[0])
        ])
        trial_corrs[np.isnan(trial_corrs)] = 0.0

        # top 10% mean
        k = max(1, int(len(trial_corrs) * 0.1))
        top_mean = np.mean(np.sort(trial_corrs)[-k:]) if k > 0 else 0.0
        all_corrs.append(top_mean)

    # average across 5 folds
    final_score = float(np.mean(all_corrs))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sio.savemat(save_path, {"score": final_score})
    print(f"Saved: {save_path} | overall top-10% mean pearson r = {final_score:.4f}")


def _resolve_split_dirs(data_path):
    train_dir = os.path.join(data_path, "train")
    dev_dir = os.path.join(data_path, "dev")
    test_dir = os.path.join(data_path, "test")
    if os.path.isdir(train_dir) and os.path.isdir(dev_dir):
        return {
            "train": train_dir,
            "dev": dev_dir,
            "test": test_dir if os.path.isdir(test_dir) else None,
        }
    return None


def _compute_top10_trial_score(y_pred, y_true):
    valid_cols = ~np.all(y_true == 0, axis=0)
    if not np.any(valid_cols):
        return 0.0

    y_pred = y_pred[:, valid_cols]
    y_true = y_true[:, valid_cols]
    trial_corrs = np.array([pearsonr(y_pred[i], y_true[i])[0] for i in range(y_true.shape[0])])
    trial_corrs[np.isnan(trial_corrs)] = 0.0

    k = max(1, int(len(trial_corrs) * 0.1))
    return float(np.mean(np.sort(trial_corrs)[-k:])) if k > 0 else 0.0


def run_prediction_train_dev_test(X_train, X_dev, X_test, y_train, y_dev, y_test, save_path):
    alphas = np.logspace(-4, 4, 10)

    X_train = np.nan_to_num(X_train, 0.0)
    X_dev = np.nan_to_num(X_dev, 0.0)
    X_test = np.nan_to_num(X_test, 0.0)
    y_train = np.nan_to_num(y_train, 0.0)
    y_dev = np.nan_to_num(y_dev, 0.0)
    y_test = np.nan_to_num(y_test, 0.0)

    best_alpha = None
    best_dev_mse = None
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        dev_pred = model.predict(X_dev)
        dev_mse = float(np.mean((dev_pred - y_dev) ** 2))
        if best_dev_mse is None or dev_mse < best_dev_mse:
            best_dev_mse = dev_mse
            best_alpha = float(alpha)

    # Keep dev strictly for validation/evaluation. Do not train directly on dev data.
    fit_X = X_train
    fit_y = y_train
    final_model = Ridge(alpha=best_alpha)
    final_model.fit(fit_X, fit_y)
    test_pred = final_model.predict(X_test)

    final_score = _compute_top10_trial_score(test_pred, y_test)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sio.savemat(save_path, {"score": final_score, "best_alpha": best_alpha, "dev_mse": best_dev_mse})
    print(
        f"Saved: {save_path} | split score = {final_score:.4f}, "
        f"best_alpha = {best_alpha:.6f}, dev_mse = {best_dev_mse:.6f}"
    )


def _resolve_cogbench_root(data_path):
    if os.path.isdir(os.path.join(data_path, "word")) and os.path.isdir(os.path.join(data_path, "word_fmri")):
        return data_path

    if os.path.basename(os.path.normpath(data_path)) == "word_fmri":
        candidate_root = os.path.dirname(os.path.normpath(data_path))
        if os.path.isdir(os.path.join(candidate_root, "word")):
            return candidate_root

    return data_path


def _load_feature_matrix(feature_json_path, stimuli_list):
    with open(feature_json_path, "r", encoding="utf-8") as f:
        feature_dict = json.load(f)

    if not feature_dict:
        raise ValueError(f"Feature file is empty: {feature_json_path}")

    hidden_size = len(next(iter(feature_dict.values())))
    missing = []
    feature_rows = []

    for stimulus in stimuli_list:
        feature = feature_dict.get(stimulus)
        if feature is None:
            missing.append(stimulus)
            feature = [0.0] * hidden_size
        feature_rows.append(feature)

    if missing:
        print(f"Warning: {len(missing)} stimuli missing in feature json, filled with zeros.")

    return np.asarray(feature_rows, dtype=np.float32)


def eval_word_fmri(args):
    data_path = _resolve_cogbench_root(str(args.data_path))
    output_root = str(args.output_dir)
    model_name = os.path.basename(os.path.normpath(str(args.model_path_or_name)))
    revision_name = args.revision_name if args.revision_name is not None else "main"
    task_output_dir = os.path.join(output_root, model_name, revision_name, "cogbench", "word_fmri")
    feature_json_path = os.path.join(task_output_dir, "word_feature.json")
    if not os.path.exists(feature_json_path):
        raise FileNotFoundError(
            f"Feature file not found: {feature_json_path}. Please run inference first."
        )
    out_dir = task_output_dir
    os.makedirs(out_dir, exist_ok=True)

    split_dirs = _resolve_split_dirs(data_path)

    if split_dirs is not None:
        split_features = {}
        split_fmri_files = {}
        available_splits = ["train", "dev"]
        if split_dirs["test"] is not None:
            available_splits.append("test")

        for split_name in available_splits:
            split_path = split_dirs[split_name]
            words_path = os.path.join(split_path, "word", "word.txt")
            with open(words_path, encoding="utf-8") as f:
                stimuli_list = [line.strip() for line in f if line.strip()]
            split_features[split_name] = _load_feature_matrix(feature_json_path, stimuli_list)
            split_fmri_files[split_name] = sorted(glob.glob(os.path.join(split_path, "word_fmri", "*_selected.mat")))

        common_subjects = None
        for split_name in available_splits:
            subjects = {
                os.path.basename(path).replace("_selected.mat", "") for path in split_fmri_files[split_name]
            }
            common_subjects = subjects if common_subjects is None else common_subjects & subjects

        common_subjects = sorted(common_subjects) if common_subjects else []
        if args.fast:
            common_subjects = common_subjects[:1]
            print("[FAST] Running sanity check only: first subject, split mode.")

        if not common_subjects:
            raise FileNotFoundError("No common *_selected.mat subject files found across train/dev/test splits.")

        for subject in common_subjects:
            has_test = "test" in split_features
            print(f"\nProcessing {subject} (split mode, eval={'test' if has_test else 'dev'})")
            split_fmri_data = {}
            for split_name in available_splits:
                fmri_path = os.path.join(split_dirs[split_name], "word_fmri", f"{subject}_selected.mat")
                split_fmri_data[split_name] = sio.loadmat(fmri_path)["examples"]

            n_train = min(split_features["train"].shape[0], split_fmri_data["train"].shape[0])
            n_dev = min(split_features["dev"].shape[0], split_fmri_data["dev"].shape[0])
            if has_test:
                n_test = min(split_features["test"].shape[0], split_fmri_data["test"].shape[0])
            else:
                n_test = n_dev

            if n_train < 5 or n_dev < 2 or n_test < 2:
                print(
                    f"Skip {subject}: not enough aligned trials "
                    f"(train/dev/test = {n_train}/{n_dev}/{n_test})."
                )
                continue

            X_train = split_features["train"][:n_train]
            X_dev = split_features["dev"][:n_dev]
            y_train = split_fmri_data["train"][:n_train]
            y_dev = split_fmri_data["dev"][:n_dev]
            if has_test:
                X_test = split_features["test"][:n_test]
                y_test = split_fmri_data["test"][:n_test]
            else:
                X_test = X_dev[:n_test]
                y_test = y_dev[:n_test]

            if args.fast:
                n_voxels = min(128, y_train.shape[1], y_dev.shape[1], y_test.shape[1])
                X_train = X_train[: min(80, X_train.shape[0])]
                X_dev = X_dev[: min(30, X_dev.shape[0])]
                X_test = X_test[: min(30, X_test.shape[0])]
                y_train = y_train[: X_train.shape[0], :n_voxels]
                y_dev = y_dev[: X_dev.shape[0], :n_voxels]
                y_test = y_test[: X_test.shape[0], :n_voxels]
                save_path = os.path.join(out_dir, f"{subject}_sanity_score.mat")
            else:
                save_path = os.path.join(out_dir, f"{subject}_score.mat")

            run_prediction_train_dev_test(X_train, X_dev, X_test, y_train, y_dev, y_test, save_path)
        return

    words_path = os.path.join(data_path, "word", "word.txt")
    with open(words_path, encoding="utf-8") as f:
        stimuli_list = [line.strip() for line in f if line.strip()]

    feature_matrix = _load_feature_matrix(feature_json_path, stimuli_list)

    fmri_dir = os.path.join(data_path, "word_fmri")
    fmri_files = sorted(glob.glob(os.path.join(fmri_dir, "*_selected.mat")))

    if args.fast:
        fmri_files = fmri_files[:1]
        print("[FAST] Running sanity check only: first subject, small subset.")

    if not fmri_files:
        raise FileNotFoundError(f"No *_selected.mat files found under {fmri_dir}")

    for fmri_path in fmri_files:
        subject = os.path.basename(fmri_path).replace("_selected.mat", "")
        print(f"\nProcessing {subject}")

        mat = sio.loadmat(fmri_path)
        fmri_data = mat["examples"]  # (672, n_selected)

        n_trials = min(feature_matrix.shape[0], fmri_data.shape[0])
        if n_trials < 5:
            print(f"Skip {subject}: not enough aligned trials ({n_trials}).")
            continue

        if args.fast:
            n_trials = max(5, min(50, n_trials))
            n_voxels = min(128, fmri_data.shape[1])
            fmri_for_eval = fmri_data[:n_trials, :n_voxels]
            feat_for_eval = feature_matrix[:n_trials]
            save_path = os.path.join(out_dir, f"{subject}_sanity_score.mat")
        else:
            fmri_for_eval = fmri_data[:n_trials]
            feat_for_eval = feature_matrix[:n_trials]
            save_path = os.path.join(out_dir, f"{subject}_score.mat")

        run_prediction(feat_for_eval, fmri_for_eval, save_path)

