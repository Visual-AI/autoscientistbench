"""Interactive Bayesian optimization workflow for manual batch selection and feedback."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge

from bayesian_batch import (
    acquisition_scores,
    build_gp,
    build_preprocess,
    fit_predict_surrogate,
    normalize_df,
)


def ensure_target_column(df, target):
    if target not in df.columns:
        df[target] = np.nan
    df[target] = pd.to_numeric(df[target], errors="coerce")


def parse_index_text(text, size):
    items = [s.strip() for s in str(text).split(",") if s.strip()]
    if not items:
        return []
    indices = []
    for item in items:
        idx = int(item)
        if idx < 0 or idx >= size:
            raise ValueError(f"Index {idx} is out of range. Valid range: 0 to {size - 1}")
        indices.append(idx)
    return indices


def parse_yield_text(text):
    value = str(text).strip().replace("%", "")
    if value == "":
        raise ValueError("Yield cannot be empty.")
    return float(value)


def describe_condition_row(row, target):
    parts = []
    for col, value in row.items():
        if col in {target, "index", "pred_mean", "pred_std", "acq"}:
            continue
        parts.append(f"{col}={value}")
    return ", ".join(parts)


def show_candidates(df, target, start=0, count=20):
    feature_cols = [c for c in df.columns if c != target]
    start = max(0, int(start))
    count = max(1, int(count))
    end = min(len(df), start + count)
    preview = df.loc[start : end - 1, feature_cols].copy()
    print("\nCandidate preview:")
    print(preview.to_string())


def show_rows(df, target, indices):
    feature_cols = [c for c in df.columns if c != target]
    preview = df.loc[indices, feature_cols].copy()
    print("\nSelected experimental conditions:")
    print(preview.to_string())


def prompt_initial_observations(df, target):
    print(f"\nNo known values were found in `{target}`. At least one experimental result is required to start optimization.")
    print("Use `list` to view the first 20 candidates, or `list <start_index> <count>` for pagination.")
    while df[target].notna().sum() == 0:
        text = input("\nEnter the indices of experiments already performed (comma-separated), or press q to quit: ").strip()
        if text.lower() == "q":
            raise SystemExit("Exited by user.")
        if text.lower().startswith("list"):
            parts = text.split()
            if len(parts) == 1:
                show_candidates(df, target)
            elif len(parts) == 3:
                show_candidates(df, target, start=int(parts[1]), count=int(parts[2]))
            else:
                print("Usage: list OR list <start_index> <count>")
            continue
        try:
            indices = parse_index_text(text, len(df))
            if not indices:
                print("Please enter at least one index.")
                continue
            show_rows(df, target, indices)
            for idx in indices:
                value = parse_yield_text(input(f"Enter the true {target} for index={idx}: "))
                df.at[idx, target] = value
            print(f"Recorded {len(indices)} initial observations.")
        except ValueError as exc:
            print(f"Invalid input: {exc}")


def choose_batch_interactive(
    Xf,
    y,
    known,
    mean,
    std,
    batch_size,
    acquisition,
    xi,
    kappa,
    surrogate,
    method,
    random_state,
):
    scores = acquisition_scores(
        mean,
        std,
        best_train=float(np.max(y.loc[known].values)),
        acq=acquisition,
        xi=xi,
        kappa=kappa,
    )
    unknown = ~known
    batch_size = int(batch_size)
    chosen = []
    m = str(method).lower()

    if m == "topk":
        s_idx = np.argsort(-scores)
        chosen = [int(i) for i in s_idx if unknown[i]][:batch_size]
        return chosen, scores

    if m == "thompson":
        rng = np.random.default_rng(random_state)
        sample = mean + std * rng.standard_normal(size=len(y))
        s_idx = np.argsort(-sample)
        for i in s_idx:
            if unknown[i]:
                chosen.append(int(i))
            if len(chosen) >= batch_size:
                break
        return chosen, scores

    cur_mean = np.asarray(mean, dtype=float).copy()
    cur_std = np.asarray(std, dtype=float).copy()
    cur_known = known.copy()
    train_y_believed = y.copy()
    best_believed = float(np.max(y.loc[known].values))

    for _ in range(batch_size):
        cur_scores = acquisition_scores(
            cur_mean,
            cur_std,
            best_train=best_believed,
            acq=acquisition,
            xi=xi,
            kappa=kappa,
        )
        s_idx = np.argsort(-cur_scores)
        next_i = None
        for i in s_idx:
            if (i not in chosen) and (not cur_known[i]):
                next_i = int(i)
                break
        if next_i is None:
            break
        chosen.append(next_i)

        # In a real experiment, belief updates must use model predictions only,
        # rather than hidden ground-truth values.
        cur_known[next_i] = True
        believed_y = cur_mean[next_i]
        train_y_believed.iloc[next_i] = believed_y
        best_believed = max(best_believed, float(believed_y))

        train_idx = np.where(cur_known)[0]

        if str(surrogate).lower() == "gp":
            model = build_gp(Xf.shape[1], random_state=random_state)
            model.fit(Xf[train_idx], train_y_believed.iloc[train_idx].values)
            cur_mean, cur_std = model.predict(Xf, return_std=True)
        else:
            model = BayesianRidge(compute_score=True)
            model.fit(Xf[train_idx], train_y_believed.iloc[train_idx].values)
            mu = model.coef_
            sigma = model.sigma_
            intercept = model.intercept_
            noise_var = 1.0 / model.alpha_
            cur_mean = Xf.dot(mu) + intercept
            var = np.einsum("ij,jk,ik->i", Xf, sigma, Xf) + noise_var
            cur_std = np.sqrt(np.maximum(var, 1e-12))

    final_scores = acquisition_scores(
        mean,
        std,
        best_train=float(np.max(y.loc[known].values)),
        acq=acquisition,
        xi=xi,
        kappa=kappa,
    )
    return chosen, final_scores


def recommend_batch(
    df,
    target,
    batch_size=5,
    acquisition="EI",
    xi=0.0,
    kappa=2.0,
    surrogate="linear",
    method="kriging",
    random_state=42,
):
    known = df[target].notna().to_numpy()
    if np.sum(known) == 0:
        raise ValueError("At least one known observation is required before recommendation can start.")
    if np.all(known):
        return pd.DataFrame()

    X = df.drop(columns=[target]).reset_index(drop=True)
    y = df[target].astype(float).reset_index(drop=True)
    preprocess = build_preprocess(X)
    Xf = preprocess.fit_transform(X)
    if hasattr(Xf, "toarray"):
        Xf = Xf.toarray()

    mean, std = fit_predict_surrogate(
        Xf,
        y,
        known,
        surrogate=surrogate,
        random_state=random_state,
    )
    chosen, scores = choose_batch_interactive(
        Xf=Xf,
        y=y,
        known=known,
        mean=mean,
        std=std,
        batch_size=batch_size,
        acquisition=acquisition,
        xi=xi,
        kappa=kappa,
        surrogate=surrogate,
        method=method,
        random_state=random_state,
    )

    feature_cols = [c for c in df.columns if c != target]
    rows = []
    for idx in chosen:
        row = {"index": int(idx)}
        for col in feature_cols:
            row[col] = df.iloc[idx][col]
        row["pred_mean"] = float(mean[idx])
        row["pred_std"] = float(std[idx])
        row["acq"] = float(scores[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def build_observed_summary(df, history_rows, target):
    observed_df = df.loc[df[target].notna()].copy().reset_index().rename(columns={"index": "data_index"})
    observed_df = observed_df.rename(columns={target: "observed_y"})
    if not history_rows:
        observed_df.insert(0, "round", 0)
        observed_df.insert(1, "source", "initial_known")
        observed_df["pred_mean"] = np.nan
        observed_df["pred_std"] = np.nan
        observed_df["acq"] = np.nan
        return observed_df

    history_df = pd.DataFrame(history_rows).copy()
    history_df["data_index"] = pd.to_numeric(history_df["index"], errors="coerce").astype("Int64")
    history_df = history_df.drop(columns=["index"])
    history_df["source"] = "manual_feedback"
    history_df = history_df.rename(columns={"observed_y": "observed_y_logged"})

    summary_df = observed_df.merge(history_df, on="data_index", how="left")
    if "observed_y_logged" in summary_df.columns:
        summary_df = summary_df.drop(columns=["observed_y_logged"])
    summary_df["round"] = summary_df["round"].fillna(0).astype(int)
    summary_df["source"] = summary_df["source"].fillna("initial_known")
    ordered_cols = ["round", "source", "data_index"]
    ordered_cols += [c for c in summary_df.columns if c not in ordered_cols]
    return summary_df[ordered_cols]


def save_outputs(df, history_rows, progress_output, history_output, summary_output, target):
    progress_path = Path(progress_output)
    history_path = Path(history_output)
    summary_path = Path(summary_output)
    df.to_csv(progress_path, index=False)
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    build_observed_summary(df, history_rows, target).to_csv(summary_path, index=False)
    print(f"\nProgress saved to: {progress_path}")
    print(f"History saved to: {history_path}")
    print(f"Observed summary saved to: {summary_path}")


def collect_batch_results(df, target, suggestions, round_no, history_rows):
    if suggestions.empty:
        return False

    print("\nEnter the true yields for this batch. Percentage values such as 42% are supported.")
    for _, row in suggestions.iterrows():
        idx = int(row["index"])
        condition_text = describe_condition_row(row, target)
        print("\n----------------------------------------")
        print(f"Condition: {condition_text}")
        print(
            f"Model prediction: pred_mean={float(row['pred_mean']):.4g}, "
            f"pred_std={float(row['pred_std']):.4g}, acq={float(row['acq']):.4g}"
        )
        while True:
            text = input(f"Enter the true {target} for this condition, or press q to end this session: ").strip()
            if text.lower() == "q":
                return True
            try:
                observed = parse_yield_text(text)
                df.at[idx, target] = observed
                history_rows.append(
                    {
                        "round": int(round_no),
                        "index": idx,
                        "observed_y": float(observed),
                        "pred_mean": float(row["pred_mean"]),
                        "pred_std": float(row["pred_std"]),
                        "acq": float(row["acq"]),
                    }
                )
                break
            except ValueError as exc:
                print(f"Invalid input: {exc}")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Interactive batch Bayesian optimization: recommend a new batch each round, then manually enter the true yields to continue."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to the candidate-condition CSV file")
    parser.add_argument("--target", type=str, default="Yield", help="Name of the target column")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of experiments recommended per round")
    parser.add_argument("--acq", type=str, default="EI", help="Acquisition function: EI/PI/UCB/LCB")
    parser.add_argument("--xi", type=float, default=0.0, help="Exploration parameter for EI/PI")
    parser.add_argument("--kappa", type=float, default=2.0, help="Exploration parameter for UCB/LCB")
    parser.add_argument("--surrogate", type=str, default="linear", help="Surrogate model: linear/gp")
    parser.add_argument("--method", type=str, default="kriging", help="Batch strategy: kriging/topk/thompson")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--max_rounds", type=int, default=0, help="Maximum number of recommendation rounds; 0 means unlimited")
    parser.add_argument(
        "--progress_output",
        type=str,
        default="interactive_progress.csv",
        help="Output CSV for the current experiment progress",
    )
    parser.add_argument(
        "--history_output",
        type=str,
        default="interactive_history.csv",
        help="Output CSV for the recommendation history of this session",
    )
    parser.add_argument(
        "--summary_output",
        type=str,
        default="interactive_observed_summary.csv",
        help="Output CSV summarizing all observed conditions and yields",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    normalize_df(df)
    ensure_target_column(df, args.target)

    print("Interactive batch Bayesian optimization started.")
    print(f"Total candidates: {len(df)}")
    print(f"Known observations: {int(df[args.target].notna().sum())}")
    print(f"Target column: {args.target}")

    if df[args.target].notna().sum() == 0:
        prompt_initial_observations(df, args.target)

    history_rows = []
    round_no = 1

    while True:
        known_count = int(df[args.target].notna().sum())
        remaining = int(df[args.target].isna().sum())
        print("\n========================================")
        print(f"Round {round_no} recommendation started")
        print(f"Known observations: {known_count}")
        print(f"Untested candidates: {remaining}")
        print(f"Current best {args.target}: {float(df[args.target].max()):.4g}")

        suggestions = recommend_batch(
            df=df,
            target=args.target,
            batch_size=args.batch_size,
            acquisition=args.acq,
            xi=args.xi,
            kappa=args.kappa,
            surrogate=args.surrogate,
            method=args.method,
            random_state=args.random_state + round_no - 1,
        )

        if suggestions.empty:
            print("\nAll candidate conditions already have observed results. The workflow is complete.")
            break

        print("\nRecommended experimental conditions:")
        display_cols = [c for c in suggestions.columns if c != "index"]
        print(suggestions[display_cols].to_string(index=False))

        should_quit = collect_batch_results(
            df=df,
            target=args.target,
            suggestions=suggestions,
            round_no=round_no,
            history_rows=history_rows,
        )
        save_outputs(
            df=df,
            history_rows=history_rows,
            progress_output=args.progress_output,
            history_output=args.history_output,
            summary_output=args.summary_output,
            target=args.target,
        )
        if should_quit:
            print("\nSession ended as requested. Next time, you can resume directly by passing the progress file to --csv.")
            break

        if df[args.target].isna().sum() == 0:
            print("\nAll candidate conditions already have observed results. The workflow is complete.")
            break

        if args.max_rounds > 0 and round_no >= args.max_rounds:
            print(f"\nThe maximum number of recommendation rounds ({args.max_rounds}) has been reached. The workflow is complete.")
            break

        cmd = input("\nPress Enter to continue to the next round, or press q to end this session: ").strip().lower()
        if cmd == "q":
            print("Session ended. Next time, you can resume directly by passing the progress file to --csv.")
            break
        round_no += 1


if __name__ == "__main__":
    main()
