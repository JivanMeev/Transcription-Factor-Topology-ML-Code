import os
import pandas as pd

from sklearn.metrics import accuracy_score, matthews_corrcoef


# ============================================================
# CONFIG
# ============================================================

PATH_TO_CSV = "data/"

N_ITERATIONS = 54

LABEL_COL = "tf_label"

NON_TF = 1
NON_EXPERIMENTAL_TF = 2
EXPERIMENTAL_TF = 3

TEST_13_FILE = f"{PATH_TO_CSV}downsample_test_13.csv"
LABEL_2_FILE = f"{PATH_TO_CSV}downsample_label2.csv"

TEST13_OUTPUT = f"{PATH_TO_CSV}test13_per_protein_batch_votes.csv"
LABEL2_OUTPUT = f"{PATH_TO_CSV}label2_per_protein_batch_votes.csv"


# ============================================================
# HELPERS
# ============================================================

def read_blast_predictions(blast_file, train_label_map):
    """
    Reads one BLAST output file and returns predictions.

    Expected BLAST direction:
        qseqid = evaluation/test protein
        sseqid = nearest training protein

    So each test protein inherits the label of its closest training hit.
    """

    blast_cols = [
        "qseqid",
        "sseqid",
        "pident",
        "length",
        "mismatch",
        "gapopen",
        "qstart",
        "qend",
        "sstart",
        "send",
        "evalue",
        "bitscore",
    ]

    if not os.path.exists(blast_file):
        raise FileNotFoundError(f"Missing BLAST file: {blast_file}")

    if os.path.getsize(blast_file) == 0:
        return {}

    blast_df = pd.read_csv(blast_file, sep="\t", names=blast_cols)

    # Only keep hits where the subject is actually from this run's train set.
    blast_df = blast_df[blast_df["sseqid"].isin(train_label_map.keys())].copy()

    if blast_df.empty:
        return {}

    # Best hit for each evaluation protein.
    blast_df = blast_df.sort_values(
        by=["qseqid", "bitscore", "pident"],
        ascending=[True, False, False],
    )

    closest_hits = blast_df.groupby("qseqid").first().reset_index()

    closest_hits["predicted_label"] = closest_hits["sseqid"].map(train_label_map)

    predictions = dict(zip(closest_hits["qseqid"], closest_hits["predicted_label"]))

    return predictions


def predict(entry, predictions):
    """
    If a protein has no BLAST hit, default to non-TF.
    """
    return predictions.get(entry, NON_TF)


def safe_mcc(y_true, y_pred):

    if len(set(y_true)) < 2:
        return 0.0

    return matthews_corrcoef(y_true, y_pred)


# ============================================================
# LOAD FIXED TEST SETS
# ============================================================

print("Loading fixed test files...")

test13_df = pd.read_csv(TEST_13_FILE)
label2_df = pd.read_csv(LABEL_2_FILE)

test13_df[LABEL_COL] = test13_df[LABEL_COL].astype(int)
label2_df[LABEL_COL] = label2_df[LABEL_COL].astype(int)

print("\nTest 1 set label counts:")
print(test13_df[LABEL_COL].value_counts().sort_index())

print("\nLabel 2 set count:")
print(len(label2_df))


# ============================================================
# SET UP PER-PROTEIN TRACKING TABLES
# ============================================================

test13_votes = {}

for _, row in test13_df.iterrows():
    entry = row["Entry"]
    true_label = int(row[LABEL_COL])

    test13_votes[entry] = {
        "Entry": entry,
        "Sequence": row["Sequence"],
        "true_label": true_label,
        "correct_batches": 0,
        "wrong_batches": 0,
        "times_predicted_non_TF": 0,
        "times_predicted_TF": 0,
    }


label2_votes = {}

for _, row in label2_df.iterrows():
    entry = row["Entry"]

    label2_votes[entry] = {
        "Entry": entry,
        "Sequence": row["Sequence"],
        "true_label": NON_EXPERIMENTAL_TF,
        "times_marked_TF_like": 0,
        "times_marked_non_TF_like": 0,
        "correct_batches_if_TF_like_is_correct": 0,
        "wrong_batches_if_TF_like_is_correct": 0,
    }


# ============================================================
# READ EACH FINISHED BLAST RUN
# ============================================================

run_metric_rows = []

for run_idx in range(1, N_ITERATIONS + 1):
    run_name = f"run_{run_idx:02d}"

    train_file = f"{PATH_TO_CSV}downsample_{run_name}_train.csv"
    blast_file = f"{PATH_TO_CSV}{run_name}_eval_vs_train.tsv"

    print(f"\nReading {run_name}...")

    train_df = pd.read_csv(train_file)
    train_df[LABEL_COL] = train_df[LABEL_COL].astype(int)

    train_label_map = dict(zip(train_df["Entry"], train_df[LABEL_COL]))

    predictions = read_blast_predictions(blast_file, train_label_map)

    # --------------------------------------------------------
    # TEST 1: labels 1 and 3
    # --------------------------------------------------------

    y_true_13 = []
    y_pred_13 = []

    for _, row in test13_df.iterrows():
        entry = row["Entry"]
        true_label = int(row[LABEL_COL])
        pred_label = int(predict(entry, predictions))

        y_true_13.append(true_label)
        y_pred_13.append(pred_label)

        if pred_label == NON_TF:
            test13_votes[entry]["times_predicted_non_TF"] += 1

        if pred_label == EXPERIMENTAL_TF:
            test13_votes[entry]["times_predicted_TF"] += 1

        if pred_label == true_label:
            test13_votes[entry]["correct_batches"] += 1
        else:
            test13_votes[entry]["wrong_batches"] += 1

    test1_accuracy = accuracy_score(y_true_13, y_pred_13)
    test1_mcc = safe_mcc(y_true_13, y_pred_13)
    test1_guessed_tfs = sum(1 for pred in y_pred_13 if pred == EXPERIMENTAL_TF)

    # --------------------------------------------------------
    # TEST 2: label 2 proteins
    # --------------------------------------------------------

    y_true_2 = []
    y_pred_2 = []

    for _, row in label2_df.iterrows():
        entry = row["Entry"]
        pred_label = int(predict(entry, predictions))

        # For label 2 analysis, predicted label 3 means "TF-like."
        pred_tf_like = 1 if pred_label == EXPERIMENTAL_TF else 0

        # We are treating label 2 as a TF group, so TF-like is counted as correct.
        true_tf_like = 1

        y_true_2.append(true_tf_like)
        y_pred_2.append(pred_tf_like)

        if pred_tf_like == 1:
            label2_votes[entry]["times_marked_TF_like"] += 1
            label2_votes[entry]["correct_batches_if_TF_like_is_correct"] += 1
        else:
            label2_votes[entry]["times_marked_non_TF_like"] += 1
            label2_votes[entry]["wrong_batches_if_TF_like_is_correct"] += 1

    test2_accuracy = accuracy_score(y_true_2, y_pred_2) if len(y_true_2) else 0.0
    test2_guessed_tfs = sum(y_pred_2)

    print("Test 1 accuracy:", round(test1_accuracy, 4))
    print("Test 1 MCC:", round(test1_mcc, 4))
    print("Test 1 guessed TFs:", test1_guessed_tfs)

    print("Test 2 total label 2s:", len(label2_df))
    print("Test 2 accuracy:", round(test2_accuracy, 4))
    print("Test 2 guessed TFs:", test2_guessed_tfs)

    run_metric_rows.append(
        {
            "run": run_idx,
            "test1_accuracy": test1_accuracy,
            "test1_mcc": test1_mcc,
            "test1_model_guessed_tfs": test1_guessed_tfs,
            "test2_total_label2_present": len(label2_df),
            "test2_accuracy": test2_accuracy,
            "test2_model_guessed_tfs": test2_guessed_tfs,
        }
    )


# ============================================================
# SAVE PER-PROTEIN VOTE CSV 1:
# LABEL 1 AND 3 TEST SET
# ============================================================

test13_vote_df = pd.DataFrame(test13_votes.values())

test13_vote_df["total_batches"] = (
    test13_vote_df["correct_batches"] + test13_vote_df["wrong_batches"]
)

test13_vote_df["fraction_correct"] = (
    test13_vote_df["correct_batches"] / test13_vote_df["total_batches"]
)

test13_vote_df["fraction_wrong"] = (
    test13_vote_df["wrong_batches"] / test13_vote_df["total_batches"]
)

test13_vote_df["fraction_predicted_TF"] = (
    test13_vote_df["times_predicted_TF"] / test13_vote_df["total_batches"]
)

test13_vote_df["fraction_predicted_non_TF"] = (
    test13_vote_df["times_predicted_non_TF"] / test13_vote_df["total_batches"]
)

test13_vote_df = test13_vote_df.sort_values(
    by=["true_label", "fraction_correct"],
    ascending=[True, False],
)

test13_vote_df.to_csv(TEST13_OUTPUT, index=False)


# ============================================================
# SAVE PER-PROTEIN VOTE CSV 2:
# LABEL 2 SET
# ============================================================

label2_vote_df = pd.DataFrame(label2_votes.values())

label2_vote_df["total_batches"] = (
    label2_vote_df["times_marked_TF_like"]
    + label2_vote_df["times_marked_non_TF_like"]
)

label2_vote_df["fraction_marked_TF_like"] = (
    label2_vote_df["times_marked_TF_like"] / label2_vote_df["total_batches"]
)

label2_vote_df["fraction_marked_non_TF_like"] = (
    label2_vote_df["times_marked_non_TF_like"] / label2_vote_df["total_batches"]
)

label2_vote_df["fraction_correct_if_TF_like_is_correct"] = (
    label2_vote_df["correct_batches_if_TF_like_is_correct"]
    / label2_vote_df["total_batches"]
)

label2_vote_df["fraction_wrong_if_TF_like_is_correct"] = (
    label2_vote_df["wrong_batches_if_TF_like_is_correct"]
    / label2_vote_df["total_batches"]
)

label2_vote_df = label2_vote_df.sort_values(
    by="fraction_marked_TF_like",
    ascending=False,
)

label2_vote_df.to_csv(LABEL2_OUTPUT, index=False)


# ============================================================
# FINAL SUMMARY
# ============================================================

metrics_df = pd.DataFrame(run_metric_rows)

print("\n" + "=" * 70)
print("FINAL SUMMARY ACROSS ALL BATCHES")
print("=" * 70)

print("\nTEST 1: labels 1 and 3")
print(f"Average accuracy: {metrics_df['test1_accuracy'].mean():.4f}")
print(f"Average MCC: {metrics_df['test1_mcc'].mean():.4f}")
print(f"Average model guessed TFs: {metrics_df['test1_model_guessed_tfs'].mean():.2f}")

print("\nTEST 2: label 2")
print(f"Total label 2s present: {len(label2_df)}")
print(f"Average accuracy: {metrics_df['test2_accuracy'].mean():.4f}")
print(f"Average model guessed TFs: {metrics_df['test2_model_guessed_tfs'].mean():.2f}")

print("\nSaved CSV files:")
print(TEST13_OUTPUT)
print(LABEL2_OUTPUT)

print("\nDone.")
