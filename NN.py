import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef


# Files are expected to already be in the data folder.
PATH_TO_CSV = "data/"
name = "methanocaldococcus_jannaschii"  # Target organism for this run.
suffix = "_tf_all"
a_second_suffix = "50_tf_all"


print(f"Loading datasets for {name}...")

# Ground-truth labels generated earlier by make_data.py.
test_df = pd.read_csv(f"{PATH_TO_CSV}{name}_test_{a_second_suffix}.csv")
train_df = pd.read_csv(f"{PATH_TO_CSV}{name}_train{suffix}.csv")

# BLAST outfmt 6 columns, in the order BLAST writes them.
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
blast_df = pd.read_csv(
    f"{PATH_TO_CSV}{name}_train_test.csv",
    sep="\t",
    names=blast_cols,
)


print("Computing 1-Nearest Neighbor predictions...")

# Lookup table from training protein ID to its TF/non-TF label.
train_label_map = dict(zip(train_df["Entry"], train_df["transcription_factor"]))

# The closest neighbor is the highest-percent-identity BLAST hit for each test sequence.
# In this BLAST setup, qseqid is the training sequence and sseqid is the test sequence.
blast_sorted = blast_df.sort_values(by="pident", ascending=False)
closest_hits = blast_sorted.groupby("sseqid").first().reset_index()

# Each test sequence inherits the label of its nearest training-sequence match.
closest_hits["predicted_label"] = closest_hits["qseqid"].map(train_label_map)
predictions_map = dict(zip(closest_hits["sseqid"], closest_hits["predicted_label"]))


# Store labels for the final metrics.
y_true = []
y_pred = []

# These are the predicted TFs that are strong enough to pass into the TDA work.
high_confidence_tfs = []

for idx, row in test_df.iterrows():
    entry_id = row["Entry"]
    true_label = row["transcription_factor"]

    # No BLAST hit means this sequence gets the safe default: not a transcription factor.
    pred_label = predictions_map.get(entry_id, 0)

    y_true.append(true_label)
    y_pred.append(pred_label)

    # Keep predicted TFs only when their nearest-neighbor identity clears the threshold.
    if pred_label == 1:
        match_row = closest_hits[closest_hits["sseqid"] == entry_id]
        if not match_row.empty and match_row["pident"].values[0] >= 40.0:
            high_confidence_tfs.append(
                {
                    "Entry": entry_id,
                    "Sequence": row["Sequence"],
                    "Percent_Identity": match_row["pident"].values[0],
                }
            )


# Standard classification metrics for the 1-NN baseline.
acc = accuracy_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

print("\n=============================================")
print(f"1-NN PERFORMANCE RESULTS FOR: {name}")
print("=============================================")
print(f"Accuracy Score: {acc:.4f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
print(f"Total Test Sequences Evaluated: {len(y_true)}")
print("=============================================\n")

# Save the filtered TF candidates for the next stage of analysis.
tda_df = pd.DataFrame(high_confidence_tfs)
tda_output_path = f"{PATH_TO_CSV}{name}_expanded_tfs_for_tda.csv"
tda_df.to_csv(tda_output_path, index=False)

print(f"Success: Extracted {len(tda_df)} high-confidence transcription factors.")
print(f"Cleaned sequence point-cloud file saved to: {tda_output_path}")
