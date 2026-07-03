import os
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

PATH_TO_CSV = "data/"
PATH_TO_BLAST = "blast/"
PATH_TO_FASTA = "blast/downsample_runs/"

FULL_DATASET_FILE = f"{PATH_TO_CSV}uniprot_tf_three_class_labeled.csv"

LABEL_COL = "tf_label"

NON_TF = 1
NON_EXPERIMENTAL_TF = 2
EXPERIMENTAL_TF = 3

N_ITERATIONS = 54
TEST_SIZE = 0.20
RANDOM_STATE = 42
NUM_THREADS = 8


# ============================================================
# HELPERS
# ============================================================

def write_fasta(df, filename):
    with open(filename, "w") as writer:
        for _, row in df.iterrows():
            entry = str(row["Entry"]).strip()
            seq = str(row["Sequence"]).strip()

            writer.write(f">{entry}\n")

            for i in range(0, len(seq), 80):
                writer.write(seq[i:i + 80] + "\n")


# ============================================================
# MAIN
# ============================================================

os.makedirs(PATH_TO_BLAST, exist_ok=True)
os.makedirs(PATH_TO_FASTA, exist_ok=True)

print("Loading full 3-class dataset...")
df = pd.read_csv(FULL_DATASET_FILE)

df = df[["Entry", "Sequence", LABEL_COL]].dropna().copy()
df[LABEL_COL] = df[LABEL_COL].astype(int)

# Remove duplicates before splitting.
df = df.drop_duplicates("Entry").copy()
df = df.drop_duplicates("Sequence").copy()

df_13 = df[df[LABEL_COL].isin([NON_TF, EXPERIMENTAL_TF])].copy()
df_2 = df[df[LABEL_COL] == NON_EXPERIMENTAL_TF].copy()

print("\nFull label counts:")
print(df[LABEL_COL].value_counts().sort_index())

print("\nLabels 1 and 3 counts:")
print(df_13[LABEL_COL].value_counts().sort_index())

print("\nLabel 2 count:")
print(len(df_2))


# ------------------------------------------------------------
# Random stratified split for labels 1 and 3
# ------------------------------------------------------------

train_13_df, test_13_df = train_test_split(
    df_13,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df_13[LABEL_COL],
)

train_13_df = train_13_df.reset_index(drop=True)
test_13_df = test_13_df.reset_index(drop=True)
df_2 = df_2.reset_index(drop=True)

train_label1_df = train_13_df[train_13_df[LABEL_COL] == NON_TF].copy()
train_label3_df = train_13_df[train_13_df[LABEL_COL] == EXPERIMENTAL_TF].copy()

print("\nTrain split label counts:")
print(train_13_df[LABEL_COL].value_counts().sort_index())

print("\nTest split label counts:")
print(test_13_df[LABEL_COL].value_counts().sort_index())


# ------------------------------------------------------------
# Save fixed evaluation files
# ------------------------------------------------------------

eval_df = pd.concat([test_13_df, df_2], ignore_index=True)
eval_df = eval_df.drop_duplicates("Entry").copy()

test_13_df.to_csv(f"{PATH_TO_CSV}downsample_test_13.csv", index=False)
df_2.to_csv(f"{PATH_TO_CSV}downsample_label2.csv", index=False)
eval_df.to_csv(f"{PATH_TO_CSV}downsample_eval_test13_and_label2.csv", index=False)

write_fasta(eval_df, f"{PATH_TO_FASTA}eval_test13_and_label2.fsa")

print("\nSaved fixed evaluation FASTA:")
print(f"{PATH_TO_FASTA}eval_test13_and_label2.fsa")


# ------------------------------------------------------------
# Make 54 non-overlapping label-1 batches
# ------------------------------------------------------------

train_label1_shuffled = train_label1_df.sample(
    frac=1,
    random_state=RANDOM_STATE,
).reset_index(drop=True)

batch_size = len(train_label1_shuffled) // N_ITERATIONS

label1_batches = []

for i in range(N_ITERATIONS):
    start = i * batch_size

    if i == N_ITERATIONS - 1:
        end = len(train_label1_shuffled)
    else:
        end = (i + 1) * batch_size

    batch_df = train_label1_shuffled.iloc[start:end].copy()
    label1_batches.append(batch_df)

print("\nDownsample setup:")
print(f"Number of runs: {N_ITERATIONS}")
print(f"All label 3 proteins per run: {len(train_label3_df)}")
print(f"Label 1 batch size range: {min(len(b) for b in label1_batches)} - {max(len(b) for b in label1_batches)}")


# ------------------------------------------------------------
# Write train FASTAs and BLAST scripts
# ------------------------------------------------------------

run_all_lines = ["#!/bin/bash\n", "set -e\n\n"]

for run_idx, label1_batch in enumerate(label1_batches, start=1):
    run_name = f"run_{run_idx:02d}"

    train_iter_df = pd.concat(
        [label1_batch, train_label3_df],
        ignore_index=True,
    )

    train_csv = f"{PATH_TO_CSV}downsample_{run_name}_train.csv"
    train_fasta = f"{PATH_TO_FASTA}{run_name}_train.fsa"
    blast_script = f"{PATH_TO_BLAST}blast_{run_name}.sh"

    train_iter_df.to_csv(train_csv, index=False)
    write_fasta(train_iter_df, train_fasta)

    script = f"""#!/bin/bash
set -e

cd downsample_runs

makeblastdb \\
  -in {run_name}_train.fsa \\
  -dbtype prot \\
  -parse_seqids \\
  -out {run_name}_train_db

blastp \\
  -query eval_test13_and_label2.fsa \\
  -db {run_name}_train_db \\
  -max_target_seqs 5 \\
  -outfmt 6 \\
  -out {run_name}_eval_vs_train.tsv \\
  -num_threads {NUM_THREADS}

mv {run_name}_eval_vs_train.tsv ../../data/
"""

    with open(blast_script, "w") as file:
        file.write(script)

    os.chmod(blast_script, 0o755)

    run_all_lines.append(f"echo 'Running {run_name}...'\n")
    run_all_lines.append(f"bash blast_{run_name}.sh\n\n")

run_all_path = f"{PATH_TO_BLAST}run_all_downsample_blast.sh"

with open(run_all_path, "w") as file:
    file.writelines(run_all_lines)

os.chmod(run_all_path, 0o755)

print("\nSuccess.")
print("Generated:")
print(f"- {N_ITERATIONS} downsampled train CSVs in data/")
print(f"- {N_ITERATIONS} train FASTAs in blast/downsample_runs/")
print(f"- {N_ITERATIONS} BLAST scripts in blast/")
print(f"- one run-all script: {run_all_path}")
