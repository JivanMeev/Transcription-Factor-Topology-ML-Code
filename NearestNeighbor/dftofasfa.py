import os

import pandas as pd


# Write one dataframe row per FASTA record.
def dftofasfa(df, filename):
    num = df.shape[0]

    with open(filename, "w") as writer:
        for i in range(num):
            entry = df["Entry"].iloc[i]
            seq = df["Sequence"].iloc[i]

            if i:
                writer.write(f"\n>{entry}\n{seq}")
            else:
                writer.write(f">{entry}\n{seq}")


# Main folders used by the CSV-to-BLAST step.
PATH_TO_CSV = "data/"
PATH_TO_BLAST = "blast/"
PATH_TO_FASTA = "blast/data/"

# Make sure the BLAST folders exist before writing files into them.
os.makedirs(PATH_TO_BLAST, exist_ok=True)
os.makedirs(PATH_TO_FASTA, exist_ok=True)

# Species names already match the sanitized file names from make_data.py.
good_species = [
    "escherichia_coli",
    "mycobacterium_tuberculosis",
    "homo_sapiens",
    "saccharomyces_cerevisiae",
    "methanocaldococcus_jannaschii",
]

# File suffixes used for the generated train/validation/test CSVs.
suffix = "_tf_all"
a_second_suffix = "50_tf_all"


print("Converting dataframes into raw FASTA sequence profiles...")
for name in good_species:
    print(f" -> Generating .fsa files for: {name}")

    train_set = pd.read_csv(f"{PATH_TO_CSV}{name}_train{suffix}.csv")
    valid_set = pd.read_csv(f"{PATH_TO_CSV}{name}_valid{suffix}.csv")
    test_set = pd.read_csv(f"{PATH_TO_CSV}{name}_test_{a_second_suffix}.csv")

    dftofasfa(train_set, f"{PATH_TO_FASTA}{name}_train.fsa")
    dftofasfa(valid_set, f"{PATH_TO_FASTA}{name}_valid.fsa")
    dftofasfa(test_set, f"{PATH_TO_FASTA}{name}_test.fsa")


print("\nWriting optimized BLAST alignment shell scripts...")
for name in good_species:
    script = f"""#!/bin/bash
# Local Database Compile
makeblastdb -in data/{name}_test.fsa -title "{name} Test DNA" -dbtype prot -parse_seqids

# Multi-threaded Lookups (Utilizing 8 cores on your Mac CPU)
blastp -query data/{name}_train.fsa -db data/{name}_test.fsa -max_target_seqs 5 -outfmt 6 -out {name}_train_test.csv -num_threads 8
blastp -query data/{name}_valid.fsa -db data/{name}_test.fsa -max_target_seqs 5 -outfmt 6 -out {name}_valid_test.csv -num_threads 8

# Route alignment matrices back to data tracking pipeline
mv {name}_train_test.csv ../data/
mv {name}_valid_test.csv ../data/
"""

    with open(f"{PATH_TO_BLAST}blast_{name}.sh", "w") as file:
        file.write(script)

print("Success! FASTA records generated and execution bash scripts compiled.")
