import numpy as np
import pandas as pd
from goatools.godag.go_tasks import get_go2children
from goatools.obo_parser import GODag


# Collect the starting GO term plus all descendant terms in the GO graph.
def get_all_children(start):
    godag = GODag("go-basic.obo", optional_attrs={"relationship"})
    optional_relationships = set("part_of")
    children_isa_partof = get_go2children(godag, optional_relationships)

    def get_children(parent):
        family = {parent}
        children = children_isa_partof.get(parent, {})

        for child in children:
            family.update(get_children(child))

        return family

    return get_children(start)


# Build a labeler that marks rows containing the target GO term or one of its children.
def go_id_labeler(go_id="GO:0003700"):
    related_go_codes = get_all_children(go_id)

    def go_id_to_indicator(x):
        if pd.isna(x):
            return 0

        for go_code in related_go_codes:
            if go_code in x:
                return 1

        return 0

    return go_id_to_indicator


# Keep file names simple and shell-friendly.
def sanitize_string(the_string):
    return (the_string.replace(" ", "_")).lower()


# Remove duplicate sequences that have conflicting TF labels.
def sequences_unsure(df):
    df_mean = df.groupby("Sequence")["transcription_factor"].mean().reset_index()
    tol = 1e-5
    seqs = df_mean[
        np.logical_and(
            df_mean["transcription_factor"] < (1 - tol),
            df_mean["transcription_factor"] > tol,
        )
    ]["Sequence"].values

    return seqs


PATH_TO_CSV = "data/"


print("Loading raw UniProt data...")
df_all = pd.read_csv(f"{PATH_TO_CSV}uniprot_data.tab", sep="\t")

# UniProt exports sometimes use slightly different names for this column.
pe_col = [
    c
    for c in df_all.columns
    if c.lower().replace("_", " ").strip() in ["protein existence", "pe"]
]

if pe_col:
    pe_col = pe_col[0]
    print(f"Filtering for purely experimentally verified entries using column: '{pe_col}'...")

    df_all = df_all[
        df_all[pe_col]
        .astype(str)
        .str.contains("protein level|1", case=False, na=False)
    ]
    print(f"Filter success! Total dataset pool set to {len(df_all)} entries.")
else:
    print("\nWARNING: Protein existence column not found!")
    print(f"Available columns in your file are: {list(df_all.columns)}")
    print(
        "The script is skipping the PE filter. You may need to re-download your "
        "UniProt data with 'Protein existence' selected.\n"
    )

# Normalize the GO column name so the rest of the script can use one label.
df_all = df_all.rename(columns={"Gene Ontology (GO)": "Gene ontology IDs"})


# Pull the binomial species name from the longer organism string.
def extract_clean_species(organism_string):
    if pd.isna(organism_string):
        return "Unknown"

    words = str(organism_string).split()

    if len(words) >= 2:
        return f"{words[0]} {words[1]}"

    return words[0] if words else "Unknown"


print("Parsing taxonomy metrics...")
df_all["Taxonomic lineage (SPECIES)"] = df_all["Organism"].apply(extract_clean_species)

GO_CODE = "GO:0003700"  # Transcription factor activity.
label = "transcription_factor"

print("Labeling transcription factors...")
df_all[label] = df_all["Gene ontology IDs"].apply(go_id_labeler(GO_CODE))
df_all = df_all[np.logical_not(df_all["Sequence"].isin(sequences_unsure(df_all)))]

# Species used as held-out test organisms for the benchmark runs.
species_to_test = [
    "Escherichia coli",
    "Mycobacterium tuberculosis",
    "Homo sapiens",
    "Saccharomyces cerevisiae",
    "Methanocaldococcus jannaschii",
]
valid = "Bacillus subtilis"


# Build train/validation/test splits using one held-out species at a time.
def train_valid_test(
    df,
    hold_out,
    valid,
    label="transcription_factor",
    level="species",
    remove_test_data_in_train=True,
):
    col_name = f"Taxonomic lineage ({level.upper()})"
    train_set = df[df[col_name] != hold_out][["Entry", "Sequence", label, "Length", col_name]]
    test_set = df[df[col_name] == hold_out][["Entry", "Sequence", label, "Length", col_name]]

    train_set.drop_duplicates("Sequence", inplace=True, ignore_index=True)
    test_set.drop_duplicates("Sequence", inplace=True, ignore_index=True)

    if remove_test_data_in_train:
        test_set = test_set[
            np.logical_not(test_set["Sequence"].isin(train_set["Sequence"]))
        ]

    valid_set = train_set[train_set[col_name] == valid]
    train_set = train_set[train_set[col_name] != valid]

    return train_set, valid_set, test_set


suffix = "_tf_all"
a_second_suffix = "50_tf_all"


print("Splitting datasets and compiling CSV matrix blocks...")
for name in species_to_test:
    print(f"-> Processing dataset partitions holding out: {name}...")

    train_set, valid_set, test_set = train_valid_test(df_all, name, valid)
    new_name = sanitize_string(name)

    train_set.to_csv(f"{PATH_TO_CSV}{new_name}_train{suffix}.csv", index=False)
    valid_set.to_csv(f"{PATH_TO_CSV}{new_name}_valid{suffix}.csv", index=False)
    test_set.to_csv(f"{PATH_TO_CSV}{new_name}_test{suffix}.csv", index=False)

    train_set, valid_set, test_set = train_valid_test(
        df_all,
        name,
        valid,
        remove_test_data_in_train=False,
    )
    test_set.to_csv(f"{PATH_TO_CSV}{new_name}_test_{a_second_suffix}.csv", index=False)

print("\nSuccess! All cross-kingdom dataset matrices generated perfectly.")
