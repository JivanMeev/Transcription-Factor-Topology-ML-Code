import time
import numpy as np
import pandas as pd
import requests

from goatools.godag.go_tasks import get_go2children
from goatools.obo_parser import GODag


# ============================================================
# CONFIG
# ============================================================

PATH_TO_CSV = "data/"

UNIPROT_INPUT_FILE = f"{PATH_TO_CSV}uniprot_data.tab"

FULL_OUTPUT_FILE = f"{PATH_TO_CSV}uniprot_tf_three_class_labeled.csv"
SPECIES_COUNTS_OUTPUT_FILE = f"{PATH_TO_CSV}uniprot_tf_three_class_label_counts_by_species.csv"
QUICKGO_CACHE_FILE = f"{PATH_TO_CSV}quickgo_experimental_tf_accessions.csv"

GO_CODE = "GO:0003700"  # DNA-binding transcription factor activity

BINARY_TF_COL = "transcription_factor"

# Final 3-class label:
# 1 = non-TF
# 2 = TF, but not experimentally verified
# 3 = experimentally verified TF
THREE_CLASS_LABEL_COL = "tf_label"
THREE_CLASS_LABEL_NAME_COL = "tf_label_name"

QUICKGO_ANNOTATION_URL = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"

# Experimental evidence ECO codes:
# EXP = ECO:0000269
# IDA = ECO:0000314
# IMP = ECO:0000315
# IGI = ECO:0000316
# IPI = ECO:0000353
# IEP = ECO:0000270
EXPERIMENTAL_ECO_CODES = [
    "ECO:0000269",
    "ECO:0000314",
    "ECO:0000315",
    "ECO:0000316",
    "ECO:0000353",
    "ECO:0000270",
]


# ============================================================
# GO TERM FAMILY HELPERS
# ============================================================

def get_all_children(start):
    """
    Collect the starting GO term plus all descendant terms in the GO graph.

    This uses is_a relationships plus part_of relationships.
    """

    godag = GODag("go-basic.obo", optional_attrs={"relationship"})

    # Important: use {"part_of"}, not set("part_of").
    optional_relationships = {"part_of"}

    children_isa_partof = get_go2children(godag, optional_relationships)

    def get_children(parent):
        family = {parent}
        children = children_isa_partof.get(parent, {})

        for child in children:
            family.update(get_children(child))

        return family

    return get_children(start)


def go_id_labeler(go_id=GO_CODE):
    """
    Build a function that labels a row as TF if its GO column contains
    GO:0003700 or one of its child/descendant terms.
    """

    related_go_codes = get_all_children(go_id)

    print(f"Total GO terms in TF family: {len(related_go_codes)}")

    def go_id_to_indicator(x):
        if pd.isna(x):
            return 0

        x = str(x)

        for go_code in related_go_codes:
            if go_code in x:
                return 1

        return 0

    return go_id_to_indicator


# ============================================================
# STRING / COLUMN HELPERS
# ============================================================

def normalize_uniprot_accession(accession):
    """
    Normalize accessions so UniProt and QuickGO can be matched.

    Examples:
        UniProtKB:P12345 -> P12345
        P12345-2         -> P12345
    """

    if pd.isna(accession):
        return None

    value = str(accession).strip()

    if ":" in value:
        value = value.split(":")[-1]

    value = value.split("-")[0]

    return value if value else None


def extract_uniprot_accession(gene_product_id):
    return normalize_uniprot_accession(gene_product_id)


def extract_clean_species(organism_string):
    """
    Pull the binomial species name from the longer organism string.

    Example:
        Homo sapiens (Human) -> Homo sapiens
    """

    if pd.isna(organism_string):
        return "Unknown"

    words = str(organism_string).split()

    if len(words) >= 2:
        return f"{words[0]} {words[1]}"

    return words[0] if words else "Unknown"


def normalize_go_column_name(df):
    """
    UniProt TSV files may use slightly different GO column names.
    This renames the detected GO column to 'Gene ontology IDs'.
    """

    possible_go_columns = [
        "Gene Ontology (GO)",
        "Gene ontology IDs",
        "Gene Ontology IDs",
        "GO",
    ]

    for col in possible_go_columns:
        if col in df.columns:
            return df.rename(columns={col: "Gene ontology IDs"})

    raise ValueError(
        "Could not find a GO column in the UniProt file.\n"
        f"Available columns: {list(df.columns)}"
    )


def find_protein_existence_column(df):
    pe_cols = [
        c
        for c in df.columns
        if c.lower().replace("_", " ").strip() in ["protein existence", "pe"]
    ]

    return pe_cols[0] if pe_cols else None


# ============================================================
# DUPLICATE / CONFLICT HELPERS
# ============================================================

def sequences_unsure_binary(df, label_col=BINARY_TF_COL):
    """
    Remove duplicate sequences that have conflicting binary TF labels.

    Example:
        Same exact sequence appears once as non-TF and once as TF.
        That sequence is unsafe, so remove it.
    """

    df_mean = df.groupby("Sequence")[label_col].mean().reset_index()

    tol = 1e-5

    unsure_sequences = df_mean[
        np.logical_and(
            df_mean[label_col] < (1 - tol),
            df_mean[label_col] > tol,
        )
    ]["Sequence"].values

    return unsure_sequences


def sequences_unsure_multiclass(df, label_col=THREE_CLASS_LABEL_COL):
    """
    Remove duplicate sequences that have conflicting 3-class labels.
    """

    label_counts = df.groupby("Sequence")[label_col].nunique().reset_index()

    unsure_sequences = label_counts[
        label_counts[label_col] > 1
    ]["Sequence"].values

    return unsure_sequences


# ============================================================
# QUICKGO HELPERS
# ============================================================

def fetch_quickgo_experimental_tf_accessions(
    go_id=GO_CODE,
    evidence_codes=EXPERIMENTAL_ECO_CODES,
    page_size=100,
    sleep_seconds=0.15,
):
    """
    Ask QuickGO for every UniProt protein that has experimental evidence
    for GO:0003700 or one of its descendant TF terms.

    This creates the experimental-TF whitelist.
    """

    experimental_tf_accessions = set()
    page = 1

    headers = {
        "Accept": "application/json"
    }

    while True:
        params = {
            "goId": go_id,
            "goUsage": "descendants",
            "goUsageRelationships": "is_a,part_of",
            "evidenceCode": ",".join(evidence_codes),
            "geneProductType": "protein",
            "limit": page_size,
            "page": page,
        }

        print(f"QuickGO request page {page}...")

        response = requests.get(
            QUICKGO_ANNOTATION_URL,
            params=params,
            headers=headers,
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])

        if not results:
            break

        for annotation in results:
            gene_product_id = annotation.get("geneProductId")
            accession = extract_uniprot_accession(gene_product_id)

            if accession:
                experimental_tf_accessions.add(accession)

        number_of_hits = data.get("numberOfHits")

        if number_of_hits is not None and page * page_size >= number_of_hits:
            break

        if len(results) < page_size:
            break

        page += 1
        time.sleep(sleep_seconds)

    return experimental_tf_accessions


def load_or_fetch_experimental_tf_accessions():
    """
    Use a cached QuickGO accession file if it already exists.
    Otherwise, fetch from QuickGO and save it.
    """

    try:
        cached = pd.read_csv(QUICKGO_CACHE_FILE)

        if "Entry" in cached.columns:
            print(f"Loaded cached QuickGO experimental TF accessions from {QUICKGO_CACHE_FILE}")
            return set(cached["Entry"].dropna().astype(str))

    except FileNotFoundError:
        pass

    print("No QuickGO cache found. Fetching experimental TF accessions from QuickGO...")

    accessions = fetch_quickgo_experimental_tf_accessions()

    cache_df = pd.DataFrame(
        sorted(accessions),
        columns=["Entry"]
    )

    cache_df.to_csv(QUICKGO_CACHE_FILE, index=False)

    print(f"Saved QuickGO experimental TF accession cache to {QUICKGO_CACHE_FILE}")
    print(f"Total experimentally verified TF accessions from QuickGO: {len(accessions)}")

    return accessions


# ============================================================
# LABELING HELPERS
# ============================================================

def add_three_class_tf_labels(df, experimental_tf_accessions):
    """
    Add final 3-class labels.

    Rules:
        1 = non-TF
        2 = TF, but not experimentally verified
        3 = experimentally verified TF
    """

    df = df.copy()

    df["Entry_normalized"] = df["Entry"].apply(normalize_uniprot_accession)

    df[THREE_CLASS_LABEL_COL] = 1
    df[THREE_CLASS_LABEL_NAME_COL] = "non_TF"

    is_tf = df[BINARY_TF_COL] == 1
    is_experimental_tf = df["Entry_normalized"].isin(experimental_tf_accessions)

    df.loc[is_tf, THREE_CLASS_LABEL_COL] = 2
    df.loc[is_tf, THREE_CLASS_LABEL_NAME_COL] = "TF_not_experimentally_verified"

    df.loc[is_tf & is_experimental_tf, THREE_CLASS_LABEL_COL] = 3
    df.loc[is_tf & is_experimental_tf, THREE_CLASS_LABEL_NAME_COL] = "experimentally_verified_TF"

    return df


# ============================================================
# MAIN SCRIPT
# ============================================================

print("Loading raw UniProt data...")
df_all = pd.read_csv(UNIPROT_INPUT_FILE, sep="\t")

print("\nRaw UniProt columns:")
print(list(df_all.columns))

# Normalize GO column name.
df_all = normalize_go_column_name(df_all)

# Keep protein-level entries if the column exists.
pe_col = find_protein_existence_column(df_all)

if pe_col:
    print(f"\nFiltering for protein-level entries using column: '{pe_col}'...")

    df_all = df_all[
        df_all[pe_col]
        .astype(str)
        .str.contains("protein level|1", case=False, na=False)
    ].copy()

    print(f"Protein-level filter success. Remaining entries: {len(df_all)}")
else:
    print("\nWARNING: Protein existence column not found.")
    print("Skipping protein-existence filtering.")

required_columns = [
    "Entry",
    "Sequence",
    "Gene ontology IDs",
    "Organism",
    "Length",
]

missing_columns = [col for col in required_columns if col not in df_all.columns]

if missing_columns:
    raise ValueError(
        f"Missing required columns from UniProt file: {missing_columns}\n"
        f"Available columns: {list(df_all.columns)}"
    )

print("\nParsing species metadata...")
df_all["Taxonomic lineage (SPECIES)"] = df_all["Organism"].apply(extract_clean_species)

print("\nLabeling binary TF/non-TF using GO:0003700 family...")
df_all[BINARY_TF_COL] = df_all["Gene ontology IDs"].apply(go_id_labeler(GO_CODE))

print("\nRemoving sequences with conflicting binary TF/non-TF labels...")
binary_conflict_sequences = sequences_unsure_binary(df_all)
df_all = df_all[
    np.logical_not(df_all["Sequence"].isin(binary_conflict_sequences))
].copy()

print(f"Removed {len(binary_conflict_sequences)} binary-conflict sequences.")

print("\nLoading/fetching experimentally verified TF accession whitelist from QuickGO...")
experimental_tf_accessions = load_or_fetch_experimental_tf_accessions()

print("\nAssigning final 3-class labels...")
df_all = add_three_class_tf_labels(df_all, experimental_tf_accessions)

print("\nRemoving sequences with conflicting 3-class labels...")
multiclass_conflict_sequences = sequences_unsure_multiclass(df_all)
df_all = df_all[
    np.logical_not(df_all["Sequence"].isin(multiclass_conflict_sequences))
].copy()

print(f"Removed {len(multiclass_conflict_sequences)} multiclass-conflict sequences.")

# Keep useful metadata, but do not split by species.
preferred_columns = [
    "Entry",
    "Entry_normalized",
    "Sequence",
    "Length",
    "Organism",
    "Taxonomic lineage (SPECIES)",
    "Gene ontology IDs",
    BINARY_TF_COL,
    THREE_CLASS_LABEL_COL,
    THREE_CLASS_LABEL_NAME_COL,
]

remaining_columns = [
    col for col in df_all.columns
    if col not in preferred_columns
]

df_all = df_all[preferred_columns + remaining_columns].copy()

print("\nFinal label counts:")
print(df_all[THREE_CLASS_LABEL_NAME_COL].value_counts())
print()
print(df_all[THREE_CLASS_LABEL_COL].value_counts().sort_index())

print("\nTop 20 species by entry count:")
print(df_all["Taxonomic lineage (SPECIES)"].value_counts().head(20))

print(f"\nSaving full 3-class labeled dataset to {FULL_OUTPUT_FILE}...")
df_all.to_csv(FULL_OUTPUT_FILE, index=False)

species_counts = (
    df_all
    .groupby(["Taxonomic lineage (SPECIES)", THREE_CLASS_LABEL_COL, THREE_CLASS_LABEL_NAME_COL])
    .size()
    .reset_index(name="count")
    .sort_values(["Taxonomic lineage (SPECIES)", THREE_CLASS_LABEL_COL])
)

species_counts.to_csv(SPECIES_COUNTS_OUTPUT_FILE, index=False)

print(f"Saved species label-count summary to {SPECIES_COUNTS_OUTPUT_FILE}")

print("\nSuccess. make_data.py created one full labeled dataset without species-based splitting.")
