#!/bin/bash
curl -L -o uniprot_data.tab.gz "https://rest.uniprot.org/uniprotkb/stream?query=reviewed%3Atrue%20AND%20length%3A%5B50%20TO%205500%5D&format=tsv&fields=accession,sequence,go,organism_name,length,protein_existence&compressed=true"

gunzip uniprot_data.tab.gz

mv uniprot_data.tab data/