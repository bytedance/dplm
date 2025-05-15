pip install huggingface_hub
# download DPLM-2 training set (PDB and SwissProt) from huggingface hub
huggingface-cli download airkingbd/pdb_swissprot --repo-type dataset --local-dir ./data-bin/pdb_swissprot
