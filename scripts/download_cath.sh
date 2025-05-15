mkdir -p data-bin
wget -r -nd -np http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/ -P data-bin/cath_4.2


mkdir -p data-bin/cath_4.3
wget -r -nd -np https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/chain_set.jsonl -P data-bin/cath_4.3
wget -r -nd -np https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/split.jsonl -P data-bin/cath_4.3
