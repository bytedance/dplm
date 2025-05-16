wget -O dplm2_metadata.tar.gz https://zenodo.org/records/15424801/files/dplm2_metadata.tar.gz?download=1
mkdir -p data-bin
tar -xzvf dplm2_metadata.tar.gz -C ./data-bin/
