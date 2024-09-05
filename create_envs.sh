#!/bin/bash

# Initialize conda for this script
eval "$(conda shell.bash hook)"

# Create conda environments for different PyTorch versions
create_env() {
    conda create -n pytorch_$1 python=3.10.12 -y
    conda install -n pytorch_$1 conda-forge::numpy -y
    conda install -n pytorch_$1 pytorch=$1 cpuonly -c pytorch -y
    conda install -n pytorch_$1 conda-forge::psutil -y
    conda install -n pytorch_$1 conda-forge::tqdm -y
}

create_env "1.13.0"
create_env "2.0.1"
create_env "2.4.1"