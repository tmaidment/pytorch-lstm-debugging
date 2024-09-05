#!/bin/bash

# Create conda environments for different PyTorch versions
create_env() {
    conda create -n pytorch_$1 python=3.8 -y
    conda activate pytorch_$1
    conda install pytorch=$1 cpuonly -c pytorch -y
    conda deactivate
}

create_env "1.13.0"
create_env "2.0.0"