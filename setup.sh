#!/bin/bash 

echo "y" | ami_atlas_post_install
pip install torch==${TORCH} --extra-index-url https://download.pytorch.org/whl/${VERSION}
pip install torch_scatter -f https://data.pyg.org/whl/torch-${TORCH}+${VERSION}.html
pip install torch_sparse -f https://data.pyg.org/whl/torch-${TORCH}+${VERSION}.html
pip install torch_cluster -f https://data.pyg.org/whl/torch-${TORCH}+${VERSION}.html

pip install .
cd torch-extensions 
pip install . 
