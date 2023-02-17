#!/bin/bash

ver=$(python -c "import torch; print(torch.cuda.is_available())")

if [[ "$ver" == "True" ]]
then 
	echo "Installing with CUDA."
fi

pip install .
