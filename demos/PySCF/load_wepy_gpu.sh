name=your_name
export PYTHONPATH="/mnt/research/PTR_bose/$name/gpu4pyscf${PYTHONPATH:+:$PYTHONPATH}"
ml purge && ml load Miniforge3 OpenBLAS CUDA && conda activate wepy-dev
cd /mnt/research/PTR_bose/$name/wepy_dev
