# activate the conda environment
conda activate MY_ENV

# go to the right folder
cd PATH_TO_PROJECT

###############
# EXPERIMENTS #
###############
python main.py --epochs 100 --outpath experiment01

# change batch_size
python main.py --epochs 100 --batch_size 16 --outpath experiment02
