# Tree Ensembles for Contextual Bandits - Accompanying Code

Code for reproducing experiments in paper 'Tree Ensembles for Contextual Bandits'.

## How to tun the code:
Populate datasets folder with the respective data files, downloaded from UCI ML Repo (https://archive.ics.uci.edu/)

Install the required python packages from the requirements.txt file (preferably in a new virtual environment)
> pip install -r requirements.txt

Run bash scripts corresponding intended exeperiments, e.g. adult dataset with categorical features:
> ./run_cr_adult_categorical.sh

(you may have to specify subfolders within the results folder corresponding to the res_path specified in the bash script)

To run on ssh connected remote server use e.g. nohup:
> nohup ./run_cr_adult_categorical.sh > adult_categorical.out &
