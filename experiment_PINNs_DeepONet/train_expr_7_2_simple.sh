#!/bin/bash
set -e

python experiment/experiment_7_2.py --mode simple --p 2    --domain square --save-npy
python experiment/experiment_7_2.py --mode simple --p 3    --domain square --save-npy
python experiment/experiment_7_2.py --mode simple --p 4    --domain square --save-npy
python experiment/experiment_7_2.py --mode simple --p 5    --domain square --save-npy
python experiment/experiment_7_2.py --mode simple --p 6    --domain square --save-npy
python experiment/experiment_7_2.py --mode simple --p 7    --domain square --save-npy
python experiment/experiment_7_2.py --mode simple --p 8    --domain square --save-npy
python experiment/experiment_7_2.py --mode simple --p 9    --domain square --save-npy
python experiment/experiment_7_2.py --mode simple --p 10   --domain square --save-npy
python experiment/experiment_7_2.py --mode simple --p 20   --domain square --save-npy
python experiment/experiment_7_2.py --mode simple --p 50   --domain square --save-npy
python experiment/experiment_7_2.py --mode simple --p 100  --domain square --save-npy
python experiment/experiment_7_2.py --mode simple --p 1000 --domain square --save-npy
