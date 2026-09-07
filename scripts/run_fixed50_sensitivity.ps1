# Phase 9 fixed-50 binning sensitivity analysis.
# Run from the project root while the .venv is active.

# 1) Create fixed-50 bin edges from training-based pilot runs.
python -m src.pilot_bin_edges --model mlp --dataset FashionMNIST --method fixed --fixed-bins 50 --epochs 5 --max-batches 79 --output-dir ./results/bin_edges_fixed50
python -m src.pilot_bin_edges --model cnn --dataset FashionMNIST --method fixed --fixed-bins 50 --epochs 5 --max-batches 79 --output-dir ./results/bin_edges_fixed50

# 2) Run fixed-50 sensitivity experiments.
# No --final-test is used here, because the sensitivity analysis compares entropy trajectories only.
python -m src.experiments_entropy --model mlp --dataset FashionMNIST --epochs 30 --seed 1 --entropy-max-batches 79 --bin-edges-dir ./results/bin_edges_fixed50 --binning-method fixed --output-dir ./results/sensitivity_fixed50/logs
python -m src.experiments_entropy --model mlp --dataset FashionMNIST --epochs 30 --seed 2 --entropy-max-batches 79 --bin-edges-dir ./results/bin_edges_fixed50 --binning-method fixed --output-dir ./results/sensitivity_fixed50/logs
python -m src.experiments_entropy --model mlp --dataset FashionMNIST --epochs 30 --seed 3 --entropy-max-batches 79 --bin-edges-dir ./results/bin_edges_fixed50 --binning-method fixed --output-dir ./results/sensitivity_fixed50/logs
python -m src.experiments_entropy --model mlp --dataset FashionMNIST --epochs 30 --seed 4 --entropy-max-batches 79 --bin-edges-dir ./results/bin_edges_fixed50 --binning-method fixed --output-dir ./results/sensitivity_fixed50/logs
python -m src.experiments_entropy --model mlp --dataset FashionMNIST --epochs 30 --seed 5 --entropy-max-batches 79 --bin-edges-dir ./results/bin_edges_fixed50 --binning-method fixed --output-dir ./results/sensitivity_fixed50/logs

python -m src.experiments_entropy --model cnn --dataset FashionMNIST --epochs 30 --seed 1 --entropy-max-batches 79 --bin-edges-dir ./results/bin_edges_fixed50 --binning-method fixed --output-dir ./results/sensitivity_fixed50/logs
python -m src.experiments_entropy --model cnn --dataset FashionMNIST --epochs 30 --seed 2 --entropy-max-batches 79 --bin-edges-dir ./results/bin_edges_fixed50 --binning-method fixed --output-dir ./results/sensitivity_fixed50/logs
python -m src.experiments_entropy --model cnn --dataset FashionMNIST --epochs 30 --seed 3 --entropy-max-batches 79 --bin-edges-dir ./results/bin_edges_fixed50 --binning-method fixed --output-dir ./results/sensitivity_fixed50/logs
python -m src.experiments_entropy --model cnn --dataset FashionMNIST --epochs 30 --seed 4 --entropy-max-batches 79 --bin-edges-dir ./results/bin_edges_fixed50 --binning-method fixed --output-dir ./results/sensitivity_fixed50/logs
python -m src.experiments_entropy --model cnn --dataset FashionMNIST --epochs 30 --seed 5 --entropy-max-batches 79 --bin-edges-dir ./results/bin_edges_fixed50 --binning-method fixed --output-dir ./results/sensitivity_fixed50/logs

# 3) Aggregate sensitivity results and compare to the main Freedman-Diaconis results.
python -m src.aggregate_sensitivity_fixed50
