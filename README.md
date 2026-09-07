# Investigation of Hidden Representation Dynamics Through Activation Entropy

This repository contains the experimental code, saved configurations, results, and analysis outputs used for the bachelor thesis:

**Investigation of Hidden Representation Dynamics Through Activation Entropy**

The repository is intended as a reproducibility supplement to the thesis. Detailed methodological and implementation explanations are provided in Chapters 3 and 4 of the thesis rather than repeated here.

## Official Experiment Scope

The reported main experiment consists of:

- Fashion-MNIST
- one MLP and one CNN
- 30 training epochs per run
- official training seeds 1–5
- fixed train–validation split seed 42
- validation and activation-entropy evaluation after every epoch
- one final held-out test evaluation after each completed official run
- a separate fixed 50-bin sensitivity analysis

Pilot seed 0 is used only to construct histogram configurations and is not part of the reported main results.

Any retained seed-42 development or practice outputs are not part of the official thesis analysis. Thesis-oriented aggregation scripts filter to complete 30-epoch runs using official seeds 1–5.

## Environment

The completed experiment used Python 3.10 on Windows with CPU execution. Exact installed package versions are recorded in:

requirements_locked.txt

A compatible environment can be created with:

powershell

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements_locked.txt

Fashion-MNIST is downloaded through Torchvision and does not need to be stored in the repository.

## Running an Official Main Run

Example MLP run:

powershell

python -m src.experiments_entropy --model mlp --dataset FashionMNIST --epochs 30 --seed 1 --entropy-max-batches 79 --final-test

Example CNN run:

powershell

python -m src.experiments_entropy --model cnn --dataset FashionMNIST --epochs 30 --seed 1 --entropy-max-batches 79 --final-test


Change --seed to 2, 3, 4, or 5 for the remaining official runs.

--entropy-max-batches 79 evaluates activation entropy on the complete 10,000-image validation set.

--final-test performs one final evaluation on the official Fashion-MNIST test set after the 30 training epochs.

## Aggregating Main Results

After the individual runs have been completed:

powershell

python -m src.aggregate_results
python -m src.create_phase8_outputs


The thesis-oriented output pipeline restricts summaries to official seeds 1–5 and complete 30-epoch runs.

## Running the Fixed 50-Bin Sensitivity Analysis

The sensitivity workflow is automated through:

powershell

.\scripts\run_fixed50_sensitivity.ps1

If required by the local PowerShell execution policy:

powershell

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run_fixed50_sensitivity.ps1


The sensitivity runs are stored separately from the main Freedman–Diaconis experiment and do not perform final test evaluation.

## Main Output Locations

For the final thesis results, the most relevant folders are:

results/phase8_tables/
results/plots/
results/tables/
results/sensitivity_fixed50/aggregated/

Individual epoch-level main-run histories are stored in:

results/logs/

Saved histogram configurations are stored in:

results/bin_edges/
results/bin_edges_fixed50/

## Reproducibility Notes

The repository includes the saved train/validation split indices and uses fixed random seeds, dedicated PyTorch DataLoader generators, matched training-data ordering for corresponding MLP and CNN seeds, "num_workers=0", and documented package versions.

PyTorch deterministic algorithms were not globally enforced, so exact bit-for-bit reproduction across different hardware, operating systems, processor implementations, or library versions is not guaranteed.

## Legacy Output Names

Some internal output filenames retain earlier implementation labels:

- `signature_entropy` refers to selected representative activation-entropy trajectories; it is not a separate entropy metric.
- `effect_sizes` refers to raw initial-to-final entropy changes with between-run variability; the thesis does not interpret these values as standardized statistical effect sizes.

These labels are retained only for compatibility with the existing analysis scripts and generated files.

## Thesis Interpretation

The repository provides the computational evidence supporting the thesis. Activation entropy is interpreted as a descriptive measure of hidden activation-distribution change and is considered alongside predictive-performance and generalization metrics.

For the full experimental rationale, entropy-estimation procedure, implementation details, limitations, and interpretation of results, refer to the thesis itself.
