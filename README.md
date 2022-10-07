# BDI
We propose **B**i**D**irectional learning for offline **I**nfinite-width model-based optimization (BDI) between the high-scoring designs and the static dataset (a.k.a. low-scoring designs).

## Installation

The environment of BDI can be installed as:
```bash
conda create --name BDI --file requirements.txt
conda activate BDI
```

## Reproducing Performance

For the TF Bind 8 task, we can run the forward mapping as:
```bash
python -u BDI.py --mode grad --task TFBind8-Exact-v0 --outer_lr 1e-1 --gamma 0.0
```
The backward mapping can be run as:
```bash
python -u BDI.py --mode distill --task TFBind8-Exact-v0 --outer_lr 1e-1 --gamma 0.0
```
Run our BDI as:
```bash
python -u BDI.py --mode both --task TFBind8-Exact-v0 --outer_lr 1e-1 --gamma 0.0
```

Similarly for AntMorphology task, we have:
```bash
python -u BDI.py --mode grad --task AntMorphology-Exact-v0 --outer_lr 1e-3 --gamma 0.001
python -u BDI.py --mode distill --task AntMorphology-Exact-v0 --outer_lr 1e-3 --gamma 0.001
python -u BDI.py --mode both --task AntMorphology-Exact-v0 --outer_lr 1e-3 --gamma 0.001
```
The training log of the above experiments can be found in run.log.

## Acknowledgements
We thank the design-bench library (https://github.com/brandontrabucco/design-bench) and the data distillation implementation (https://colab.research.google.com/github/google-research/google-research/blob/master/kip/KIP.ipynb).