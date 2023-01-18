# FGTL
Source code and appendix for paper "FGTL: Federated Graph Transfer Learning for Node Classification".

## Usage

1. Use `data/preprocess.ipynb` to prepare the datasets for all experiments.
2. Run one of these methods using `scripts/*.sh`, such as the following command:

```bash
chmod +x scripts/fgtl.sh
nohup scripts/fgtl.sh > scripts/fgtl.out 2>&1 &
```