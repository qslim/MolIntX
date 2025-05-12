# MolIntX: A Taylor Sensitivity Framework for Quantifying and Enhancing Atomic Interactions in Molecular Graph Representation Learning

[![MIT License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

This is the code of the paper "MolIntX: A Taylor Sensitivity Framework for Quantifying and Enhancing Atomic Interactions in Molecular Graph Representation Learning"

## Requirements

The following packages need to be installed:

- `pytorch==2.0.1`
- `dgl==1.1.2`
- `ogb==1.3.6`
- `numpy`
- `easydict`
- `tensorboard`
- `tqdm`
- `json5`

## Usage

#### ZINC
- Change your current directory to [zinc](zinc);
- Configure hyper-parameters in [ZINC.json](zinc/ZINC.json);
- Run the script: `sh run_script.sh`.

#### ogbg-molpcba
- Change your current directory to [ogbg/mol](ogbg/mol);
- Configure hyper-parameters in [ogbg-molpcba.json](ogbg/mol/ogbg-molpcba.json);
- Run the script: `sh run_script.sh`.

#### TUDataset
- Change your current directory to [tu](tu);
- configure hyper-parameters in [configs/\<dataset\>.json](tu/configs);
- Input the dataset name in [run_script.sh](tu/run_script.sh);
- Run the script: `sh run_script.sh`.


## License

[MIT License](LICENSE)
