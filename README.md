# Resume Parser

This repository contains the source code for the paper:
"Extracting Section Structure from Resumes in Brazilian Portuguese"

# Environment tested

- Ubuntu 22.04
- Intel(R) Core(TM) i5-12400 CPU @ 4.40GHz,
- 32 GB of RAM
- GeForce RTX 4070 Ti

# Dependencies

Before installing the dependencies, please install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

```shell
conda env create -n resume_parser -f environment.yml
```

# Building
```shell
conda activate resume_parser
pip install -e .
```

# Reproducing the experiments

Due to user privacy concerns, it is only possible to reproduce the results of the Page Template Identification task.

## Page Template Identification

### Datasets

Our anonymized version of our dataset can be downloaded [here](https://www.dropbox.com/s/px2iud43z3an6lk/page_template_dataset.zip?dl=0).

Unzip the compressed file under the `resources/layout` folder.

It is also essential to update the `root_dirpath` field inside the `split_[0-4].conf` files.

### Running

All output results will be saved under the `output` folder.
In the scripts below, update the `dataset_split` to the root directory of the `split_[0-4].conf` files before running.

#### Feature extractor

For training the model + Validation results:
```shell
cd ./resume_parser/layout
cp ./scripts/run_classifier.sh ./
sh run_classifier.sh
```

For testing:
```shell
python test.py --experiment_dir ./output/classifier/<generated-folder>
```

#### Fine-tuning

The procedure is analogous to the `Feature extractor`.

For training the model + Validation results:
```shell
cd ./resume_parser/layout
cp ./scripts/run_model.sh ./
sh run_model.sh
```

For testing:
```shell
python train.py --experiment_dir ./output/classifier/<generated-folder>
```
