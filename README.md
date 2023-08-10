# Seq2Tens: An efficient representation of sequences by low-rank tensor projections
This repository contains supplementary code to the paper https://arxiv.org/abs/2006.07027.
***
## Update (2023)
We released a new and updated version of the code, intended to be much more modular and streamlined, and coming with an easy-to-configure experimental pipeline.
## Installing
To get started, you should first clone the repository using git, e.g. with the command
```
git clone https://github.com/tgcsaba/seq2tens.git
```
and then create and activate virtual environment with Python <= 3.7
```
conda create -n env_name python=3.7
conda activate env_name
```
Then, install the requirements using pip by
```
pip install -r requirements.txt
```
Note that if GPUs are required, the `tensorflow-gpu` version should be [compatible with the CUDA installation](https://www.tensorflow.org/install/source#gpu), or [compiled from source](https://www.tensorflow.org/install/source).
***
## Time series classification experiments
The tsc directory contains the appropriate scripts used to run the time series classification experiments in the paper.
Steps to reproduce the results:
- Remove/rename the `benchmarks` and the `tmp` directory, since the run script skips all existing results
- Optional: Open the file `configs/configs.yaml` file, and change the model configurations as desired
- Call `python run_experiments.py [GPU_ID]` for running on GPU, or leave GPU_ID empty for CPU
- Visualize the results and generate the plots using the `results.ipynb` notebook

## Support
We encourage the use of this code for applications, and we aim to provide support in as many cases as possible. For further assistance or to tell us about your project, please send an email to

`csaba.toth@maths.ox.ac.uk` / `patric.bonnier@maths.ox.ac.uk` / `harald.oberhauser@maths.ox.ac.uk`.
