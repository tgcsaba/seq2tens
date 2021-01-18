# Seq2Tens: An efficient representation of sequences by low-rank tensor projections
This repository contains supplementary code to the paper https://arxiv.org/abs/1906.08215.
***
## Disclaimer
The present code is what was used at the time of submission of the paper, therefore at the moment is not intended to be very user-friendly. A cleaned up and streamlined version of the code with many extra features is in the making and soon will be released, so stay tuned!
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
***
## Download datasets
The tsc directory contains the appropriate scripts used to run the tsc experiments in the paper. The datasets can be downloaded from our dropbox folder using the `download_data.sh` script in the `./tsc/datasets` folder by running
```
cd tsc
bash ./datasets/download_data.sh
```
or manually by copy-pasting the dropbox url containd within the aforementioned script.

## Support
We encourage the use of this code for applications, and we aim to provide support in as many cases as possible. For further assistance or to tell us about your project, please send an email to

`csaba.toth@maths.ox.ac.uk` / `patric.bonnier@maths.ox.ac.uk` / `harald.oberhauser@maths.ox.ac.uk`.
