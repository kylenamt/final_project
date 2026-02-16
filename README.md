# final_project

How to run:

this project uses anaconda
```{bash}
conda create -n conda_env3.10 python=3.10
conda activate conda_env3.10

conda install -c nvidia cuda-toolkit=11.2 cudnn=8.1

```

Then, install dependencies:
```{bash}
pip install -e .
```