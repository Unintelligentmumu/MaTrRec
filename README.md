
### Requirements
  * `pip install recbole`
  * `pip install causal-conv1d>=1.4.0`
  * `pip install mamba-ssm`
Our experimental platform is equipped with an NVIDIA V100 GPU, Ubuntu 22.04, PyTorch 2.1.2, and CUDA 12.1.

You can also refer to the required environment specifications in requirements.txt
### Run MaTrRec

```python run_recbole.py --model=MaTrRec --dataset=ml-1m  --config_files=ml.yaml```  

```python run_recbole.py --model=MaTrRec --dataset=amazon-musical-instruments  --config_files=amazon.yaml```  

```python run_recbole.py --model=MaTrRec --dataset=amazon-health-personal-care  --config_files=amazon.yaml```

```python run_recbole.py --model=MaTrRec --dataset=amazon-electronics --config_files=amazon.yaml```

```python run_recbole.py --model=MaTrRec --dataset=amazon-office-products --config_files=amazon.yaml```

### Run MaTrRec*

```python run_recbole.py --model=MaTrRec1 --dataset=ml-1m  --config_files=ml.yaml```

```python run_recbole.py --model=MaTrRec1 --dataset=amazon-musical-instruments  --config_files=amazon.yaml```

```python run_recbole.py --model=MaTrRec1 --dataset=amazon-health-personal-care  --config_files=amazon.yaml```

```python run_recbole.py --model=MaTrRec1 --dataset=amazon-electronics --config_files=amazon.yaml```

```python run_recbole.py --model=MaTrRec1 --dataset=amazon-office-products --config_files=amazon.yaml```

### Comparative experiment

```python run_recbole.py --model=SASRec --dataset=amazon-office-products  --config_files=ml.yaml```

```python run_recbole.py --model=SASRec --dataset=amazon-musical-instruments  --config_files=amazon.yaml```

```python run_recbole.py --model=Mamba4Rec --dataset=amazon-office-products  --config_files=ml.yaml```

```python run_recbole.py --model=Mamba4Rec --dataset=amazon-musical-instruments  --config_files=amazon.yaml```



## Acknowledgment

This project is based on [Mamba](https://github.com/state-spaces/mamba), [Causal-Conv1d](https://github.com/Dao-AILab/causal-conv1d), and [RecBole](https://github.com/RUCAIBox/RecBole). Thanks for their excellent works.
