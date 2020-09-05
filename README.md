# Comparision of POS Tagging using different GRU cell mutations

Code submitted as mini project for partial fullfilment of M.Tech degree.

## Setup

Ensure that you have python2/3.


This project is organized into 3 folders ```literature```, ```logs```, ```report_ppt``` and 5 files ```gru.py```, ```main.py```, ```pos_tagger.py```, ```train.py```, ```TreeBankDataSet.py```

	PosTagging
	├── gru.py
	├── literature
	│   └── Empirical Analysis of RNN.pdf
	├── logs
	│   └── logs_400_400_0.1_10_40_s_Adam_treebank_nll_0.8_file.log
	├── main.py
	├── pos_tagger.py
	├── README.md
	├── report_ppt
	│   ├── POS Tagging using GRU.pdf
	│   ├── ppt.pdf
	│   └── report_sc18m002_pos_tagging.lyx~
	├── train.py
	└── TreeBankDataSet.py

```literature``` folder contains the base research paper from which the idea is inspired.
```logs```  folder contains generated logs for experiment.
```report_ppt```  folder contains final report and presentation.



### Hardware requirements

This project requires an NVIDIA-GPU with CUDA 9.0 to run PyTorch with GPU computing capabilities. 

## Dataset

We have used  ```TreeBankDataSet``` for the expermients.


## Code
```main.py``` is starting script for running the experiment
```train.py``` is training script
```pos_tagger.py``` architecture of generic pos_tagger
```gru.py```, contains architecture of GRU cell



### Training the model (```train_code```)
	
Start the training process by calling ```main.py```
