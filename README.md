# Multi-Passage Reading Comprehension Deep Q-Network

## About this repository
This repository contains the source code of the EMNLP 2020 paper [Interactive Fiction Game Playing as Multi-Paragraph Reading Comprehension with Reinforcement Learning](https://arxiv.org/abs/2010.02386).

## Dependencies
To get started with the framework, install the following dependencies:
- Python 3.7
- PyTorch (>=1.4.0) 
- Jericho (2.4.3 or 2.4.2)
- spaCy (2.2.3)

## Set up interactive fiction games simulator
The code base is built for Jericho 2.4.3 and 2.4.2. To install Jericho 2.4.2, run: 
```
pip install jericho==2.4.2
```
The game roms are in the [roms](roms/) folder.  


## Training the models for interactive fiction games
1. Prepare and pre-process the glove word embeddings.

1.1 Download the 100d [GLoVe word embeddings](https://nlp.stanford.edu/projects/glove/) and copy the file to the [agents/glove](agents/glove) folder. 

1.2 Pre-process the glove word embeddings; run:
```
python glove_utils.py
``` 
or download the processed embeddings from [here](https://ibm.box.com/s/3k2akbk4svnr1fczgjnllk9j1iyjauya).  

2. Train the models for interactive fiction games. For example, to train on the game Zork1, run:
```
python train.py --batch_size 64 --env_id "zork1.z5"
```


## License
MIT License
