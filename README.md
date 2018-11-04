# WaveGlow 

A PyTorch implementation of the [WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/abs/1811.00002)

## Quick Start:

1. Install requirements:
```
pip install -r requirements.txt
```

2. Download dataset:
```
wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_slt_arctic-0.95-release.tar.bz2
tar xf cmu_us_slt_arctic-0.95-release.tar.bz2
```

3. Extract features: 
feature extracting pipeline is the same as [tacotron](https://github.com/keithito/tacotron)

4. Training with default hyperparams:
```
python train.py
```

5. Synthesize from model:
```
python generate.py --checkpoint=/path/to/model --local_condition_file=/path/to/local_conditon
```

## Notes:
  * This is not offical implementation, some details are not necessarily correct.
  * Work in progress.
