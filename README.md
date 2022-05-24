# Emote2Pitch

Course project for Advanced Machine Learning Course @MAU. In this project we attempt to extend work done by Elena Rivas Ruzafa in her Master thesis ['Pix2Pitch'](https://oa.upm.es/63694/1/TFM_ELENA_RIVAS_RUZAFA.pdf), where we will be implementing some of the recommended future work. 


## Team
This project is a group effort with:

- Nigel Sjölin Grech (@NGrech)
- Peter Magnusson (petermgn)
- Rafi Khaliqi (rafikhaliqi71)


## Environment Setup

### Requirements
- python 3.8+
- [Poetry](https://python-poetry.org/)

### Requirements Installation

First install non pyTorch dependencies with poetry.
To install the virtual environment with required packages run:

```bash
poetry install 
```

Then you can install the appropriate version of pyTorch by first running: 

```bash
poetry shell 
```

Then install torch, torchvision, torchaudio, torchtext and torchdata using the appropriate command for your system:

For Cuda support on Windows and Linux (Sorry MAC no cuda for you):

```bash
poe pytorch_cuda
```

For CPU only support on Windows and MAC:

```bash
poe pytorch_cpu
```

For CPU only support on Linux:

```bash
poe linus_pytorch_cpu
```

# Data Setup

TODO: add how to get the data

