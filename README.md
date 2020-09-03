Submission to the Cornell Birdcall Identification Kaggle Challenge
==================================================================

Implementation of the submission of team "[CP-JKU] Three Geese and a GAN" of
the [Institute of Computational
Perception](https://https://www.jku.at/en/institute-of-computational-perception/)
at Johannes-Kepler-University Linz, Austria, to the [Cornell Birdcall
Identification Challenge](https://www.kaggle.com/c/birdsong-recognition/) run
at Kaggle from July to September 2020.

A Pytorch-based template for audio classification and filterbank learning
experiments.

Prerequisites
-------------

You will need Python 3 with at least the following modules:
* numpy
* scikit-learn
* torch>=1.5.1
* tqdm
* pandas

For some features, additional modules are required:
* tensorboard
* apex
* matplotlib

With pip, prerequisites can be installed with:
```bash
pip3 install numpy scikit-learn tqdm pandas matplotlib
```

Similar with conda:
```bash
conda install numpy scikit-learn tqdm pandas matplotlib
```

Pytorch should be installed following the instructions on
[pytorch.org](https://pytorch.org). You will need Pytorch 1.5.1 or later.

Tensorboard can be installed with `pip3 install tensorboard` or
`conda install -c conda-forge tensorboard`.

Apex currently needs to be installed via pip (even when using conda):
```
pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" https://github.com/NVIDIA/apex/archive/master.zip
```

Apart from Python, you will need `ffmpeg`. On Debian and Ubuntu, this can be
installed with `sudo apt install ffmpeg libavcodec-extra`.
