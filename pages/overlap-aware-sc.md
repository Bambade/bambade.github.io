---
layout: page
title: Multi-class Spectral Clustering with Overlaps for Speaker Diarization 
---

This post describes the implementation of our paper _"Multi-class spectral clustering with 
overlaps for speaker diarization"_, accepted for publication at IEEE SLT 2021.

The code consists of 2 parts: overlap detector, and our modified spectral clustering method
for overlap-aware diarization.

We have implemented the diarization recipe in Kaldi, and modified scikit-learn's 
spectral clustering class for our modification. The entire code to reproduce our results
is available at: `https://github.com/desh2608/kaldi/tree/slt21_spectral`

### Installation and setup

Since the recipe is implemented using Kaldi, you first need to install Kaldi by 
following the instructions at: `https://github.com/kaldi-asr/kaldi/blob/master/INSTALL`

In the Kaldi installation, miniconda is not installed by default. To install it, go to
`tools/` and run:

```
extras/install_miniconda.sh
```

Install a [modified version](https://github.com/desh2608/scikit-learn/tree/overlap) of scikit-learn in the miniconda Python installation:

```
$HOME/miniconda3/bin/python -m pip install git+https://github.com/desh2608/scikit-learn.git@overlap
```


### Usage

The recipe containing the overlap detector and the spectral clustering for AMI
can be found at `egs/ami/s5c`. Additionally, the recipe also contains example for
different clustering methods, namely AHC, VBx, and spectral clustering, to reproduce
the single-speaker baselines in the paper.

The `run.sh` script does not contain stages for training an x-vector extractor, since we 
used the same extractor from the CHiME-6 baseline system.

The key stages in the script are as follows.

* `--stage 8`: Trains the overlap detector.
* `--stage 9`: Performs decoding with a trained overlap detector.
* `--stage 10`: Performs spectral clustering informed by the output from stage 9.


### Where to find the clustering implementation?

We use scikit-learn's [spectral clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html) class to implement our modified clustering
method. In the default scikit-learn class, the argument `assign_labels` can take
on 2 values:

1. `kmeans`: This performs the conventional spectral clustering using the Ng-Jordan-Weiss method.
2. `discretize`: This implements the clustering described in [this paper](https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf) and we modify it for our
implementation.

We modified the `discretize()` function [here](https://github.com/desh2608/scikit-learn/blob/fe74ed9573160d87aa5e40d8bd9d9af20283b3bc/sklearn/cluster/_spectral.py#L23) by adding an
additional argument which specifies the overlap vector. The vector is used in L141-148
to assign a second label to the overlapping segments.


### Pre-trained model

For all our clustering-based diarization experiments, we used the x-vector extractor
that was provided with the CHiME-6 baseline system, and is available [here](http://kaldi-asr.org/models/m12). To use it, first download the extractor using `wget` and then extract it using 
`tar -xvzf` and copy the contents to your `exp` directory.


### Citation

If you find this code useful, consider citing our paper:

```
@inproceedings{Raj2021MultiSC,
  title={Multi-class spectral clustering with overlaps for speaker diarization},
  author={Desh Raj and Zili Huang and Sanjeev Khudanpur},
  booktitle={IEEE Spoken Language Technology Workshop},
  year={2021},
}
```

### Questions

For any questions about using the code, you can contact me at `draj@cs.jhu.edu`.

