---
layout: page
title: "Integration of speech separation, diarization, and recognition for multi-speaker 
meetings: System description, comparison, and analysis"
---

This post describes how to reproduce and extend our pipeline of speech separation,
diarization and ASR. We have provided separated audio files along with entire
end-to-end reproducible recipes to supplement [our SLT 2021 paper](https://arxiv.org/abs/2011.02014).

Here is a summary of contents of this post:

1. [Summary of our pipeline](#pipeline)
2. [Datasets](#data)
3. [Reproducible recipes](#recipe)
4. [Example use cases (for extending this work)](#example)
5. [Credits](#credits)

<a name="pipeline"></a>

### Summary of our pipeline

Our speech processing pipeline consists of the following 3 stages in sequence. For
each of these stages, we experimented with the methods mentioned below.

1. **Speech separation**
    
    1. Mask-based MVDR
    2. Sequential neural beamforming

2. **Speaker diarization**
    
    1. Clustering: Agglomerative hierarchical clustering, spectral clustering,
    Variational Bayes based x-vector clustering (VBx)
    2. Region proposal networks
    3. Target speaker voice activity detection

3. **Speech recognition (ASR)**
    
    1. Hybrid TDNNF-based
    2. End-to-end transformer based

<a name="data"></a>

### Datasets

In our paper, we compare the performance of our models on mixed and separated audio,
where the separation is performed using the methods mentioned earlier. 

#### Downloading and preparing the mixed (original) LibriCSS data

```bash
$ git clone https://github.com/chenzhuo1011/libri_css.git
$ conda env create -f conda_env.yml
$ conda activate libricss_release
$ cd libri_css && ./dataprep/scripts/dataprep.sh
```

#### Downloading the separated audio data

Additionally, we also provided 2-stream and 3-stream separated audio wav files
through [Zenodo](https://doi.org/10.5281/zenodo.4415163). You can download them
using the following commands:

```bash
$ wget https://zenodo.org/record/4415163/files/libricss_mvdr_2stream.tar.gz
$ wget https://zenodo.org/record/4415163/files/libricss_sequential_3stream.tar.gz
```

Once the archived files are downloaded, you can extract them and then use the path
in the Kaldi or ESPNet recipes.

<a name="recipe"></a>

### Reproducible recipes

#### Kaldi recipe

We have provided Kaldi recipes `s5_mono` and `s5_css` [here](https://github.com/kaldi-asr/kaldi/tree/master/egs/libri_css).

1. `s5_mono`: This is a single channel diarization + ASR recipe which takes as the
input a long single-channel recording containing mixed audio. It then performs SAD,
diarization, and ASR on it and outputs speaker-attributed transcriptions, 
which are then evaluated with cpWER (similar to CHiME6 Track 2).

2. `s5_css`: This pipeline uses a speech separation module at the beginning,
so the input is 2-3 separated audio streams. We assume that the separation is
window-based, so that the same speaker may be split across different streams in
different windows, thus making diarization necessary.

`s5_mono` evaluates diarization and ASR on mixed audio, while `s5_css` does the
same for separated audio streams. 

**Note:** Only clustering-based diarization is available in this recipe at the 
time of making this post, but we are also preparing RPN and TS-VAD setups.

For ease of reproduction, we have included training stages in the `s5_mono` recipe. 
We also provide pretrained models for both diarization and ASR systems:

* **SAD**: CHiME-6 baseline TDNN-Stats SAD available [here](http://kaldi-asr.org/models/m12).

* **Speaker diarization**: CHiME-6 baseline x-vector + AHC diarizer, trained on VoxCeleb 
with simulated RIRs available [here](http://kaldi-asr.org/models/m12).

* **ASR**: We used the chain model trained on 960h clean LibriSpeech training data available
[here](http://kaldi-asr.org/models/m13). It was then additionally fine-tuned for 1
epoch on LibriSpeech + simulated RIRs. For LM, we trained a TDNN-LSTM language model
for rescoring. All of these models are available at this 
[Google Drive link](https://drive.google.com/file/d/13ceXdK6oAUuUyxn7kjQVVqpe8r6Sc7ds/view?usp=sharing).

#### ESPNet recipe

The ESPNet recipe corresponding to the Kaldi `s5_mono` recipe is available 
[here](https://github.com/espnet/espnet/tree/master/egs/libri_css) as `asr1`.
A recipe corresponding to `s5_css` is not available yet, but it should be simple
to extend `asr1` similar to the `s5_css` recipe, since it follows Kaldi-style
diarization. For help or other details, please contact 
[Pavel Denisov](https://www.ims.uni-stuttgart.de/en/institute/team/Denisov/) 
who created the ESPNet LibriCSS recipe.
 
<a name="example"></a>

### Example use cases (for extending this work)

Let us now look at how this research may be extended through 2 examples.

#### Example 1: A new window-based separation method

Suppose you have a new _window-based_ "continuous" speech separation model, and 
you want to evaluate the downstream ASR performance (and compare it with the
methods in our paper). This can be done as follows:

1. Download and prepare the "mixed" LibriCSS audio data as described [here](#data).

2. Run your separation method on this data and store the generated audio streams
using the following naming convention: 
`overlap_ratio_10.0_sil0.1_1.0_session7_actual10.1_channel_1.wav`. It is similar 
to the naming of the original LibriCSS files, with the addition of *_channel_1* 
at the end which denotes the stream. 

    > **Note**: channel here does not refer to the microphone; it refers to the 
    separated stream; so if your model separated the audio into 2 streams, they 
    would have the suffixes *_channel_0* and *_channel_1*.

3. Store all the output wav files in a directory. They can have any hierarchy within
this directory as long as they follow the naming convention.

4. Download and install Kaldi, and navigate to the `egs/libri_css/s5_css` folder.

5. In the `run.sh`, replace the paths to the mixed LibriCSS data and your own
separated audio files.

6. Run the script. It is recommended to run the stages one by one, since the
evaluation outputs after the SAD and diarization stage are also printed to
the standard output.

7. At the end of decoding, the cpWERs will be printed as follows:

```
Dev WERs:
best_wer_session0_CH0_0L %WER 10.98 [ 130 / 1184, 34 ins, 12 del, 84 sub ]
best_wer_session0_CH0_0S %WER 15.10 [ 269 / 1782, 67 ins, 23 del, 179 sub ]
best_wer_session0_CH0_OV10 %WER 25.12 [ 465 / 1851, 156 ins, 85 del, 224 sub ]
best_wer_session0_CH0_OV20 %WER 18.86 [ 342 / 1813, 94 ins, 33 del, 215 sub ]
best_wer_session0_CH0_OV30 %WER 20.42 [ 395 / 1934, 117 ins, 40 del, 238 sub ]
best_wer_session0_CH0_OV40 %WER 28.47 [ 636 / 2234, 236 ins, 137 del, 263 sub ]
Eval WERs:
0L %WER 22.36 [ 2446 / 10938, 785 ins, 413 del, 1248 sub ]
0S %WER 19.81 [ 2970 / 14994, 861 ins, 431 del, 1678 sub ]
OV10 %WER 21.39 [ 3412 / 15951, 1060 ins, 580 del, 1772 sub ]
OV20 %WER 23.49 [ 3984 / 16963, 1128 ins, 747 del, 2109 sub ]
OV30 %WER 26.06 [ 4789 / 18376, 1415 ins, 988 del, 2386 sub ]
OV40 %WER 25.45 [ 4818 / 18932, 1410 ins, 676 del, 2732 sub ]
```

These can also be found at: `exp/chain_cleaned/tdnn_1d_sp/decode_dev${data_affix}_diarized_2stage_rescore/scoring_kaldi_multispeaker/best_wer`

**Note:** To evaluate performance using end-to-end Transformer based ASR, you would need
to first create an `s5_css` equivalent in ESPNet by extending the `asr1` recipe.

#### Example 2: A new cross-stream diarizer

One of the observations in our paper was that it is hard to perform good diarization
on top of separated audio streams:

1. Methods such as VBx cannot be used for this purpose because of their
time continuity constraint.

2. Although models like RPN and TS-VAD do well on mixed audio, they fail on
separated audio due to a train-test mismatch (they were trained on simulated
overlapping mixtures).

Yet, this "separation+diarization" system is very promising, especially
considering that such a [system from Microsoft](https://arxiv.org/pdf/2010.11458v2.pdf)
obtained the best performance in the recent VoxConverse diarization challenge.

Suppose you have a new diarization method which works across separated audio
streams. To evaluate your method on LibriCSS:

1. Download the separated audio data from Zenodo.

2. If your method is implemented in Kaldi, clone and install Kaldi and follow the
`s5_css` recipe until the diarization stage. Otherwise, run your implementation
on the separated audio files and compute the final DER.

3. You can compare your performance against the results reported in Table 3 of
our paper. The baselines can be reproduced by running the `s5_css` recipe till
stage 3.

<a name="credits"></a>

### Credits

This work was conducted during JSALT 2020 with support from Microsoft, Amazon,
and Google. If you use the data or code in your research, consider citing:

```
@article{Raj2021IntegrationOS,
  title={Integration of speech separation, diarization, and recognition for multi-speaker 
  meetings: System description, comparison, and analysis},
  author={Desh Raj and Pavel Denisov and Zhuo Chen and Hakan Erdogan and Zili Huang 
  and Maokui He and Shinji Watanabe and Jun Du and Takuya Yoshioka and Yi Luo and 
  Naoyuki Kanda and Jinyu Li and Scott Wisdom and John R. Hershey},
  journal={2021 IEEE Spoken Language Technology ({SLT}) Workshop},
  year={2021}}
}
```

### Other resources

* Paper: [Link](https://arxiv.org/abs/2011.02014)
* Slides: [Link](https://desh2608.github.io/static/ppt/slt21_jsalt_slides.pdf)