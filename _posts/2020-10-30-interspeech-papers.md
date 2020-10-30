---
layout: post
title: Desh's curated list of ASR, diarization, and related papers from Interspeech 2020
tags: ["conference","speech processing"]
mathjax: true
---
Interspeech 2020 just ended, and here is my curated list of papers that I found interesting from the proceedings.

*Disclaimer: This list is based on my research interests at present: ASR, speaker diarization, target speech extraction, and general training strategies.* 

## A. Automatic speech recognition

### I. Hybrid DNN-HMM systems

1. [ASAPP-ASR: Multistream CNN and Self-Attentive SRU for SOTA Speech Recognition](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2947.pdf)

	* Key contributions: multi-stream CNN for acoustic modeling and self-attentive simple recurrent unit for language modeling.
	* Using SpecAugment and N-best rescoring, achieves 1.75% and 4.46% on test-clean and test-other.
	* Most of the improvement seems to come from 24-layer SRU LM.

2. [Faster, Simpler and More Accurate Hybrid ASR Systems Using Wordpieces](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1995.pdf)

	* Removes GMM bootstrapping, decision tree building steps by using word-pieces instead of context-dependent phones, and CTC instead of cross-entropy training. 

3. [On Semi-Supervised LF-MMI Training of Acoustic Models with Limited Data](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2242.pdf)

	* Key idea: use an error detector mechanism to control which transcripts are used for semi-supervised training.
	* The error detector is a neural network classifier which takes the ASR decoded output and predicts if it contains errors.

4. [On the Robustness and Training Dynamics of Raw Waveform Models](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/0017.pdf)

	* They study raw waveforms as inputs (instead of MFCCs) on TIMIT, Aurora-4, and WSJ, in both matched and mismatched conditions.
	* In mismatched case, MFCCs perform better, but raw waveform performance can be improved using normalization techniques.
	* In matched techniques, raw waverforms performed better. Better alignments improve performance considerably.

5. [Speaker Adaptive Training for Speech Recognition Based on
Attention-over-Attention Mechanism](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1727.pdf)

	* Instead of using external embeddings (like d-vector) for speaker adaptation, the embedding is obtained using an attention mechanism on the frames of the utterance.
	* This paper improves upon a [previous work](http://www.apsipa.org/proceedings/2018/pdfs/0000183.pdf) by the authors by replacing frame-attention with attention-over-attention.

6. [Frame-wise Online Unsupervised Adaptation of DNN-HMM Acoustic Model from Perspective of Robust Adaptive Filtering](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1301.pdf)

	* Key things are "online" and "unsupervised", as opposed to existing methods which are offline and supervised.
	* This is done by formulating a gradient based on the conditional likelihood of the acoustic model and using a particle filter approach for efficient computation (I did not quite understand the details).

7. [Leveraging Unlabeled Speech for Sequence Discriminative Training of Acoustic Models](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2056.pdf)

	* Strong BLSTM teacher is used to guide the sequence discriminative training of LSTM student model.

8. [Context-Dependent Acoustic Modeling without Explicit Phone Clustering](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1244.pdf)

	* Replace state tying with a method that jointly models the tied context-dependent phones with DNN training.
	* Phonetic context representation relies on phoneme embeddings.
	* Interesting idea, but needs more empirical work to improve performance compared to standard tying approach.

### II. End-to-end models

1. [On the comparison of popular end-to-end models for large scale speech recognition](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2846.pdf)

	* Compared RNN-transducer, RNN-AED, transformer AED models on 65k hours of speech.
	* Transformer AED beats other systems in both streaming and non-streaming modes.
	* Both are better than hybrid model. 

2. [Single headed attention based sequence-to-sequence model for state-of-the-art results on Switchboard](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1488.pdf)

	* Through careful tuning of the whole seq2seq pipeline, they achieve SOTA performance on 300h SWBD subset.
	* Lots of ablation experiments; SpecAug seems to be the most helpful.

3. [SpecSwap: A Simple Data Augmentation Method for End-to-End Speech Recognition](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2275.pdf)

	* Idea: swap random time-bands and frequency-bands in the input spectogram.
	* From ablation, time swapping seems more useful than frequency swapping
	* Slightly worse than using SpecAug

4. [Speech Transformer with Speaker Aware Persistent Memory](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1281.pdf)

	* This is like i-vector based speaker adaptation (used often in hybrid ASR), but applied to end-to-end transformers.
	* All speaker i-vectors are concatenated into a matrix (called "persistent memory") and transformed to get matrices for key and value, and appended to the respective self-attention block matrices.
	* Consistent improvements on SWBD, Librispeech (100h), and AISHELL-1. 

5. [Robust Beam Search for Encoder-Decoder Attention Based Speech Recognition without Length Bias](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1958.pdf)

	* A new beam search algorithm is proposed which mitigates the problem with longer utterance decoding in ASR.
	* The new algorithm explicitly models utterance length in the sequence posterior, and is robust across different beam sizes.

6. [A New Training Pipeline for an Improved Neural Transducer](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1855.pdf)

	* Extensive experiments on the RNN-transducer model, showing that it outperforms attention-based models on long sequences (on SWBD).
	* __NOTE TO SELF:__ Needs more careful reading.

7. [Semi-supervised end-to-end ASR via teacher-student learning with conditional posterior distribution](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1574.pdf)

	* A new TS training scheme for E2E ASR models.
	* The scheme involves: (i) teacher forcing using 1-best hypothesis from a teacher model, and (ii) matching the student's conditional posterior to the teacher's posterior through the 1-best decoding path.
	* Improvements seen in WSJ and Librispeech.

8. [Early Stage LM Integration Using Local and Global Log-Linear Combination](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2675.pdf)

	* New method for integrating external LM into sequence-to-sequence model training: log-linear model combination with a per-token renormalization (as opposed to shallow fusion which is global renormalization).

9. [An investigation of phone-based subword units for end-to-end speech recognition](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1873.pdf)

	* One-pass decoding with BPE is introduced, with corresponding forward function and beam search decoding.
	* Phone BPEs outperform char BPEs on WSJ and SWBD.


### III. Training strategies

1. [Unsupervised Regularization-Based Adaptive Training for Speech Recognition](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1689.pdf)

	* New regularization objectives proposed for speaker adaptation of CTC-based models:
		* _Center loss_: penalizes the distances between speaker embeddings and the center
		* _Speaker variance loss_: minimizes the speaker interclass variance
	* Key idea is to remove speaker-specific deep embedding variances from the acoustic model

2. [Iterative Pseudo-Labeling for Speech Recognition](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1800.pdf)

	* Semi-supervised training with unlabeled data --- iteratively generate pseudo-labels for the unlabeled data in each iteration of training.
	* External LM and data augmentation is important to avoid local minima.
	* SOTA WERs on Librispeech (100h and 960h).
	* Also release a large text corpus from Project Gutenberg books (not overlapping with LibriVox and LibriSpeech).

3. [Improved Noisy Student Training for Automatic Speech Recognition](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1470.pdf)

	* Several techniques for semi-supervised training are proposed, which results in SOTA performance on Librispeech.
	* Normalized filtering score, sub-modular sampling, gradational filtering, gradational augmentation -> all are relatively simple ideas but useful in combination.

4. [Speech-XLNet: Unsupervised Acoustic Model Pretraining For Self-Attention Networks](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1511.pdf)

	* Using XL-Net like pretraining schemes for self-attention network based ASR models.
	* Pretraining objective is next-frame prediction. 
	* Importantly, L1 and L2 losses fail to converge, so they used something called Mean Absolute Error (Huber loss). Also, only last 20% frames are predicted.
	* Experiments conducted on both hybrid and E2E settings.

5. [Combination of end-to-end and hybrid models for speech recognition](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2141.pdf)

	* 3 ways of combining: (i) ROVER over 1-best hypothesis, (ii) MBR-based combination, and (iii) ROVER over N-best lists (approximation of (ii)).
	* MBR combination worked best, and consistently improved over best single model.
	* Note that length normalization was required for LAS and RNN-T models since they prefer shorter sequences.

## B. Speaker diarization and recognition

### I. Diarization

1. [End-to-End Speaker Diarization for an Unknown Number of Speakers with Encoder-Decoder Based Attractors](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1022.pdf)

	* This paper extends the EEND diarization system to unknown number of speakers.
	* This is done using encoder-decoder attractor (EDA). The idea is to pass the EEND hidden state to an LSTM encoder-decoder which can produce a flexible number of outputs.
	* The output of the attractor is passed to a sigmoid which decides when to stop. A threshold is used to then select the number of speakers.

2. [Target-Speaker Voice Activity Detection: a Novel Approach for Multi-Speaker Diarization in a Dinner Party Scenario](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1602.pdf)

	* TS-VAD was used in STC's winning submission to the CHiME-6 challenge.
	* The idea is to use speaker i-vectors from a first-pass diarization to perform a multi-label classification per-frame, which considers all speakers simultaneously.
	* Limitation: it can only handle a fixed number of speakers (4 in the case of CHiME-6).
	* My slides from a reading group presentation: [slides](https://desh2608.github.io/static/ppt/ts-vad.pdf)

3. [New advances in speaker diarization](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1879.pdf)

	* Multiple small tweaks in clustering-based diarization.
	* Key contributions:
		* using x-vectors + d-vectors 
		* using a neural networks for scoring segment similarity (also see [this paper](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1908.pdf) which uses self-attention based scoring)
		* better estimation of speaker count
	* All these tweaks provide some gains on Callhome dataset (8.6% to 5.1%).

4. [Speaker attribution with voice profiles by graph-based semi-supervised learning](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1950.pdf)

	* Key idea: create a graph with nodes as the subsegment embeddings and edges containing similarity. Then add profile nodes and use "label propagation" to assign labels to other nodes.

5. [Speaker Diarization System based on DPCA Algorithm For Fearless Steps Challenge Phase-2](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1666.pdf)

	* Key novelty is in using a new clustering method called Density Peak Clustering.
	* Performance is better than AHC and spectral clustering for data containing non-convex clusters and outliers.

6. [Detecting and Counting Overlapping Speakers in Distant Speech Scenarios](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2671.pdf)

	* Overlap detection and VAD is formulated as an OSDC task, and temporal convolutional networks (TCNs) are used to tackle it.
	* Experiments on AMI and CHiME-6 dataset show that TCNs are better than LSTM and CRNN models for this task.

### II. Recognition

1. [Improved RawNet with Feature Map Scaling for Text-independent Speaker Verification using Raw Waveforms](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf)

	* Proposes improvements over the [RawNet](https://arxiv.org/abs/1904.08104) architecture for end-to-end speaker verification from raw waveforms.
	* This is an alternative to the traditional approach which involves a front-end embedding extractor and a back-end like a PLDA classifier.

2. [In defence of metric learning for speaker recognition](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1064.pdf)

	* Through extensive experimentation (20k GPU hours) on VoxCeleb, they show that metric learning learns better speaker embeddings than classification-based losses.
	* Good overview of various loss functions, including a new angular prototypical loss.
	* __NOTE TO SELF:__ Needs careful reading.

3. [Wav2Spk: A Simple DNN Architecture for Learning Speaker Embeddings from Waveforms](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1287.pdf)

	* Replace MFCC, VAD, and CMVN with deep learning tools, and extract embeddings directly from waveforms.
	* Outperforms x-vector system on VoxCeleb1.

4. [A Comparative Re-Assessment of Feature Extractors for Deep Speaker Embeddings](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1765.pdf)

	* Extensive comparison of feature extraction methods (14 methods studied)
	* Contains good overview of available extraction methods

## C. Speech enhancement/separation

### I. Target speech extraction

1. [Neural Spatio-Temporal Beamformer for Target Speech Separation](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1458.pdf)

	* Multi-tap MVDR beamformer which uses complex-valued masks for enhancement in multi-channel scenario.
	* The model is jointly trained with ASR objective.

2. [SpEx+: A Complete Time Domain Speaker Extraction Network](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1397.pdf)

	* This is a follow up on their previous [SpEx model](https://arxiv.org/abs/2004.08326), with the difference that the speaker embedding is now also in time domain.
	* This is done to avoid phase estimation that would be required to reconstruct the target signal, if the extraction is performed in frequency domain (e.g. in SpeakerBeam).

3. [X-TaSNet: Robust and Accurate Time-Domain Speaker Extraction Network](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1706.pdf)

	* This is another time-domain speaker extraction model, but it's based on TasNet (which is SOTA for speech separation).
	* A new training strategy called __Speaker Presence Invariant Training (SPIT)__ is proposed, to solve the problem that happens when such models are asked to extract speakers not present in the mixture.

4. [Time-Domain Target-Speaker Speech Separation With Waveform-Based Speaker Embedding](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2108.pdf)

	* Another time-domain extraction model, but works entirely in time-domain. The model is called __WaveFilter__.
	* Auxiliary input is fed step-wise into the separation network through residual blocks.
	* Experiments show improvements in SDR over SpeakerBeam.

5. [VoiceFilter-Lite: Streaming Targeted Voice Separation for On-Device Speech Recognition](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1193.pdf)

	* This is a follow-up on VoiceFilter, which uses d-vectors to extract the speaker's speech from mixture.
	* The novelty is that this extraction is now performed directly on filterbanks, and an asymmetric L2 loss is used to mitigate over-suppression problem.
	* Model is made to fit on-device by 8-bit quantization.

### II. Speech enhancement

1. [Single-channel speech enhancement by subspace affinity minimization](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2982.pdf)

	* Learns separate speech and noise embeddings from the input using a subspace affinity loss function.
	* Theoretically proven to maximally decorrelate speech and noise representations; empirically outperforms other popular single-channel methods on VCTK.

2. [Noise Tokens: Learning Neural Noise Templates for Environment-Aware Speech Enhancement](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1030.pdf)

	* A "noise encoder" learns noise representation from the noisy speech, which is then used to obtain enhanced STFT magnitude.
	* Improves VoiceFilter performance in terms of PESQ and STOI.

3. [Real Time Speech Enhancement in the Waveform Domain](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2409.pdf)

	* Causal [DEMUCS](https://arxiv.org/abs/1911.13254) model that runs in real time and matches performance of non-causal models.
	* Trained using multiple objective functions.
	* Augmentation schemes: Remix, Band-Mask, Revecho, and random shift.


## D. Joint modeling, LMs, and others

1. [Joint Speaker Counting, Speech Recognition, and Speaker Identification for Overlapped Speech of Any Number of Speakers](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1085.pdf)

	* Extends the [Serialized Output Training](https://arxiv.org/abs/2003.12687) model for multispeaker ASR using an attention-based encoder-decoder.
	* Speaker inventory is used as auxiliary input from which the speaker embeddings are obtained.
	* Experiments conducted using simulated mimxtures from LibriSpeech.

2. [Identifying Important Time-frequency Locations in Continuous Speech Utterances](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2637.pdf)

	* By masking increasing regions of the spectogram while keeping the error rate consistent, the authors determine the important regions of the input.
	* They mention that they will use this analysis to aid data augmentation in future work.

3. [FusionRNN: Shared Neural Parameters for Multi-Channel Distant Speech Recognition](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2102.pdf)

	* A simple technique to perform early fusion of multi-microphone inputs, through a "fusion layer".
	* Consistent improvements on DIRHA dataset over delay-and-sum beamforming.

4. [Speaker-Conditional Chain Model for Speech Separation and Extraction](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2418.pdf)

	* The task is to extract the speech of all the speakers in a multi-speaker recording.
	* This is done by sequentially extracting the speech, conditioned on the embeddings of the previously extracted ones.
	* Promising results on both WSJ-mix and LibriCSS.
	* __Note:__ This "chain" model is closely associated with the EEND line of work on diarization.

5. [Neural Speech Completion](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2110.pdf)

	* Proposes "completion" tasks for speech-to-text and speech-to-speech, and trains encoder-decoder models for these.
	* Speech-to-text completion models performed better than RNN-LM and BERT baselines on WER (although I don't quite understand what is considered a "correct" completion).

6. [Serialized Output Training for End-to-End Overlapped Speech Recognition](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/0999.pdf)

	* Similar line of work as the joint training (see #1 in this list); task is multi-speaker overlapped ASR.
	* Transcriptions of the speakers are generated one after another.
	* Several advantages over the traditional permutation invariant training (PIT).

7. [Multi-talker ASR for an unknown number of sources: ()Joint training of source counting, separation and ASR](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2519.pdf)

	* Main novelty is that the model works for unknown number of speakers, using an iterative speech extraction system.
	* Promising results on WSJ-mix with 2,3, and 4 speakers.

8. [Rescore in a Flash: Compact, Cache Efficient Hashing Data Structures for N-gram Language Models](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1939.pdf)

	* Data struture "DashHashLM" to store n-gram LMs with efficient lookup for rescoring.
	* 6x query speedup at 10% increased memory requirement.

9. [LVCSR with Transformer Language Models](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1164.pdf)

	* Several tweaks are proposed to improve rescoring time with transformer LMs, and to use them in single-pass systems.
	* Proposed improvements include quantization of LM state, common prefix, and a hybrid lattice/n-best list rescoring.

10. [Vector-Quantized Autoregressive Predictive Coding](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1228.pdf)

	* __Best paper award__
	* Quantized representations are produced using the APC self-supervised objective.
	* Probing tasks and mutual information used to show the presence and absence of information in learned representations from increasingly limited models.

## E. Datasets

1. [Spot the conversation: speaker diarisation in the wild](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2337.pdf)

	* New _VoxConverse_ dataset for multi-modal diarization.
	* Dev: 1218 mins, test: 53 hours.

2. [DiPCo - Dinner Party Corpus](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2800.pdf)

	* Dinner party like conversations with close-talk and array microphone recordings.
	* 10 sessions, between 15 and 45 minutes
	* Using the Kaldi CHiME-5 acoustic model with adaptation provides approx. 80% WER on far-field setting.

3. [Speech recognition and multi-speaker diarization of long conversations](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/3039.pdf)

	* Long-form multi-speaker recordings (approx 1 hour each) collected from _This American Life_ podcast.
	* Contains approx 640 hours of speech comprising 6608 unique speakers.
	* Aligned transcripts are made [publicly available](https://github.com/calclavia/tal-asrd)

4. [JukeBox: A Multilingual Singer Recognition Dataset](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2972.pdf)

	* 467 hours of singing audio sampled at 16 kHz, containing 936 unique singers and 18 different languages.
	* Publicly available [here](http://iprobe.cse.msu.edu/dataset_detail.php?id=8&?title=JukeBox:_A_Speaker_Recognition_Dataset_with_Multi-lingual_Singing_Voice_Audio).

5. [MLS: A Large-Scale Multilingual Dataset for Speech Research](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2826.pdf)

	* Multilingual Librispeech data containing 32K hours of English and 4.5k in other languages.
	* Will be made available on [OpenSLR](www.openslr.org).
	* Paper includes baselines using wav2letter++.

## F. Toolkits

1. [PYCHAIN: A Fully Parallelized PyTorch Implementation of LF-MMI for End-to-End ASR](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/3053.pdf)

	* Fully parallelized PyTorch implementation of end-to-end LFMMI.
	* Provides wrapper around the forward-backward computations required for LFMMI gradient computation in Kaldi, so that it can be used for PyTorch based nnet training.

2. [Asteroid: the PyTorch-based audio source separation toolkit for researchers](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1673.pdf)

	* Provides Kaldi-style reproducible recipes for single-channel source separation datasets.
	* Contains implementations of popular architectures performing at par with the reference papers, such as deep clustering, TasNet, WaveSplit, etc.

3. [Surfboard: Audio Feature Extraction for Modern Machine Learning](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2879.pdf)

	* Feature extraction Python library
	* Can be used in native Python or as CLI tool.
