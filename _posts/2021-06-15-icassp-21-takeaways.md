---
layout: post
title: My 3 takeaways from IEEE ICASSP 2021
tags: ["conference","speech processing"]
mathjax: true
---
I attended the virtual ICASSP 2021, and this is a short post with my 3 key take-aways from the conference. As with my previous conference summary posts, this post is heavily biased by my research interests --- speech recognition and speaker diarization.

### One: Self-training and contrastive learning are here to stay

It is no secret that Facebook is heavily backing self-supervised learning, due in large part to the success of huge pre-trained models in NLP (read: BERT and friends). [Wav2vec 2.0](https://arxiv.org/abs/2006.11477) is undoubtedly the first big success story for SSL in speech, particularly with its availability in [HuggingFace](https://huggingface.co/transformers/model_doc/wav2vec2.html). [Contrastive learning](https://arxiv.org/abs/2002.05709) has played an important role in speech SSL, since pre-training (or "self-training") often involves predicting latent representations for masked or future frames, learned by contrasting positive examples against a batch of negative examples. At ICASSP, the following papers stood out in this category.

1. **Hubert: How much can a bad teacher benefit ASR pre-training? (Facebook, CMU)**
[Paper](https://ieeexplore.ieee.org/document/9414460){: .btn}
    * This paper proposes a simple pre-training strategy by predicting class labels which are cluster identities of filterbanks.
    * The loss function is cross-entropy of masked tokens, and so it does not require any negative sample selection unlike contrastive loss based methods.

2. **Contrastive learning of general-purpose audio representations (Google)**
[Paper](https://arxiv.org/abs/2010.10915){: .btn}
    * Idea: different segments from the same audio should have similar representations, and those from different recordings should be different.
    * This idea is implemented with a contrastive loss and used to self-train a model on Audioset.
    * On 9 classification tasks, models fine-tuned from COLA show improvement over supervised baselines.

3. **Contrastive semi-supervised learning for ASR (Facebook)**
[Paper](https://arxiv.org/abs/2103.05149){: .btn}
    * In conventional pseudo-labeling, the student learns from labels predicted by the teacher. But this can be detrimental if the teacher's predictions are noisy.
    * Instead, this paper uses the teacher's predictions to select positive and negative examples for contrastive learning.
    * Some tricks: label-aware batching, large batches, grad accumulation, STM masking.

### Two: Transducer models + teacher-student learning = streaming ASR

At this point it's an open secret that the big players (Google, Amazon, Microsoft) are using [transducer-based models](https://arxiv.org/abs/1211.3711) (colloquially called "RNN-Transducers") for streaming speech recognition in their production systems. While Google is still making progress with their [2-pass LAS based systems](https://arxiv.org/abs/1908.10992), there were several papers investigating ways to make streaming transducers faster and more accurate:

1. **A better and faster end-to-end model for streaming ASR: Google**
[Paper](https://arxiv.org/abs/2011.10798){: .btn}
    * Better performance by replacing RNNs with [Conformers](https://arxiv.org/abs/2005.08100).
    * Faster (in terms of partial latency) by using [FastEmit](https://arxiv.org/abs/2010.11148).

2. **Cascaded encoders for unifying streaming and non-streaming ASR: Google**
[Paper](https://arxiv.org/abs/2010.14606){: .btn}
    * Proposes a model which can work in both streaming and non-streaming modes.
    * Both encoders share the causal layers, but non-causal encoder additionally has non-causal layers.
    * Trained by stochastically choosing mode of operation with a shared decoder.

3. **Less is more: Improved RNN-T decoding using limited label context and path merging: Google**
[Paper](https://arxiv.org/abs/2012.06749){: .btn}
    * Finding 1: RNN-T decoder context can be limited to 4 word-piece tokens without degrading WER.
    * Finding 2: Since history is now limited to last 4 tokens, path merging can be done to improve lattice diversity.

4. **Efficient knowledge distillation for RNN-T models: Google**
[Paper](https://arxiv.org/abs/2011.06110){: .btn}
    * RNN-T loss is computed by summing over all encoder-output alignments, so naively doing distillation would involve the teacher and student's lattice being equal.
    * But this requires large memory, so the  K-dimensional posterior is reduced to 3-dim by representing all non-blank output tokens as a single token, i.e., only learning the teacher's transition behavior.
    * Overall loss is linear combination of RNN-T loss and distillation loss.

5. **Developing real-time streaming transformer-transducer for speech recognition on large-scale dataset: Microsoft**
[Paper](https://arxiv.org/abs/2010.11395){: .btn}
    * Two tricks to make real-time streaming model: truncated history, and limited right context.
    * Both of these can be done by manipulating the attention mask matrix.
    * Faster inference tricks: caching and chunk-wise compute.

6. **Improving streaming ASR with non-streaming model distillation on unsupervised data: Google**
[Paper](https://research.google/pubs/pub50119/){: .btn}
    * Non-streaming teacher is used to generate pseudo-labels for large unsupervised data.
    * This is then used to distill knowledege for streaming student model.

Given the proven dominance of transformers in end-to-end modeling with large data (and also the use of Conformers in some of the papers), it seems to be a no-brainer that the several advances in streaming transformer-based encoder-decoder models (see: [MoChA](https://arxiv.org/abs/1712.05382) and [papers from Sony](https://arxiv.org/abs/1910.11871)) are waiting to be applied to transformer-transducers.

### Three: Speaker diarization is wide open

Similar to ASR moving from hybrid HMM-DNN models to end-to-end approaches, speaker diarization is also witnessing new paradigms. The 2020s were dominated by methods that could be largely described as "clustering of DNN-based speaker embeddings". Popular embedding methods were either [TDNN-based neural networks](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) with stats pooling ("x-vectors"), or [fully connected networks](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf) ("d-vectors"), which replaced the [GMM-based embeddings](https://www.crim.ca/perso/patrick.kenny/Najim_TASLP2009.pdf) ("i-vectors") that were popular at the start of the decade.

These clustering-based approaches have 2 major problems: (i) they do not handle overlaps naturally; (ii) all the components are independently optimized. Recently, a new paradigm of supervised training based modeling has become popular, spearheaded by Hitachi's [EEND](https://arxiv.org/abs/2003.02966). It has quickly become applicable to the diarization of audio from different domains, as seen in its use in [DIHARD-3](https://arxiv.org/abs/2102.01363). Still, it suffers from 2 key challenges: (i) estimation of number of speakers; and (ii) difficulty in long recordings.

At ICASSP '21, researchers from Hitachi and NTT proposed 2 different ways to combine EEND with clustering-based systems. A drastically new (and well-performing) approach was seen in Microsoft's system for the VoxSRC challenge, which introduced continuous speech separation for diarization.

1. **End-to-end speaker diarization as post-processing (Hitachi, JHU)**
[Paper](https://arxiv.org/abs/2012.10055){: .btn}
    * A 2-speaker EEND model is really good at diarization of exactly 2 speakers. Clustering based systems are good at estimating the number of speakers.
    * So a clustering-based diarization is applied first, and then 2-speaker chunks are fed into EEND to refine the diarization output.
    * During the refinement, the EEND also handles overlapping speech between the 2 speakers.
    * **Note:** This model gave the single-best DER for our experiments in DIHARD 3.


2. **Integrating end-to-end neural and clustering-based diarization: Getting the best of both worlds (NTT)**
[Paper](https://arxiv.org/abs/2010.13366){: .btn}
    * NTT researchers approach the issue from the other angle: they modify EEND to output global speaker embeddings.
    * These embeddings are then used to perform speaker clustering across the whole recording.
    * One advantage of this method over Hitachi's approach may be that we can use differentiable clustering to perform end-to-end optimization of the system.


3. **Microsoft Speaker Diarization System for the VoxCeleb Speaker Recognition Challenge 2020 (Microsoft)**
[Paper](https://arxiv.org/abs/2010.11458){: .btn}
    * For highly overlapping recordings, a good separation system can be a game-changer. Microsoft demonstrated this through the use of their [CSS system](https://arxiv.org/abs/2001.11482) with a clustering-based diarization.
    * Some finer details need to be handled carefully, such as tricks for avoiding leakage, and using strong embedding extractors for the clustering. The CSS module itself needs to be well-trained.