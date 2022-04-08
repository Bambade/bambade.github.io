---
layout: post
title: "GBO notes: i-vectors and x-vectors"
tags: ["gbo", "i-vectors"]
mathjax: true
---

In this note, we will review the two most popular speaker embedding extraction methods,
namely **i-vectors** and **x-vectors**. But first, it would be useful to quickly recap
generative and discriminative models.

Suppose we have some observed variables $X$ and some target variables $Y$. In the case of
speaker recognition, $X$ can be an audio sample spoken by a speaker, and $Y$ can be the
speaker id (or class index from a given set of speakers). A discriminative model directly
models the conditional probability $P(Y\mid X)$. In our example, the discriminative model
would output a probability distribution over the set of speakers, given any utterance.

A generative model is a model of the joint probability $P(X,Y) = P(X\mid Y)P(Y)$. It can also be used
to perform classification, by using the Bayes rule as follows:

$$ P(Y\mid X) = \frac{P(X,Y)}{P(Y)} = \frac{P(X\mid Y)P(Y)}{\sum_Y P(X\mid Y)P(Y)}.$$

Neural network models trained for classification tasks are discriminative models that predict logits 
which are then converted into a distribution over the targets using softmax.

## Generative embeddings with i-vectors

We basically need a way to characterize the variability in acoustic features resulting from
speaker and channel differences. Suppose the utterances are represented by a sequence of
feature vectors (usually MFCCs). A naive way may be to compute the average over the whole
sequence; but this would also capture things like lexical content, phonetic variation, and
so on. So what we actually want to do is to compare the *difference* between the features
of the target utterance compared with those from a general set of utterances, in the hope
that this difference would remove any effects arising from phonetic/lexical content.

This is done with the help of Gaussian mixture models (GMMs). First, we pool together a large
set of utterances and compute their features. We then assume that all the feature vectors
are generated I.I.D. using a GMM with a fixed number of components (usually 2048). We use
the EM algorithm to learn the parameters of this GMM, and call it the Universal Background
Model (UBM).

Suppose we now concatenate the means of all the UBM components into a giant vector (called a
"supervector"), and denote it as $m$. Suppose we adapt the UBM for a target speaker (or utterance)
and the concatenated means of this "target model" is denoted by $M$. Then, we can make
the assumption of factor analysis and claim that

$$ M = m + Tw, $$

where $T$ is a low-rank total variability matrix, and $w$ is the i-vector (total factors).
We can then say the the i-vector $w$ will characterize all the variations in the target
utterance arising from channel or speaker chracteristics.

The parameters of the model $T$ and $\Sigma$ are estimated jointly with the latent variable 
$w$ using the EM algorithm. In the E-step, we fix $T$ (randomly initialized) and $\Sigma$ 
(initialized from UBM), and compute the $w$ for each utterance. In the M-step, we use the 
sufficient statistics to update the model parameters.

At inference time, the i-vector of the utterance is computed as the MAP estimate of the feature
sequence given the model parameters, i.e.,

$$ \hat{w} = \text{arg} \max_{w} \prod_{c=1}^C \prod_{i=1}^N \mathcal{N}(o_i \mid M_c + T_c w, \Sigma_c) p(w), $$

where $C$ is the number of Gaussian components, and $o_1, \ldots, o_N$ is the sequence of feature
vectors.

## Discriminative embeddings with x-vectors

While i-vectors are mathematically attractive and useful in the sense that they do not need a
labeled training set, they are computationally expensive. Training the UBM model and the
parameters of the GMM is time-consuming, and since they model the total variability (i.e.,
speaker and channel), it is hard to tease apart the speaker-specific information.

Deep neural network based speaker embeddings have become popular as a discriminatively-trained
alternative to i-vectors. Given a sequence of feature vectors, we use a neural network to obtain
an utterance-level representation, which is then fed into a softmax layer that estimates a
distribution over a large set of speakers. The parameters of the network are trained using
gradients of a classification-style loss.

Conventionally, the x-vector neural network consists of TDNN (basically 1-D strided CNN)
layers are the bottom which operate at the frame-level, followed by a statistics pooling
layer. This layer aggregates over the whole input sequence and computes its mean and standard
deviation. These utterance-level stats are concatenated together and passed to further hidden
layers that operate at the segment level and produce the final representation that is fed
to the softmax.

At the time of inference, the output layer is discarded and the segment-level representation
immediately preceding it is used as the embedding, also called an x-vector.
