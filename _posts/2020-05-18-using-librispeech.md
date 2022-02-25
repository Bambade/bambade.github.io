---
layout: post
title: How to use the pre-trained Librispeech model in Kaldi
tags: ["kaldi","speech recognition"]
mathjax: true
---
_[Update on Feb 25, 2022]_ The pre-trained model did not have a `frame_subsampling_factor` file, which is required for correct decoding. This has now been added and WER results updated for WSJ. The high WERs earlier were due to train-test mismatch in the subsampling factor.

This is a tutorial on how to use the [pre-trained Librispeech model](http://kaldi-asr.org/models/m13) available from kaldi-asr.org to decode your own data. For illustration, I will use the model to perform decoding on the [WSJ data](https://catalog.ldc.upenn.edu/LDC94S13A). 

## Setting up Kaldi

[Josh Meyer](http://jrmeyer.github.io/asr/2016/01/26/Installing-Kaldi.html) and [Eleanor Chodroff](https://www.eleanorchodroff.com/tutorial/kaldi/installation.html) have nice tutorials on how you can set up Kaldi on your system. Follow either of their instructions. 

## Preparing the decoding data

First we prepare the data that we will be decoding. Since Kaldi already has a [WSJ recipe](https://github.com/kaldi-asr/kaldi/tree/master/egs/wsj), I will just use that for the purpose of illustration. If you want to decode your own data, you will need to first create a recipe (without any training stages). You should look at [this documentation](http://kaldi-asr.org/doc/data_prep.html) page, especially the section on "Files you need to create yourself".  

#### Files you need to create yourself

From a barebones perspective, you only need a directory `data/<your-data-dir>` containing 3 files:

1. `wav.scp`: This has a list of utterance ids and corresponding wav locations on your system
2. `utt2spk`: List of utterance ids and corresponding speaker ids. If you don't have speaker information, you can just replicate the utt-id as the spk-id.
3. `text`: The transcriptions for the utterances. This will be needed to score your decoding output. 

For our WSJ example, I will decode the `dev93` and `eval92` subsets. So first I need to prepare these. 
Set `wsj0` and `wsj1` as the paths to the downloaded LDC corpora WSJ0 and WSJ1, and then run:

```console
foo@bar:~kaldi/egs/wsj/s5$ local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?
```

After preparing, your directory would look like this:

```console
foo@bar:~kaldi/egs/wsj/s5$ tree data/test*
data
├── test_dev93
│   ├── spk2utt
│   ├── text
│   ├── utt2spk
│   └── wav.scp
└── test_eval92
    ├── spk2utt
    ├── text
    ├── utt2spk
    └── wav.scp
```

#### Feature extraction

Now that we have prepared our decoding data, we need to generate MFCC features. Note that we only need 40-dim MFCCs for each dataset, since we will not be decoding using any GMM model.

We create a `conf` directory containing configuration options for the MFCC:

```console
foo@bar:~kaldi/egs/wsj/s5$ mkdir conf & cd conf
foo@bar:~kaldi/egs/wsj/s5$ touch mfcc_hires.conf
```

Add the following in `mfcc_hires.conf`:

```console
--use-energy=false   # use average of log energy, not energy.
--num-mel-bins=40     # similar to Google's setup.
--num-ceps=40     # there is no dimensionality reduction.
--low-freq=20     # low cutoff frequency for mel bins... this is high-bandwidth data, so
                  # there might be some information at the low end.
--high-freq=-400 # high cutoff frequently, relative to Nyquist of 8000 (=7600) 
```

Now we compute features and CMVN stats for our data.

```console
foo@bar:~kaldi/egs/wsj/s5$ for datadir in test_eval92 test_dev93; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done

foo@bar:~kaldi/egs/wsj/s5$ for datadir in test_eval92 test_dev93; do
    steps/make_mfcc.sh --nj 20 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires
    steps/compute_cmvn_stats.sh data/${datadir}_hires
    utils/fix_data_dir.sh data/${datadir}_hires
  done
```

After the feature extraction is successfully completed, your data directory should contain the following files:

```console
foo@bar:~kaldi/egs/wsj/s5$ tree data/test_dev93_hires -L 1
data/test_dev93_hires
├── cmvn.scp
├── conf
├── data
├── feats.scp
├── frame_shift
├── log
├── q
├── spk2utt
├── text
├── utt2dur
├── utt2num_frames
├── utt2spk
└── wav.scp

4 directories, 9 files
```

Now we are ready to download and use the pre-trained model!

## Downloading the pre-trained model

The following models are available in the [LibriSpeech kit](http://kaldi-asr.org/models/m13):

  1. TDNN-F chain model
  2. i-vector extractor
  3. Language models (pruned 3-gram and RNNLM)

```console
foo@bar:~kaldi/egs/wsj/s5$ wget http://kaldi-asr.org/models/13/0013_librispeech_v1_chain.tar.gz
foo@bar:~kaldi/egs/wsj/s5$ wget http://kaldi-asr.org/models/13/0013_librispeech_v1_extractor.tar.gz
foo@bar:~kaldi/egs/wsj/s5$ wget http://kaldi-asr.org/models/13/0013_librispeech_v1_lm.tar.gz
```

Once they have finished downloading, we extract them. By default, the contents
will get extracted to the `data` and `exp` directories.

```console
foo@bar:~kaldi/egs/wsj/s5$ tar -xvzf 0013_librispeech_v1_chain.tar.gz
foo@bar:~kaldi/egs/wsj/s5$ tar -xvzf 0013_librispeech_v1_extractor.tar.gz
foo@bar:~kaldi/egs/wsj/s5$ tar -xvzf 0013_librispeech_v1_lm.tar.gz
```

We have provided 2 language models: `tgsmall` (small trigram model) and `rnnlm` (LSTM-based), both
of which are trained on the LibriSpeech training transcriptions. We will use the `tgsmall` model
for decoding and the RNNLM for rescoring.

The `exp/chain_cleaned` directory contains the pre-trained chain model, and the `exp/nnet3_cleaned` contains the ivector extractor. Now we are ready for decoding.

## Using the model for decoding

We will do the following:

1. Extract i-vectors for the test data
2. Decode using the small trigram LM
3. Rescore using the RNNLM


#### Extracting i-vectors

First we use the i-vector extractor to obtain i-vectors for our test data.

```console
foo@bar:~kaldi/egs/wsj/s5$ for data in test_dev93 test_eval92; do
    nspk=$(wc -l <data/${data}_hires/spk2utt)
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj "${nspk}" \
      data/${data}_hires exp/nnet3_cleaned/extractor \
      exp/nnet3_cleaned/ivectors_${data}_hires
  done
```

This will extract 100-dim i-vectors to `exp/nnet3_cleaned`. 

#### Decoding

We first create the decoding graph using the `tgsmall` LM:

```console
foo@bar:~kaldi/egs/wsj/s5$ export dir=exp/chain_cleaned/tdnn_1d_sp
foo@bar:~kaldi/egs/wsj/s5$ export graph_dir=$dir/graph_tgsmall
foo@bar:~kaldi/egs/wsj/s5$ utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov \
  data/lang_test_tgsmall $dir $graph_dir
```

Now we decode using the created graph:

```console
foo@bar:~kaldi/egs/wsj/s5$ export decode_cmd="queue.pl --mem 2G"
foo@bar:~kaldi/egs/wsj/s5$ for decode_set in test_dev93 test_eval92; do
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
    --nj 8 --cmd "$decode_cmd" \
    --online-ivector-dir exp/nnet3_cleaned/ivectors_${decode_set}_hires \
    $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}_tgsmall
done
```

Let us check the WER at this point before rescoring. Here we use `sclite` for scoring, which
is available in Kaldi and used for most of the egs.

```console
foo@bar:~kaldi/egs/wsj/s5$ for decode_set in test_dev93 test_eval92; do
  steps/score_kaldi.sh --cmd "run.pl" data/${decode_set}_hires $graph_dir $dir/decode_${decode_set}_tgsmall
done
foo@bar:~kaldi/egs/wsj/s5$ cat exp/chain_cleaned/tdnn_1d_sp/decode_test_dev93_tgsmall/scoring_kaldi/best_wer
%WER 14.22 [ 1171 / 8234, 183 ins, 93 del, 895 sub ] exp/chain_cleaned/tdnn_1d_sp/decode_test_dev93_tgsmall/wer_11_0.5
foo@bar:~kaldi/egs/wsj/s5$ cat exp/chain_cleaned/tdnn_1d_sp/decode_test_eval92_tgsmall/scoring_kaldi/best_wer
%WER 11.73 [ 662 / 5643, 94 ins, 29 del, 539 sub ] exp/chain_cleaned/tdnn_1d_sp/decode_test_eval92_tgsmall/wer_10_1.0
```

As a comparison, a model trained on the WSJ training data, and using a matched LM gives ~6.9% WER on both dev and eval at this stage. 

#### Rescoring

We now rescore using the RNNLM. Note that if your directory does not contain an `rnnlm` symlink, you can create it
using `ln -s ../../../scripts/rnnlm .`.

```console
foo@bar:~kaldi/egs/wsj/s5$ export decode_cmd="queue.pl --mem 2G"
foo@bar:~kaldi/egs/wsj/s5$ for decode_set in test_dev93 test_eval92; do
    decode_dir=exp/chain_cleaned/tdnn_1d_sp/decode_${decode_set}_tgsmall;
    rnnlm/lmrescore_pruned.sh \
        --cmd "$decode_cmd --mem 8G" \
        --weight 0.45 --max-ngram-order 4 \
        data/lang_test_tgsmall exp/rnnlm_lstm_1a \
        data/${decode_set}_hires ${decode_dir} \
        exp/chain_cleaned/tdnn_1d_sp/decode_${decode_set}_rescore
done
```

The scoring is included in the `lmrescore_pruned.sh` script.

```console
foo@bar:~kaldi/egs/wsj/s5$ cat exp/chain_cleaned/tdnn_1d_sp/decode_test_dev93_rescore/scoring_kaldi/best_wer
%WER 11.79 [ 971 / 8234, 159 ins, 80 del, 732 sub ] exp/chain_cleaned/tdnn_1d_sp/decode_test_dev93_rescore/wer_11_1.0
foo@bar:~kaldi/egs/wsj/s5$ cat exp/chain_cleaned/tdnn_1d_sp/decode_test_eval92_rescore/scoring_kaldi/best_wer
%WER 9.75 [ 550 / 5643, 81 ins, 25 del, 444 sub ] exp/chain_cleaned/tdnn_1d_sp/decode_test_eval92_rescore/wer_13_1.0
```

Finally, the obtained WERs are shown in the table below:

| System  | test_dev93 | test_eval92 |
|---------|------------|-------------|
| tgsmall | 14.22      | 11.73       |
| rnnlm   | 11.79      | 9.75       |
