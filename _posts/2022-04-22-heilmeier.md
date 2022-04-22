---
layout: post
title: "Heilmeier Catachism of my research"
tags: ["general"]
mathjax: true
---

Since I am preparing for my GBO exam (which is a kind of qualifying exam where a committee
evaluates your preparedness towards your PhD), my advisor Sanjeev Khudanpur suggested that
in addition to preparing low-level details, I should also be able to answer high-level
questions about my research. He mentioned something known as the 
[Heilmeier Catachism](https://www.depts.ttu.edu/research/ordc/Resources/heilmeier-catechism.php).

In summary, these are a set of questions that a former DARPA director came up with for
researchers proposing a new project. It consists of big-picture things like who would be impacted
by the outcome of the research, or how would you know if the project is a success. It is a
nice formalism to take a step back from the nitty-gritties of neural network training and
think about my research as a whole, so I decided to spend some time and answer the questions.
One of the instructions for answering the questions is that the answers should be as specific,
quantitative, and jargon-free as possible.

**Q1: What are you trying to do? Articulate your objectives using absolutely no jargon.**

I am trying to build systems that can transcribe multi-pary conversations real-time, while
also annotating the transcripts with who spoke which parts. I am trying to benchmark existing
systems that perform this task, and build new systems which are more accurate, have lower
latency, and are less complex than the existing systems.

**Q2: How is it done today, and what are the limits of current practice? Why are improvements needed? What are the consequences of doing nothing?**

Currently, it is done by building a pipeline consisting of smaller components that independently
handle voice activity detection, speech enhancement, speech recognition, and speaker diarization.
This pipeline approach has 4 major limitations: (i) errors can propagate from one component to the next, 
(ii) independent optimization may be sub-optimal, (iii) multiple decoding steps leads to 
higher latency, and (iv) the pipeline involves a lot of engineering effort, so it is difficult
to deploy. All these issues make it hard to use these systems in real scenarios. If we do
nothing, the advances in speech processing would be limited to isolated single-user settings,
and multi-part settings would not be able to benefit from them.

**Q3: What is new in your approach and why do you think it will be successful? What preliminary work have you done? How have you tested your assumptions on a small scale?**

My proposed approach builds upon the recent success in applying transducer models for speech
recognition. These models are naturally streaming and are part of the "end-to-end" era of ASR.
Most ASR services provided by companies such as Google and Microsoft are based on transducers.
As such, I think we have the infrastructure in place to adapt these models for multi-party 
conversations.

During an internship at Microsoft, I built upon Streaming Unmixing and Recognition Transducers,
which is a transducer-based multi-talker ASR system. On the LibriCSS dataset (which contains
simulated mixture of Librispeech utterances replayed in real meeting rooms), we were able to
obtain competetive performance with offline modular systems at a low latency of 150 ms.

**Q4: Who cares? Identify your stakeholders. Who will benefit from your successful project?**

The first-order stakeholders are big corporations such as Google and Microsoft which provide
ASR as a service. At present, their ASR market is mostly limited to single-user scenarios, such
as voice-based search. If we have real-time systems that can transcribe multi-party conversations,
it will unlock new applications such as real-time meeting transcription.

There are also secondary stakeholders, especially in education and healthcare. In order to understand
collaborative learning or therapy sessions, a lot of human effort goes into annotating the 
conversations. By building these systems, we will reduce this required effort which can instead
be redirected towards more meaningful tasks.

**Q5: If you're successful, what difference will it make? What will your successful project mean for your research? For the infrastructure of your institution and future capabilites? For your discipline? Related disciplines? For society? For the funding agency? What applications are enabled as a result?**

If I am successful in building robust systems for real-time multi-party transcriptions, it
will unlock further research into building dialog agents that communicate with groups of
users instead of single users. Since the research center has a strong focus on speech and
language processing, this is perhaps one of the ultimate goals that everyone is working towards.
A lot of my research ties together advances in computational linguistics and signal processing,
and so I would like to believe that this is a subject that is of interest for both of these
communities and would foster more collaborations.

**Q6: What are the risks and the payoffs? Why are the potential rewards worth the risk? What have you done to mitigate risk? What's Plan B?**

The biggest risk is that the system, as it is currently designed, does not work on real
settings with the current scale of model and data. Even if this turns out to be the case,
the project will still be useful since one of our aims is to formalize the different methods
for the task, benchmark them on public datasets, and analyze their advantages and
shortcomings.

**Q7: How much will it cost? How long will it take? Who needs to be involved to ensure success? What institutional resources need to be committed?**

Since end-to-end systems require training on large data, the most expensive part of the project
is the computational resources. We have 50k GPU hours of compute allocated on a compute cluster,
which we will use towards this project. We have some preliminary experiments in place, including
benchmarking of the baseline systems, and the remainder should take at most 2 years.

**Q8: What are the midterm and final "exams" to check for success? How will you assess progress and make midcourse corrections? What are the metrics for success? How will you know you're done?**

The task involves 2 aspects: word recognition, and speaker identification. The former is usually
measured through word error rate, and the latter through diarization error rate. For our
joint task, we will use a combination of these metrics known as concatenated minimum permutation
word error rate (cpWER). Preliminary results suggest that the cpWER of modular systems for LibriCSS
and AMI are ~15% and ~40%, respectively. Progress in this task would mean improving on these
error rates. Considering that a single-speaker AMI system can achieve under 20% WER when
provided oracle segments and a close-talking microphone recording, there is still a long way to go. 
