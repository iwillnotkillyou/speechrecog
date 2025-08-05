# Locating and Detecting Language Model Grounding in Whisper with Fakepedia


## Introduction
For my project I decided to follow the paper "A Glitch in the Matrix?  Locating and Detecting Language Model Grounding with Fakepedia" and check how similar Whisper is to LLMs when it comes to activation patterns when hallucinating.


## The dataset
The dataset consists of factual wikipedia like entries where each of the statements is modified by replacing a Name to make the statement false. 


The original entries are selected such that a model pretrained on the internet ought to have learned them as factual. 


The model is expected to answer a question about one piece of information from the modified entry, a specific Subject Object relationship (i.e. question:Microsoft Office XP, a product developed by ... answer:(Microsoft,Nintendo, ...)), with the information from the entry. 


The answer is determined by the first token returned and is said to be grounded if it is faithful to the entry (i.e. the first token of the Object stated in the entry) if it is the first token of the true fact it is deemed unfaithful, otherwise it is ignored.


Because Whisper often rated words like “the”, “an” and “and” highly I had to modify this to look at whether the faithful token is contained in the top 20 and the unfaithful is not and symmetrically. I again ignore the cases when both are in the top 20 or neither is.


## The method
To analýze model behavior data for predicting whether the answer was based on the entry or not the authors use causal tracing. This is a method that allows to trace the influence of specific layers at input positions on the output of a model. The idea is to replace the tokens in the question containing the Subject with the mask tokens, which should hopefully have a relatively meaningfree embedding. Then when doing a forward pass we run both the corrupted input and the original input in parallel. We choose a module kind at a layer and restore the output value in the corrupted run with the original output value. We then analyze how much of the answer (the most probable answer regardless of whether grounded or unfaithful) output probability change is removed in the partially restored output.


## Setup
I borrowed a large part of the codebase from the original paper. The things I had to add was converting the text only fakepedia entries into audio files, and then running the Whisper encoder on these.


To turn the text into audio files I decided to use the text2wave utility from the festival package. It sounds very unnatural but is very legible. To check feasibility I reran the Whisper model on a subset of the librispeech dataset where I replaced the actual vocalization with the text2wave sound, as a result the WER rose from 5% to 10%. I consider this a plus because I worried that for natural unnoised inputs the conditioning would have too strong an effect and I could not ignore it during causal tracing. One issue was that I had to provide some speech even for the answer to the question.


For this I decided to use the grounded name but add an independently sampled normal noise with 0 mean and 0.2 std to each time step of its waveform (applied on waveforms in the torchaudio internal format ranging from -1 to 1). Once I had the conditioning signal the Whisper decoder could be treated similar to a decoder-only LLM.


I also added multiprocessing support so I didn't need to wait so long when processing fakepedia.


## Analysis
In this section I will replicate the graphs from the original paper for Whisper and present a detailed description of my Whisper hallucination detection model.


## Conclusions
It seems from the experiments that overall the Whisper decoder exhibits the same behavior as medium sized LLMs.

This is quite interesting given how strongly conditioned the model is. The speech signal, even when unnatural and, at the final position, noisy, should hold all the information necessary for predicting the final token. Even further surprising was the amount of impact masking the unnoised tokens in the decoder had.

It seems that the language model learned in the decoder is indeed somewhat strict when given uncertain audio. This likely explains why the model hallucinates so much when given out of distribution or noisy inputs.
