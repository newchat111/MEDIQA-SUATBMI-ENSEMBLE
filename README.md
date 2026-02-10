# SUAT-BMI: Ensemble of Few-Shot Learning, RAG Enhanced Response, and BERT Fine-Tuning.

This repository contains the end-to-end pipeline for the **MEDIQA 2026 Competition**. The workflow executes few-shot prompting using **QWEN30b**. 

**this repo is a complete mess, sorry :( We will try to organize our code in the following days in a more user-friendly way, thank you!
---
## GETTING STARTED

1.clone the repo

```bash
git clone -b H100 git@github.com:newchat111/MEDIQA-MEDGEMMA.git
```

2.download the model from huggingface Qwen/Qwen3-30B-A3B

## RUN THE SCRIPT
To run the model, simply change the paths in the .sh file. You need to define the data paths and the model path. You only need to change the variables that has comments behind them.

```bash
bash runs/qwen30b-bootstrap/infer.sh
```

The input file already has already been prepared. If you need to see the full pipeline, feel free to contact us! email:xinzhe@ucsb.edu

## OUR RESULTS

Our prediction is submission.zip. This prediction is an ensemble of all the best models we've had so far, which is a combination of RAG, BERT, and few-shot. However, due to the limited time, runs/qwen30b-bootstrap/infer.sh contains the few-shot method only. We will push all of our codes in the following days if we have the chance to write the paper, which will probably be a new git repository.
