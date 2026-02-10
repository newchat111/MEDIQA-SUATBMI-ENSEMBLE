# MEDIQA-EVAL 2026

## Introduction

In the 2026 MEDIQA-EVAL task as part of CLINICALNLP@LREC, we are pleased to present the first shared task on medical open response evaluation. Leveraging open QA datasets from MEDIQA-M3G 2024 and MEDIQA-WV 2025, we present participants with expert ratings of multiple LLM system responses across dermatology and woundcare, in English and Chinese. The task is to provide automatic evaluation scores that are most correlated with human evaluations. We hope the dataset here can encourage more work in medical natural language evaluation.


## Source Datasets

In this challenge, we draw from two underlying multimodal datasets. These datasets were created from online posts with images and textual content. Responses are given by doctors.

DermaVQA: Dermatology dataset part of the MEDIQA-M3G challenge
Dataset: https://osf.io/72rp3/overview
Paper: https://papers.miccai.org/miccai-2024/paper/2444_paper.pdf

WoundcareVQA: Wound care dataset used in the MEDIQA-WV 2025 challenge.
Dataset: https://osf.io/xsj5u/overview
Paper: https://www.sciencedirect.com/science/article/pii/S1532046425001170


| dataset | split | #query | #gold-responses | #system-responses | lang |
|---------|-------|--------|-----------------|-------------------|------|
| woundcare | valid | 105 | 210| 315 | {en,zh} |
| woundcare | test | 93 | 279 | 279 | {en,zh} |
| dermavqa-iiyi | valid | 56 | 417 | 168 | {en,zh} |
| dermavqa-iiyi | test | 100 | 926 | 300 | {en,zh} |


We generate 3 systems for each case.


## Human evaluations


Each system response for EN {woundcare/iiyi} datasets was rated according to the following description by a practicing medical doctor:
- **disagree_flag**: 1 if expert disgrees, 0 otherwise
- **completeness**: {0,0.5,1.0} 1 for complete answer to question, 0.5 partial, 0.0 inaccurate/missing critical information
- **factual-accuracy**: {0,0.5,1.0} 1 for factually acurate answer to question, 0.5 partial, 0.0 inaccurate/missing critical information
- **relevance**: {0,0.5,1.0} 1 for relevant question, 0.5 partially relevant, 0.0 irrelevant information
- **writing-style**: {0,0.5,1.0} 1 for appropriate writing style, 0.5 partial, 0 otherwise
- **overall**: 1 for complete answer to question, 0.5 partial, 0.0 inaccurate/missing critical information
EN woundcare test dataset had 2 raters; where as EN woundcare valid had 1 rater.

Each system response for ZH {woundcare/iiyi} datasets was rated according to the following description by 1 domain expert trained at a Chinese Medical School:
- **factual-consistency-wgold**: {0,0.5,1.0} 1 for factual consistency with gold standard, 0.5 partial, 0 otherwise
- **writing-style**: {0,0.5,1.0} 1 for appropriate writing style, 0.5 partial, 0 otherwise


#### Total Judgements

**EN**
| dataset | split | #system-responses | #raters |
|---------|-------|-------------------|---------|
| woundcare | valid | 315 | 1 |
| woundcare | test | 279 | 2 |
| iiyi | valid | 168 | 2 |
| iiyi | test | 300 | 2 |

**ZH**
| dataset | split | #system-responses | #raters |
|---------|-------|-------------------|---------|
| woundcare | valid | 315 |  1 |
| woundcare | test | 279 |  1 |
| iiyi | valid | 168 | 1 |
| iiyi | test | 300 | 1 |


## Data

Both human/system files will be in CSV files with the following columns:
- **dataset**: dataset name {iiyi, woundcare}
- **encounter_id**: encounter id
- **candidate**: system output candidate
- **candidate_author_id**: candidate author_id, we provide three systems: {'SYSTEM001','SYSTEM002','SYSTEM003'}
- **lang**: language {'en','zh'}
- **rater_id**: this will be the expert rater id; for system, you can put your eval metric system id here
- **metric**: metric you are evaluating for en you should include {disagree_flag,completeness,factual-accuracy,relevance,- writing-style,overall}, for chinese {factual-consistency-wgold,writing-style,overall}
- **value**: your metric value

You can name your rater_id anything, however in our code, we will take the first appearing entry with a unique key combination: dataset, language, encounter_id, candidate_author_id, metric.

If you only want to work on one language, please still include the full number of appropriate rows. Then set rater_id='NA' and value=-1.

Note the valid set has both less raters and fewer gold standards. Our preliminary experiments have found that the best systems on the test set may not be the same as the valid set. We encourage participants to identify ways to enrich the train/validation data with additional data/human ratings.


### Expected Number of Rows for Automatic Metrics

Please check that your output files have the correct number of rows.

**VALID**
| dataset | lang | split | #system-responses | #metrics | #automatic-judgements |
|---------|------|-------|------------------|-----------|-----------------------|
| iiyi | en | valid | 168 | 6 | 1008 |
| woundcare | en | valid | 315 | 6 | 1890 |
| iiyi | zh | valid | 168 | 2 | 336 |
| woundcare | zh | valid | 315 | 2 | 630 |

Valid-total: 3864
Valid-EN: 2898
Valid-ZH: 966

**TEST**
| dataset | lang | split | #system-responses | #metrics | #automatic-judgements |
|---------|------|-------|------------------|-----------|-----------------------|
| iiyi | en | test | 300 | 6 | 1800 |
| woundcare | en | test | 279 | 6 | 1674 |
| iiyi | zh | test | 300 | 2 | 600 |
| woundcare | zh | test | 279 | 2 | 558 |

Test-total: 4632
Test-EN: 3474
Test-ZH: 1158


## Code

Code for calculating correlations {spearman,kendall,pearson} are located in the `mediqa_eval_script.py`.

- **mediqa_eval_script.py**: evaluation script used in the codabench evaluation
- **create_sample_systemfiles-valid.ipynb**: notebook illustrating how to produce a system output with the correct number of rows

You can run the evaluation with the following command:
```
python mediqa_eval_script.py <human-eval-ratings> <auto-scorer-ratings>
```