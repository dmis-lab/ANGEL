<h3 align="center">
<p>ANGEL
<a href="https://github.com/dmis-lab/ANGEL/blob/main/LICENSE">
   <img alt="GitHub" src="https://img.shields.io/badge/license-GPLv3-blue">
</a>
</h3>

<div align="center">
    <p>Le<b>a</b>rning from <b>N</b>egative Samples in <b>G</b>enerative Biomedical <b>E</b>ntity <b>L</b>inking Framework</p>
</div>

---

## Introduction
**ANGEL** is a novel framework designed to enhance generative biomedical entity linking (BioEL) by incorporating both positive and negative samples during training. 
Traditional generative models primarily focus on positive samples, which can limit their ability to distinguish between similar entities. 

**ANGEL** addresses this limitation through **preference optimization** with negative sampling, significantly improving model accuracy while maintaining the memory efficiency of generative approaches.

Key features of ANGEL include:
- **Memory Efficiency**: Retains the low memory footprint characteristic of generative models by avoiding the need for large external embeddings.
- **Negative-aware Learning**: Enhances the model's ability to differentiate between similar entities by using both correct and incorrect examples during training.

For a detailed description of our method, please refer to our [paper](https://arxiv.org/abs/2408.16493).

### Features

#### Memory-Efficient Generative Approach
ANGEL leverages a generative model that inherently requires less memory compared to similarity-based methods, making it suitable for large-scale biomedical applications.

#### Enhanced Learning through Negative Sampling
By integrating negative samples during training, ANGEL improves the model's ability to distinguish between entities that have similar surface forms but different meanings.

---

## Requirements
ANGEL requires two separate virtual environments: one for **positive-only training** and another for **negative-aware training**. 
Ensure that CUDA version 11.1 is installed for optimal performance.

To set up the environments and install the required dependencies, run the following script:

```bash
bash script/environment/set_environment.sh
```

## Dataset Preparation
The datasets (NCBI, BC5CDR, COMETA, and AAP) were used as provided by GenBioEL, while MedMentions was processed similarly using the GenBioEL code. 
If you need the pre-processing code, check out the [GenBioEL](https://github.com/Yuanhy1997/GenBioEL) repository. 
To download these datasets and set up the experimental environment, execute the following steps:

```bash
bash script/dataset/process_dataset.sh
```

### Dataset Format
For training, prepare the data in the following format:

train.source: Contains JSON lines with the input text, including marked mentions.
train.target: Contains JSON lines with two elements: the prefix the mention is and the target entity.

Example:

- '.source': ['Ocular manifestations of START juvenile rheumatoid arthritis END.']
- '.target': ['juvenile rheumatoid arthritis is ', 'juvenile rheumatoid arthritis']

For trie construction:

- If using prefix prompt tokens, set the trie root as 16 (the token ID for is).
- If not using prefix tokens, set the root as 2 (the BART decoder’s BOS token).

To experiment with your own dataset, preprocess it in the format described above.

## Pre-training

#### Positive-Only Training

We conducted positive-only pre-training using the code from [GenBioEL](https://github.com/Yuanhy1997/GenBioEL). 
If you wish to replicate this, follow the instructions provided in the GenBioEL repository.

#### Negative-Aware Training

Negative-aware pre-training was conducted using the code from [alignment-handbook](https://github.com/huggingface/alignment-handbook). 
This step refines the model’s ability to differentiate between closely related entities by learning from negative examples.


## Fine-tuning

#### Positive-Only Training

To fine-tune NCBI-disease dataset using positive-only training, run:
```bash
# NCBI-disease
bash bash script/train/train_positive.sh 0 ncbi 3e-7 20000
```

#### Negative-Aware Training

For negative-aware fine-tuning on the NCBI-disease dataset, execute:
```bash
# NCBI-disease
bash script/train/train_negative.sh 0 ncbi 1e-5
```


## Evaluation

#### Running Inference with the Best Model on Huggingface

To perform inference with our best model hosted on Huggingface, use the following script:
```bash
# NCBI-disease
bash script/inference/inference.sh 0 ncbi
```


#### BEST Score
|              Model                | Acc@1/Acc@5 | 
|:----------------------------------|:--------:|   
| [ANGEL-NCBI-disease](https://huggingface.co/chanwhistle/ANGEL_ncbi) | **92.8**/**95.7** | 
| [ANGEL-BC5CDR](https://huggingface.co/chanwhistle/ANGEL_bc5cdr) | **94.5**/**96.8** |
| [ANGEL-COMETA](https://huggingface.co/chanwhistle/ANGEL_cometa) | **82.8**/**88.6** |
| ANGEL-AskAPatient | **90.2**/**95.2** | 
| [ANGEL-MedMentions](https://huggingface.co/chanwhistle/ANGEL_mm) | **73.3**/**84.3**  | 



## Result

The results file in your model folder contains the final scores:
```json
  {
    "count_top1": 92.812,
    "count_top2": 94.062,
    "count_top3": 95.208,
    "count_top4": 95.625,
    "count_top5": 95.729,
    ...
}
```

Additionally, the file lists candidates for each mention, indicating correctness:
```json
{
    "correctness": "correct",
    "given_mention": "non inherited breast carcinomas",
    "result": [
      " breast carcinomas",
      " breast cancer",
      ...
    ],
    "cui_label": [
      "D001943"
    ],
    "cui_result": [
      ["D001943"],
      ["114480","D001943"],
      ...
      ]
}
```


## Citations

If interested, please cite:
```bibtex
@misc{kim2024learningnegativesamplesgenerative,
      title={Learning from Negative Samples in Generative Biomedical Entity Linking}, 
      author={Chanhwi Kim and Hyunjae Kim and Sihyeon Park and Jiwoo Lee and Mujeen Sung and Jaewoo Kang},
      year={2024},
      eprint={2408.16493},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.16493}, 
}
```
Some of our code are modified from YuanHongyi's [GenBioEL](https://github.com/Yuanhy1997/GenBioEL) work.