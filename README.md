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
**ANGEL** is a novel framework designed to enhance generative biomedical entity linking (BioEL) by incorporating both positive and negative samples during training. Traditional generative models primarily focus on positive samples during training, which can limit their ability to distinguish between similar entities. We address this limitation by using *direct preference optimization* with negative samples, significantly enhancing model accuracy.

Key features of ANGEL include:
- **Negative-aware Learning**: Enhances the model's ability to differentiate between similar entities by using both correct and incorrect examples during training.
- **Memory Efficiency**: Retains the low memory footprint characteristic of generative models by avoiding the need for large pre-computed entity embeddings.

For a detailed description of our method, please refer to our paper: 
- **Paper**: [Learning from Negative Samples in Generative Biomedical Entity Linking](https://arxiv.org/abs/2408.16493).

## Requirements
ANGEL requires two separate virtual environments: one for *positive-only training* and another for *negative-aware training*.
Please ensure that CUDA version 11.1 is installed for optimal performance.

To set up the environments and install the required dependencies, run the following script:

```bash
bash script/environment/set_environment.sh
```

## Dataset Preparation
Most datasets (i.e., NCBI-disease, BC5CDR, COMETA, and AskAPatient) were used as provided by GenBioEL, and MedMentions was processed in a similar manner using the GenBioEL code. 
If you need the pre-processing code, you can find it in the [GenBioEL](https://github.com/Yuanhy1997/GenBioEL) repository. 
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

<!-- Negative-aware pre-training was conducted using the code from [alignment-handbook](https://github.com/huggingface/alignment-handbook). 
This step refines the model’s ability to differentiate between closely related entities by learning from negative examples. -->

Negative-aware pre-training code will be open-sourced shortly.
The pre-trained model utilized in this project is uploaded on Hugging Face. 
If you want to use negative-aware pre-trained model, you can use the model with the following script:

```python
from transformers import AutoModel

# Load the model
model = AutoModel.from_pretrained("chanwhistle/ANGEL_pretrained")
```

## Fine-tuning

### Positive-Only Training

To fine-tune downstream dataset using positive-only training, run:
```bash
DATASET=ncbi # bc5cdr, cometa, aap, mm
LEARNING_RATE=3e-7 # 1e-5, 2e-5
STEPS=20000 # 30000, 40000

bash script/train/train_positive.sh $DATASET $LEARNING_RATE $STEPS
```
The script for other datasets is in the train_positive.sh file.

### Negative-Aware Training

For negative-aware fine-tuning on the downstream dataset, execute:
```bash
DATASET=ncbi # bc5cdr, cometa, aap, mm
LEARNING_RATE=2e-5 # 1e-5, 2e-5

bash script/train/train_negative.sh $DATASET $LEARNING_RATE
```
The script for other datasets is in the train_negative.sh file.

## Evaluation

### Running Inference with the Best Model on Huggingface

To perform inference with our best model hosted on Huggingface, use the following script:
```bash
DATASET=ncbi # bc5cdr, cometa, aap, mm

bash script/inference/inference.sh $DATASET
```


#### BEST Score
|              Model                | Acc@1/Acc@5 | 
|:----------------------------------|:--------:|   
| [ANGEL_ncbi](https://huggingface.co/chanwhistle/ANGEL_ncbi) | **92.8**/**95.7** | 
| [ANGEL_bc5cdr](https://huggingface.co/chanwhistle/ANGEL_bc5cdr) | **94.5**/**96.8** |
| [ANGEL_cometa](https://huggingface.co/chanwhistle/ANGEL_cometa) | **82.8**/**88.5** |
| ANGEL_aap | **90.2**/**95.2** | 
| [ANGEL_mm](https://huggingface.co/chanwhistle/ANGEL_mm) | **73.3**/**84.3**  | 



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

## Acknowledgement
Parts of the code are modified from [GenBioEL](https://github.com/Yuanhy1997/GenBioEL). We appreciate the authors for making GenBioEL open-sourced.
