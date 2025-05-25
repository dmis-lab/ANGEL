<h3 align="center">
<p>ANGEL
<a href="https://github.com/dmis-lab/ANGEL/blob/main/LICENSE">
   <img alt="GitHub" src="https://img.shields.io/badge/license-GPLv3-blue">
</a>
</h3>

<div align="center">
    <p>Le<b>a</b>rning from <b>N</b>egative Samples in <b>G</b>enerative Biomedical <b>E</b>ntity <b>L</b>inking Framework</p>
</div>

![Main Image](https://github.com/dmis-lab/ANGEL/blob/main/assets/main_figure_git.png)

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

- source: Input text including marked mentions with **START** and **END**.
- ex) ['Ocular manifestations of **START** juvenile rheumatoid arthritis **END**.']
- target: Pair of the prefix ['mention is'] and the ['target entity'].
- ex) ['juvenile rheumatoid arthritis is ', 'juvenile rheumatoid arthritis']

A Prefix Tree (or **Trie**) is a type of tree data structure used to efficiently store and manage a set of strings. 
Each node in the Trie represents a single token, and entire strings are formed by tracing a path from the root to a specific node.
In this context, we tokenize target entities and build a Trie structure to restrict the output space to the target knowledge base. 
To construct the Trie, you need to create a target_kb.json file formatted as a dictionary like below.

```json
{
  "C565588": ["epidermolysis bullosa with diaphragmatic hernia"], 
  "C567755": ["tooth agenesis selective 6", "sthag6"], 
  "C565584": ["epithelial squamous dysplasia keratinizing desquamative of urinary tract"],
  ...
}
```

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
model = AutoModel.from_pretrained("dmis-lab/ANGEL_pretrained")
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
BATCH_SIZE=16 # 8, 16, 32

bash script/train/train_negative.sh $DATASET $LEARNING_RAT $BATCH_SIZE
```
The script for other datasets is in the train_negative.sh file.

## Evaluation

### Running Inference with the Best Model on Huggingface

To perform inference with our best model hosted on Huggingface, use the following script:
```bash
DATASET=ncbi # bc5cdr, cometa, aap, mm

bash script/inference/inference.sh $DATASET
```

### Output format

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

### Scores

We utilized five popular BioEL benchmark datasets: NCBI-disease (NCBI), BC5CDR, COMETA, AskAPatient (AAP), and MedMentions ST21pv (MM-ST21pv).

<table border="1" cellspacing="0" cellpadding="5">
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="5">Dataset</th>
    </tr>
    <tr>
      <th>NCBI</th>
      <th>BC5CDR</th>
      <th>COMETA</th>
      <th>AAP</th>
      <th>MM-ST21pv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://github.com/dmis-lab/BioSyn">BioSYN</a> (Sung et al., 2020)</td>
      <td>91.1</td>
      <td>-</td>
      <td>71.3</td>
      <td>82.6</td>
      <td>-</td>
    </tr>
    <tr>
      <td><a href="https://github.com/cambridgeltl/sapbert">SapBERT</a> (Liu et al., 2021)</td>
      <td>92.3</td>
      <td>-</td>
      <td>75.1</td>
      <td>89.0</td>
      <td>-</td>
    </tr>
    <tr>
      <td><a href="https://github.com/Yuanhy1997/GenBioEL">GenBioEL</a> (Yuan et al., 2022b)</td>
      <td>91.0</td>
      <td>93.1</td>
      <td>80.9</td>
      <td>89.3</td>
      <td>70.7</td>
    </tr>
    <tr>
      <td>ANGEL (<b>Ours</b>)</td>
      <td><b>92.8</td>
      <td><b>94.5</td>
      <td><b>82.8</td>
      <td><b>90.2</td>
      <td><b>73.3</td>
    </tr>
  </tbody>
</table>

- The scores of GenBioEL were reproduced.
- We excluded the performance of BioSYN and SapBERT on BC5CDR, as they were evaluated separately on the chemical and disease subsets, differing from our settings.

## Model Checkpoints

| Pre-trained | NCBI | BC5CDR | COMETA | MM-ST21pv | 
|:------------:|:-----:|:-----:|:-----:|:-----:| 
| [ANGEL_pretrained](https://huggingface.co/dmis-lab/ANGEL_pretrained)| [ANGEL_ncbi](https://huggingface.co/dmis-lab/ANGEL_ncbi) | [ANGEL_bc5cdr](https://huggingface.co/dmis-lab/ANGEL_bc5cdr) | [ANGEL_cometa](https://huggingface.co/dmis-lab/ANGEL_cometa) |  [ANGEL_mm](https://huggingface.co/dmis-lab/ANGEL_mm) |

- The AskAPatient dataset does not have a predefined split; therefore, we utilized a 10-fold cross-validation method to evaluate our model. As a result, there are 10 model checkpoints corresponding to the AskAPatient dataset. Due to this, we have not open-sourced the checkpoints for this dataset.


## Direct Use

To run the model without any need for a preprocessed dataset, you can use the run_sample.sh script. 

```bash
bash script/inference/run_sample.sh ncbi
```

If you want to modify the sample or input, you can change the script in run_sample.py.

#### Define Your Inputs
- input_sentence: The sentence that includes the entity you want to normalize. Make sure the entity is enclosed within **START** and **END** markers.
- prefix_sentence: A sentence that introduces the context. This should follow the format "**entity** is".
- candidates: A list of candidate entities that the model will use to attempt normalization.

```python
if __name__ == '__main__':
    
    config = get_config()
    
    # Define the input sentence, marking the entity of interest with START and END
    input_sentence = "The r496h mutation of arylsulfatase a does not cause START metachromatic leukodystrophy END"
    # Define the prefix sentence to provide context
    prefix_sentence = "Metachromatic leukodystrophy is"
    # List your candidate entities for normalization
    candidates = ["adrenoleukodystrophy", "thrombosis", "anemia", "huntington disease", "leukodystrophy metachromatic"]
    
    run_sample(config, input_sentence, prefix_sentence, candidates)
```

```terminal
    # Expected Output
    Input  :  The r496h mutation of arylsulfatase a does not cause START metachromatic leukodystrophy END.
    Output :  Metachromatic leukodystrophy is leukodystrophy metachromatic
```

By modifying input_sentence, prefix_sentence, and candidates, you can tailor the examples used by the model to fit your specific needs.


## Citations

If you are interested in our work, please cite:
```bibtex
@article{kim2024learning,
  title={Learning from Negative Samples in Generative Biomedical Entity Linking},
  author={Kim, Chanhwi and Kim, Hyunjae and Park, Sihyeon and Lee, Jiwoo and Sung, Mujeen and Kang, Jaewoo},
  journal={arXiv preprint arXiv:2408.16493},
  year={2024}
}
```

## Acknowledgement
This code includes modifications based on the code of [GenBioEL](https://github.com/Yuanhy1997/GenBioEL). We are grateful to the authors for providing their code/models as open-source software.
