# GeoThought: A Dataset for Enhancing Mathematical Geometry Reasoning in Vision-Language Models

This repository contains the code and data for the paper titled "GeoThought: A Dataset for Enhancing Mathematical Geometry Reasoning in Vision-Language Models".

## Resources
- **Paper**: [Paper](https://arxiv.org/user/)
- **Datasets**: 
  - [GeoThought-6k](https://huggingface.co/datasets/xinlingdedeng/GeoThought-6k)
  - [Geo-Thought-Augmented-10K](https://huggingface.co/datasets/xinlingdedeng/Geo-Thought)
- **Models**: [InternVL3-8B-10834](https://huggingface.co/xinlingdedeng/InternVL3-8B-10834)

## Installation

```bash
cd GeoThought
conda create -n geothought python=3.10 -y
conda activate geothought
pip install -e .

## Enable Docker

```bash
pip install deepspeed

## Data Preparation

Download our dataset and place the data under playground/data. Here is the data structure:

playground/data/
├── images/
│   ├── geo3k/
│   ├── geoqa_plus/
│   ├── test/
├── alignment.json
├── qa_tuning.json
├── test_question.jsonl
├── test_answers.jsonl

test_question.jsonl and test_answers.jsonl correspond to the test set of GeoQA.

## Training

This stage enables the model to better interpret the content of geometric figures.

## Evaluation

```bash
bash scripts/eval_multi.sh \
    path-to-model \
    playground/data/test_questions.jsonl \
    path-to-output \
    path-to-image-folder \
    num_gpus \
    temperature

## Acknowledgement

This project builds upon previous work in multimodal learning and geometric reasoning. We thank the research community for their foundational contributions.

## Citation
If you find our code and dataset helpful to your research, please consider citing our work:








