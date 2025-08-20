<div align="center">

# UQ: Assessing Language Models on Unsolved Questions

🌐 [Website](https://uq.stanford.edu/) | 📄 [Paper](https://arxiv.org/abs/TODO) | 🤗 [Dataset](https://hf.co/datasets/uq-project/uq)

</div>

UQ provides resources to assess LLMs on unsolved questions: (1) UQ-Dataset provides curated unsolved questions; (2) UQ-Validators are LLM-based validation strategies to check answer-correctness (3) UQ-Platform is a website to engage with the questions and answers.

 <img src="visuals/uq.png"/>

- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Validation](#validation)
- [Visuals](#visuals)
- [Citation](#citation)

## Dataset

You can load the data from [🤗 uq-project/uq](https://huggingface.co/datasets/uq-project/uq) via:

```python
# pip install -q datasets
from datasets import load_dataset
dataset = load_dataset("uq-project/uq", split="test")
```

## Evaluation

```bash
python gen_answer.py --model_name o3
```


## Validation

Once you have your model predictions, you can use UQ-validators via:

```bash
python validate.py --input_file <your-answer-file.jsonl> --model o3 --strategy sequential --turns 3 --multi_turn_voting majority
```

## Visuals

All figures and some tables are created via [this colab](TODO) equivalent to `visuals/visuals.ipynb`. Some are subsequently edited via the `visuals/visuals.fig` file, which you can load in Figma. The output figures are in `visuals/` in pdf or png format.

## Citation

```bibtex
TODO
```
