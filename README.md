# Refined Evaluation Metrics Toolkit for TEACH

This toolkit computes refined BLEU and ROUGE scores, using a word alignment self-labeling approach. It includes two main scripts:

- `was.py`: Computes word alignments based on reference and candidate sentences.
- `refined_metrics.py`: Calculates refined BLEU and ROUGE scores using alignment results.

---

## 🛠️ Environment Setup

```bash
# Clone and install Fairseq (custom branch required)
git clone https://github.com/CZWin32768/fairseq
cd fairseq
git checkout czw
pip install --editable .

# Install dependencies
pip install --user sentencepiece==0.1.95
pip install --user transformers==4.3.2
```

> ✅ Make sure you are using Python 3.7+ and have CUDA enabled if running on GPU.

---

## 📁 Required Directory Structure

You must **manually prepare** the following files:

```
alignment-test-sets/
└── your-model-name/
    ├── test.ref   # Reference sentences (one per line)
    └── test.can   # Candidate (generated) sentences (one per line)
```

> `test.ref`: one **reference** sentence per line  
> `test.can`: corresponding **generated translation** from your model per line

---

## 📦 Step-by-Step Usage

### Step 1: Word Alignment Self-labeling

```bash
python was.py \
  --test_model your-model-name \
  --trans_pair ref-can \
  --test_set_dir alignment-test-sets \
  --alignments_output_dir output \
  --model_type chinese \
  --model_name_or_path AnchiBERT
```

- Input: `alignment-test-sets/your-model-name/test.ref`, `test.can`
- Output: `output/alignments_your-model-name.jsonl`

> 💡 Replace `AnchiBERT` with your own pretrained model folder, which must contain:
> `pytorch_model.bin`, `config.json`, `vocab.txt`.

---

### Step 2: Calculate Refined BLEU and ROUGE Scores

```bash
python refined_metrics.py --test_model your-model-name
```

- Input: `output/alignments_your-model-name.jsonl`
- Output: Refined scores printed to console:

```
Improved ROUGE score for your-model-name: xxx
Improved BLEU score for your-model-name: xxx
```

---

## Licenses

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

This work is licensed under a [MIT License](https://lbesson.mit-license.org/).

---

## 📧 Citation

Please cite our paper if you use our toolkit.

```bibtex
@inproceedings{wei-etal-2025-teach,
    title = "{TEACH}: A Contrastive Knowledge Adaptive Distillation Framework for Classical {C}hinese Understanding",
    author = "Wei, Yuting  and
      Meng, Qi  and
      Xu, Yuanxing  and
      Wu, Bin",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.178/",
    pages = "3537--3550",
    ISBN = "979-8-89176-251-0",
}
```

---

## 🙏 Acknowledgements

The `was.py` script was based on the structure of [XLM-Align Word Aligner](https://github.com/CZWin32768/XLM-Align/tree/master/word_aligner). We are grateful for this work and would like to acknowledge their significant contributions to the community.
