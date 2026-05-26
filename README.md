# CPM-Bench

Open-source code for **CPM-Bench**, a Cost-Per-Meaning benchmark for studying multilingual token efficiency in large language model workflows.

CPM-Bench measures how token usage changes across languages when the underlying semantic workload is held constant. The V1 pipeline builds controlled English prompt-output pairs, translates them into target languages, validates semantic preservation with back-translation similarity, and compares token counts across tokenizers.

## Dataset

The CPM-Bench V1 dataset is available on Zenodo:

```text
10.5281/zenodo.20380913
```

The large dataset files are not stored in this repository. Please download the dataset from Zenodo using the DOI above.

## V1 Overview

- 40 practical LLM task categories
- 75 prompts per category
- 3,000 controlled English prompt-output pairs
- 20 target languages
- 60,000 prompt-language rows
- 120,000 tokenizer-expanded analysis rows
- NLLB-200 distilled 1.3B translation and back-translation
- LaBSE semantic similarity validation
- Qwen3 14B Base tokenizer and NLLB tokenizer comparison

## Repository Structure

```text
.
├── notebooks/
│   └── cpm_bench_v1_pipeline.ipynb
├── src/
│   └── cpm_bench_pipeline.py
├── environment.yml
├── requirements.txt
├── LICENSE
└── README.md
```

## Method Summary

1. Build a controlled English prompt dataset across 40 task categories.
2. Create deterministic English reference outputs.
3. Translate prompts into 20 target languages using NLLB.
4. Back-translate prompts into English for semantic checking.
5. Translate reference outputs into the same target languages.
6. Score semantic preservation using LaBSE.
7. Count input, output, and total tokens with Qwen3 and NLLB tokenizers.
8. Compute Token Efficiency Ratio relative to English.

## Core Metrics

```text
input_TER = target_prompt_tokens / English_prompt_tokens
```

```text
output_TER = target_reference_output_tokens / English_reference_output_tokens
```

```text
total_TER = (target_prompt_tokens + target_output_tokens) / (English_prompt_tokens + English_output_tokens)
```

## Running the Code

Create the environment:

```bash
conda env create -f environment.yml
conda activate cpm-bench
```

Run the Python pipeline:

```bash
python src/cpm_bench_pipeline.py
```

Or run the notebook:

```text
notebooks/cpm_bench_v1_pipeline.ipynb
```

The pipeline was originally run in a Kaggle/offline environment. Before running, update the model paths in the notebook or Python script for:

- Qwen3 14B Base
- NLLB-200 distilled 1.3B
- LaBSE encoder
- LaBSE preprocessor

## Paper

The preprint link will be added here after the manuscript is finalized.

## License

This code is released under the MIT License.
