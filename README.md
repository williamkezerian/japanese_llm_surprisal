# Does Model Scale Improve Verb Prediction Across Long Dependencies in Japanese?

**IST700: Understanding LLMs · Syracuse University · William J. Kezerian**

## Overview
This project investigates whether model scale improves verb prediction across long syntactic dependencies in Japanese. Using surprisal as a measure of prediction difficulty, three GPT-NeoX language models of increasing size are evaluated on the UD Japanese GSD corpus. Results suggest that scaling from 340M to 3B parameters does not reduce verb surprisal or dampen the distance effect, contrary to findings in English.

## Repository Structure
```
├── japanese_llm_surprisal.ipynb   # Main analysis notebook
├── data/
│   ├── ja_gsd-ud-train.conllu
│   ├── ja_gsd-ud-dev.conllu
│   └── ja_gsd-ud-test.conllu
├── results/
│   └── surprisal_results.csv
└── executive_summary.pdf
```

## Data
Download the UD Japanese GSD treebank from https://github.com/UniversalDependencies/UD_Japanese-GSD and place the three `.conllu` files in a `data/` folder.

## Models
The three models used are available on HuggingFace. You will need to download them before running the notebook:

| Label  | Model | Params |
|--------|-------|--------|
| Small  | [rinna/japanese-gpt2-medium](https://huggingface.co/rinna/japanese-gpt2-medium) | ~340M |
| Medium | [rinna/japanese-gpt-1b](https://huggingface.co/rinna/japanese-gpt-1b) | ~1.3B |
| Large  | [cyberagent/open-calm-3b](https://huggingface.co/cyberagent/open-calm-3b) | ~3B |

The small and medium models are saved to Google Drive during setup. The large model is loaded directly from the HuggingFace cache each session due to file size.

## Running the Notebook
The notebook is designed to run in Google Colab with a GPU runtime (A100 recommended). Run cells in order — the first cells handle Drive mounting and model setup, and subsequent cells run the full analysis pipeline through to results and visualisation.

## Data
Corpus data is from the [Universal Dependencies Japanese GSD treebank](https://github.com/UniversalDependencies/UD_Japanese-GSD), located in the `data/` folder. It is licensed under Creative Commons and freely available.

## Key Finding
Scaling from 340M to 3B parameters does not reduce verb surprisal or dampen the dependency distance effect in Japanese. All three models return positive controlled distance coefficients, with the large model's coefficient (β=0.193) slightly exceeding the small model's (β=0.145). See the executive summary for full results.
