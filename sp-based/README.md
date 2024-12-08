# Semantic Parsing-based

## Overview

- Based on [KQA Pro baseline models](https://github.com/shijx12/KQAPro_Baselines)

## Datasets

- [KQA Pro datasets from Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/04ce81541e704a648b03/?dl=1): has `predicate` key error in `kb.json` when training
- [KQA Pro datasets from Hugging Face](https://huggingface.co/datasets/drt/kqa_pro): the problem above is fixed, but the formats of `train.json`, `val.json` and `test.json` don't match the codes

So, to get the necessary files for training, we use the following:

1. **Train, Validation, and Test Data**: 
   - Download the files `train.json`, `val.json`, and `test.json` from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/04ce81541e704a648b03/?dl=1).
   
2. **Knowledge Base**: 
   - Download the file `kb.json` from [Hugging Face](https://huggingface.co/datasets/drt/kqa_pro).
