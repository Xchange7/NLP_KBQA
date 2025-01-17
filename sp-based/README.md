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

## Models

### RNN Based Model

```properties
SPARQLParser(
  (word_embeddings): Embedding(48554, 300)
  (word_dropout): Dropout(p=0.3, inplace=False)
  (question_encoder): GRU(
    (encoder): GRU(300, 1024, num_layers=2, batch_first=True, dropout=0.2)
  )
  (sparql_embeddings): Embedding(45693, 300)
  (decoder): GRU(
    (encoder): GRU(300, 1024, num_layers=2, batch_first=True, dropout=0.2)
  )
  (sparql_classifier): Sequential(
    (0): Linear(in_features=1024, out_features=1024, bias=True)
    (1): ReLU()
    (2): Linear(in_features=1024, out_features=45693, bias=True)
  )
  (att_lin): Linear(in_features=1024, out_features=1024, bias=True)
)
```

### BART Integrated Model

```properties
BartForConditionalGeneration(
  (model): BartModel(
    (shared): BartScaledWordEmbedding(50265, 768, padding_idx=1)
    (encoder): BartEncoder(
      (embed_tokens): BartScaledWordEmbedding(50265, 768, padding_idx=1)
      (embed_positions): BartLearnedPositionalEmbedding(1026, 768)
      (layers): ModuleList(
        (0-5): 6 x BartEncoderLayer(
          (self_attn): BartSdpaAttention(
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (activation_fn): GELUActivation()
          (fc1): Linear(in_features=768, out_features=3072, bias=True)
          (fc2): Linear(in_features=3072, out_features=768, bias=True)
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
      (layernorm_embedding): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (decoder): BartDecoder(
      (embed_tokens): BartScaledWordEmbedding(50265, 768, padding_idx=1)
      (embed_positions): BartLearnedPositionalEmbedding(1026, 768)
      (layers): ModuleList(
        (0-5): 6 x BartDecoderLayer(
          (self_attn): BartSdpaAttention(
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (activation_fn): GELUActivation()
          (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (encoder_attn): BartSdpaAttention(
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (encoder_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=768, out_features=3072, bias=True)
          (fc2): Linear(in_features=3072, out_features=768, bias=True)
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
      (layernorm_embedding): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
  )
  (lm_head): Linear(in_features=768, out_features=50265, bias=False)
)
```

