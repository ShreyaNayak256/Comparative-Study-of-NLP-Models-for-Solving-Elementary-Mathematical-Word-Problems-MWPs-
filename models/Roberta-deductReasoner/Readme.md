# How to import the model

### Link to the model
#Use GPU only
Drive link: https://drive.google.com/file/d/1xHD7S5uhSIyJwXDTy6s1iQIoLj-Xto7C/view?usp=sharing

```python
!pip install requirements.txt

from core.args import TrainerArguments
from deductreasoner.model import DeductReasoner,HfArgumentParser
import torch

parser = HfArgumentParser(TrainerArguments)
args_dict = torch.load("/content/drive/MyDrive/MsAT-main/checkpoints/SVAMP-Deduct/s_0/model.pt") #Write the link to model.pt
trainer_args = parser.parse_dict(args=args_dict)
dec_tokenizer = DecoderTokenizer(trainer_args)
model_args = DeductReasoner.parse_model_args(args_dict)
model_args.num_const = dec_tokenizer.nwords
model = DeductReasoner(model_args)
```

```python
DeductReasoner(
  (roberta): RobertaAdapterModel(
    (shared_parameters): ModuleDict()
    (roberta): RobertaModel(
      (shared_parameters): ModuleDict()
      (invertible_adapters): ModuleDict()
      (embeddings): RobertaEmbeddings(
        (word_embeddings): Embedding(50265, 768, padding_idx=1)
        (position_embeddings): Embedding(514, 768, padding_idx=1)
        (token_type_embeddings): Embedding(1, 768)
        (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (encoder): RobertaEncoder(
        (layer): ModuleList(
          (0-11): 12 x RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(
                  in_features=768, out_features=768, bias=True
                  (loras): ModuleDict()
                )
                (key): Linear(
                  in_features=768, out_features=768, bias=True
                  (loras): ModuleDict()
                )
                (value): Linear(
                  in_features=768, out_features=768, bias=True
                  (loras): ModuleDict()
                )
                (dropout): Dropout(p=0.1, inplace=False)
                (prefix_tuning): PrefixTuningShim(
                  (prefix_gates): ModuleDict()
                  (pool): PrefixTuningPool(
                    (prefix_tunings): ModuleDict()
                  )
                )
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
                (adapters): ModuleDict()
                (adapter_fusion_layer): ModuleDict()
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(
                in_features=768, out_features=3072, bias=True
                (loras): ModuleDict()
              )
              (intermediate_act_fn): GELUActivation()
            )
            (output): RobertaOutput(
              (dense): Linear(
                in_features=3072, out_features=768, bias=True
                (loras): ModuleDict()
              )
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
              (adapters): ModuleDict()
              (adapter_fusion_layer): ModuleDict()
            )
          )
        )
      )
      (pooler): RobertaPooler(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (activation): Tanh()
      )
      (prefix_tuning): PrefixTuningPool(
        (prefix_tunings): ModuleDict()
      )
    )
    (heads): ModuleDict()
  )
  (label_rep2label): Linear(in_features=768, out_features=1, bias=True)
  (linears): ModuleList(
    (0-5): 6 x Sequential(
      (0): Linear(in_features=2304, out_features=768, bias=True)
      (1): ReLU()
      (2): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (3): Dropout(p=0.1, inplace=False)
    )
  )
  (stopper_transformation): Sequential(
    (0): Linear(in_features=768, out_features=768, bias=True)
    (1): ReLU()
    (2): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (3): Dropout(p=0.1, inplace=False)
  )
  (stopper): Linear(in_features=768, out_features=2, bias=True)
  (variable_gru): GRUCell(768, 768)
  (variable_scorer): Sequential(
    (0): Linear(in_features=768, out_features=768, bias=True)
    (1): ReLU()
    (2): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (3): Dropout(p=0.1, inplace=False)
    (4): Linear(in_features=768, out_features=1, bias=True)
  )
)
```
