# How to import the model

### Link to the model
#Use GPU only
Drive link: https://drive.google.com/file/d/1xHD7S5uhSIyJwXDTy6s1iQIoLj-Xto7C/view?usp=sharing

```python
!pip install requirements.txt

from core.args import TrainerArguments
from deductreasoner.model import DeductReasoner,
import torch

parser = HfArgumentParser(TrainerArguments)
args_dict = torch.load("/content/drive/MyDrive/MsAT-main/checkpoints/SVAMP-Deduct/s_0/model.pt") #Write the link to model.pt
trainer_args = parser.parse_dict(args=args_dict)
dec_tokenizer = DecoderTokenizer(trainer_args)
model_args = DeductReasoner.parse_model_args(args_dict)
model_args.num_const = dec_tokenizer.nwords
model = DeductReasoner(model_args)
```
