Link to the model:


# How to import the model:

from core.args import TrainerArguments
from deductreasoner.model import DeductReasoner
parser = HfArgumentParser(TrainerArguments)

args_dict = torch.load("/content/drive/MyDrive/MsAT-main/checkpoints/SVAMP-Deduct/s_0/model.pt") #Replace by model.pt link
trainer_args = parser.parse_dict(args=args_dict)
dec_tokenizer = DecoderTokenizer(trainer_args)
model_args = DeductReasoner.parse_model_args(args_dict)
model_args.num_const = dec_tokenizer.nwords
model = DeductReasoner(model_args)
