from argparse import ArgumentParser

from transformers import (
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
)

parser = ArgumentParser()
parser.add_argument("pretrained_model_name_or_path")
parser.add_argument("save_directory")
parser.add_argument("--safe-serialize", "-s", action="store_true")

args = parser.parse_args()

for cls in [AutoProcessor, AutoTokenizer, AutoFeatureExtractor]:
    instance = cls.from_pretrained(args.pretrained_model_name_or_path)
    instance.save_pretrained(args.save_directory)
    instance.save_pretrained(args.save_directory + "-fp16")
    instance.save_pretrained(args.save_directory + "-bf16")

model: PreTrainedModel = AutoModelForSpeechSeq2Seq.from_pretrained(
    args.pretrained_model_name_or_path
)
model.save_pretrained(args.save_directory, safe_serialize=args.safe_serialize)

model.half().save_pretrained(
    args.save_directory + "-fp16", safe_serialize=args.safe_serialize
)
model.bfloat16().save_pretrained(
    args.save_directory + "-bf16", safe_serialize=args.safe_serialize
)
