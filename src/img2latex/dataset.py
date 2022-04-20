import albumentations as alb
from albumentations.pytorch import ToTensorV2
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


test_transform = alb.Compose(
    [
        alb.ToGray(always_apply=True),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        # alb.Sharpen()
        ToTensorV2(),
    ]
)


def generate_tokenizer(equations, output, vocab_size):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        special_tokens=["[PAD]", "[BOS]", "[EOS]"],
        vocab_size=vocab_size,
        show_progress=True,
    )
    tokenizer.train(trainer, equations)
    tokenizer.save(path=output, pretty=False)
