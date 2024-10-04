from torchtune.datasets import instruct_dataset
from torchtune.modules.tokenizers import ModelTokenizer

def code_bagel(tokenizer: ModelTokenizer):
    """
    Instruction python code pairs from Replete-AI/code_bagel
    """
    return instruct_dataset(
        tokenizer=tokenizer,
        source="Replete-AI/code_bagel",
        packed=True,
        split="train",
    )
