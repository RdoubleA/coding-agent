from torchtune.datasets import SFTDataset, PackedDataset
from torchtune.data import InputOutputToMessages
from torchtune.modules.tokenizers import ModelTokenizer

def tiny_codes(tokenizer: ModelTokenizer):
    """
    Python subset of nampdn-ai/tiny-codes. Instruct and code response pairs.
    """
    ds = SFTDataset(
        model_transform=tokenizer,
        source="nampdn-ai/tiny-codes",
        message_transform=InputOutputToMessages(
            column_map={"input": "prompt", "output": "response"},
        ),
        filter_fn=lambda x: x["language"] == "python",
        split="train",
    )
    return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=False)
