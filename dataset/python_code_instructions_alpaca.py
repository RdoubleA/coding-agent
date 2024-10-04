from torchtune.datasets import SFTDataset, PackedDataset, instruct_dataset
from torchtune.datasets._alpaca import AlpacaToMessages
from torchtune.modules.tokenizers import ModelTokenizer

def python_code_instructions_alpaca(tokenizer: ModelTokenizer):
    """
    Python code instruct-input-output pairs from iamtarun/python_code_instructions_18k_alpaca templated with Alpaca.
    """
    ds = SFTDataset(
        model_transform=tokenizer,
        source="iamtarun/python_code_instructions_18k_alpaca",
        message_transform=AlpacaToMessages(
            train_on_input=False,
        ),
        split="train",
    )
    return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=False)

def flytech_python_codes(tokenizer: ModelTokenizer):
    """
    Instruction python code pairs from flytech/python-codes-25k
    """
    return instruct_dataset(
        tokenizer=tokenizer,
        source="flytech/python-codes-25k",
        column_map={"input": "instruction", "output": "output"},
        packed=True,
        split="train",
    )
