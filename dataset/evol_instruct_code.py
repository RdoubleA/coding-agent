from torchtune.datasets import instruct_dataset, SFTDataset, PackedDataset
from torchtune.data import InputOutputToMessages
from torchtune.modules.tokenizers import ModelTokenizer

def code_feedback_filtered_instruction(tokenizer: ModelTokenizer):
    """
    Instruction python code pairs from m-a-p/CodeFeedback-Filtered-Instruction
    """
    ds = SFTDataset(
        model_transform=tokenizer,
        source="m-a-p/CodeFeedback-Filtered-Instruction",
        message_transform=InputOutputToMessages(
            column_map={"input": "query", "output": "answer"},
        ),
        filter_fn=lambda x: x["lang"] == "python",
        split="train",
    )
    return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=False)

def evol_code_alpaca(tokenizer: ModelTokenizer):
    """
    Instruction python code pairs from theblackcat102/evol-codealpaca-v1
    """
    return instruct_dataset(
        tokenizer=tokenizer,
        source="theblackcat102/evol-codealpaca-v1",
        column_map={"input": "instruction", "output": "output"},
        packed=True,
        split="train",
    )
