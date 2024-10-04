from torchtune.datasets import text_completion_dataset
from torchtune.modules.tokenizers import ModelTokenizer

def codeparrot_github(tokenizer: ModelTokenizer):
    """
    Python subset of codeparrot/github-code-clean. This is unstructured text,
    so we use the text completion dataset.
    """
    return text_completion_dataset(
        tokenizer=tokenizer,
        source="codeparrot/github-code-clean",
        name="Python-mit",
        column="code",
        packed=True,
        split_across_pack=True,
        trust_remote_code=True,
    )
