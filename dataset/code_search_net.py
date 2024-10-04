from torchtune.datasets import text_completion_dataset
from torchtune.modules.tokenizers import ModelTokenizer

def code_search_net(tokenizer: ModelTokenizer):
    """
    Python subset of code-search-net/code_search_net. This consists of function code
    with their docstrings. Since this is unstructured text, we use the text completion dataset.
    """
    return text_completion_dataset(
        tokenizer=tokenizer,
        source="code-search-net/code_search_net",
        column="whole_func_string",
        packed=True,
        split_across_pack=True,
        filter_fn=lambda x: x["language"] == "python",
        trust_remote_code=True,
        split="train",
    )
