# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from typing import Any, Dict, List, Mapping, Optional

import torch
from omegaconf import DictConfig

from torchtune import config, training, utils
from torchtune.data import Message

from torchtune.generation import sample

from torchtune.modules.transforms import Transform
from torch.utils.data import Dataset
from datasets import load_dataset
import pdb
import json


class APPSBenchmark(Dataset):
    def __init__(
        self,
    ) -> None:
        self.to_messages = GenerationMessages(column_map={"input": "question"})
        self._data = load_dataset("codeparrot/apps", split="test", name="all")
        # Don't use the starter code for simplicity for now
        self._data.filter(lambda x: x["starter_code"] == 0)


    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        messages = self.to_messages(sample)
        return {
            "messages": messages,
            "inputs": json.loads(sample["input_output"])["inputs"],
            "outputs": json.loads(sample["input_output"])["outputs"],
        }





class GenerationMessages(Transform):
    def __init__(self, column_map: Optional[Dict[str, str]] = None) -> None:
        self._column_map = column_map or {"input": "input"}

    def __call__(self, sample: Dict[str, Any]) -> List[Message]:
        messages = []
        messages.append(
            Message(
                role="user",
                content=sample[self._column_map["input"]],
            ),
        )
        # Finally, add an empty assistant message to kick-start generation
        messages.append(Message(role="assistant", content=""))
        return messages


class OfflineEvalRecipe:
    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        self._logger = utils.get_logger(cfg.log_level)
        training.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        """Setup the model and transforms."""
        # Load checkpointer and state_dict
        _checkpointer = config.instantiate(cfg.checkpointer)
        _ckpt_dict = _checkpointer.load_checkpoint()

        # Instantiate model
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg.model)
        model.load_state_dict(_ckpt_dict[training.MODEL_KEY])
        self.model = model
        self._logger.info(f"Model was initialized with precision {self._dtype}.")

        # Instantiate dataset and tokenizer
        self.tokenizer = config.instantiate(cfg.tokenizer)
        self.benchmark = APPSBenchmark()
        self.log_every_n_samples = cfg.log_every_n_samples

    def log_metrics(self, total_time: int, tokens_per_second: float) -> None:
        """Logs the following metrics: total time for inference, tokens/sec,
        bandwidth achieved, and max memory allocated.

        Feel free to modify this function to log additional metrics.
        """
        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(self.model.parameters(), self.model.buffers())
            ]
        )
        self._logger.info(
            f"Time for inference: {total_time:.02f} sec total, {tokens_per_second:.02f} tokens/sec"
        )
        self._logger.info(
            f"Bandwidth achieved: {model_size * tokens_per_second / 1e9:.02f} GB/s"
        )
        self._logger.info(
            f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB"
        )

    @torch.inference_mode()
    def generate(self, cfg: DictConfig):
        """The main entry point for generating tokens from a prompt."""

        total_response_length = cfg.max_new_tokens + self.tokenizer.max_seq_len

        # Setup KV cache
        with self._device:
            self.model.setup_caches(
                batch_size=1,
                dtype=self._dtype,
                encoder_max_seq_len=None,
                decoder_max_seq_len=total_response_length,
            )

        # Pre-allocate causal mask and input_pos
        causal_mask = torch.tril(
            torch.ones(
                size=(total_response_length, total_response_length),
                dtype=torch.bool,
                device=self._device,
            )
        )
        input_pos = torch.arange(total_response_length)

        for i in range(len(self.benchmark)):
            # Apply tokenization
            model_inputs = self.tokenizer(self.benchmark[i], inference=True)
            seq_len = len(model_inputs["tokens"])

            # Collate to batch size of 1 and tensor-ify
            batch = {}
            prompt = torch.tensor(
                model_inputs["tokens"], device=self._device
            ).unsqueeze(0)
            batch["mask"] = causal_mask[None, :seq_len]
            batch["input_pos"] = input_pos[None, :seq_len]
            utils.batch_to_device(batch, self._device)

            # Prefill step
            generated_tokens = []
            t0 = time.perf_counter()
            logits = self.model(prompt, **batch)[:, -1]
            token = sample(logits, temperature=cfg.temperature, top_k=cfg.top_k)
            generated_tokens.append(token.item())

            # Continue generating
            for _ in range(cfg.max_new_tokens):

                # Update position and mask for incremental decoding
                batch["input_pos"] = input_pos[None, seq_len]
                batch["mask"] = causal_mask[None, seq_len, None, :]

                if token.item() in self.tokenizer.stop_tokens:
                    break

                logits = self.model(token, **batch)[:, -1]
                token = sample(logits, temperature=cfg.temperature, top_k=cfg.top_k)
                generated_tokens.append(token.item())
                seq_len += 1

            t = time.perf_counter() - t0

            # Translate tokens back to text
            decoded = self.tokenizer.decode(generated_tokens)

            # Log metrics
            tokens_per_second = len(generated_tokens) / t
            if i % self.log_every_n_samples == 0:
                self._logger.info(f"\n\n{decoded}\n")
                self.log_metrics(total_time=t, tokens_per_second=tokens_per_second)
                self._logger.info(f"Generated {i + 1} of {len(self.benchmark)}.")


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="OfflineEvalRecipe", cfg=cfg)
    recipe = OfflineEvalRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
