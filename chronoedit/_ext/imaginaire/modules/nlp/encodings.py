# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Union

import torch
from einops import repeat

from chronoedit._ext.imaginaire.modules.nlp.t5xxl.t5encoder import T5Encoder


class TextEncoder:
    """Text encoder base class."""

    name: str

    def update_encoding_params(self, *args: Any, **kwargs: Any) -> None:
        """Updates encoding params of your text encoder.

        Args:
            *args: Whatever you need to update, e.g. max_len of encoding.
            **kwargs: Keyword arguments are also possible.

        """
        raise NotImplementedError

    def __call__(self, input_text: Union[str, list[str]], **kwargs: Any) -> Any:
        """Performs text encoding.

        Args:
            input_text: A string or a list of strings to encode.
            **kwargs: Keyword arguments are also possible.

        Return:
            Your model's output.
        """
        raise NotImplementedError


class T5TextEncoder(TextEncoder):
    """Get T5 encoder for obtaining text encodings.

    Args:
        t5_tokens_num (int): Max sequence length.
        device (str): Device to load the model on to.
        max_len (int): Max length of text encoded tokens to be returned.
        dim (int): Dimension of each text encoded token.
    """

    def __init__(
        self,
        t5_tokens_num: int,
        device: str = "cuda",
        max_len: int = 113,
        dim: int = 1024,
        return_offsets_mapping: bool = False,
    ):
        super().__init__()
        self.name = "T5XXL"
        self.model = T5Encoder(max_seq_len=t5_tokens_num, device=device, return_offsets_mapping=return_offsets_mapping)
        self.model = self.model.eval()
        self.update_encoding_params(max_len=max_len, dim=dim, return_offsets_mapping=return_offsets_mapping)

    def update_encoding_params(self, max_len: int = 113, dim: int = 1024, return_offsets_mapping: bool = False):
        self.max_len = max_len
        self.dim = dim
        self.return_offsets_mapping = return_offsets_mapping
        if self.return_offsets_mapping:
            assert self.model.return_offsets_mapping, (
                "T5TextEncoder needs to be initialized with return_offsets_mapping=True. "
                + "Cannot turn it on after initialization."
            )

    @torch.no_grad()
    def __call__(self, input_text: Union[str, list[str]]):
        if isinstance(input_text, str):
            input_text = [input_text]
        if self.model is None:
            out = (torch.zeros(1, self.max_len, self.dim), torch.zeros(1, self.max_len), None)
        else:
            self.model.half()
            out = self.model.encode(input_text)

        output = {
            "t5_text_embeddings": out[0].float(),
            "t5_text_mask": out[1],
            "mask": out[1],
        }

        if self.return_offsets_mapping:
            output["t5_offsets_mapping"] = out[2]

        return output


class CLIPTextEncoder(TextEncoder):
    """Get CLIP encoder for obtaining text encodings.

    Args:
        device (str): Device to load the model on to.
        max_len (int): Max length of text encoded tokens to be returned.
        dim (int): Dimension of each text encoded token.
    """

    def __init__(
        self,
        device: str = "cuda",
        max_len: int = 77,
        attr_max_len: int = 64,
        dim: int = 1024,
        return_offsets_mapping: bool = False,
    ):
        super().__init__()
        self.name = "CLIP"
        self.model = None
        self.update_encoding_params(
            max_len=max_len, dim=dim, attr_max_len=attr_max_len, return_offsets_mapping=return_offsets_mapping
        )

    def update_encoding_params(
        self, max_len: int = 77, attr_max_len: int = 64, dim: int = 1024, return_offsets_mapping: bool = False
    ):
        self.max_len = max_len
        self.attr_max_len = attr_max_len
        self.dim = dim
        self.return_offsets_mapping = return_offsets_mapping

    @torch.no_grad()
    def __call__(self, input_text: Union[str, list[str]]):
        raise NotImplementedError
        if isinstance(input_text, str):
            input_text = [input_text]
        if self.model is None:
            out = (torch.zeros(1, self.max_len, self.dim), torch.zeros(1, self.max_len), torch.zeros(1, self.dim), None)
        else:
            self.model.half()
            out = self.model(input_text)

        output = {"clip_text_embeddings": out[0].float(), "clip_text_mask": out[1], "mask": out[1]}

        if self.return_offsets_mapping:
            output["clip_offsets_mapping"] = out[2]

        return output


def repeat_embedding(embedding, batch_size):
    return {k: repeat(v, "b ... -> (b n) ...", n=batch_size) for k, v in embedding.items()}


def get_text_embeddings(
    text_input: Union[str, list[str]],
    text_encoders: list[TextEncoder],
    batch_size: int,
    negative: bool = False,
    override_masks_with_1s=True,
):
    """Gets text embeddings of input text.
    Args:
        text_input (str or list of strs): Input text to be encoded.
        text_encoders (list[TextEncoder]): list of TextEncoders to be applied to each input text str.
        attr_encoder (CLIPTextEncoder or None): encoder for attributes.
        batch_size (int): Batch size for replication of encodings.
        negative (bool): True if negative prompt.
        override_masks_with_1s (bool): True if you want all text encoding masks to be filled with 1s.
        This is necessary for some edify image models.
    """
    error_status = ""
    embeddings = {}

    # Prepare suffix if negative prompt.
    if negative:
        key_suffix = "_neg"
    else:
        key_suffix = ""

    # Encode the text.
    for encoder in text_encoders:
        output = encoder(text_input)

        # When the text mask is all 1's, the number of tokens in the text prompt
        # is >= the max tokens we can handle. We send a error message in this case,
        if (output.pop("mask") == 0).sum().item() == 0:
            if negative:
                error_status = error_status + f"{encoder.name}: Negative prompt is too long"
            else:
                error_status = error_status + f"{encoder.name}: Text prompt is too long"

        for k, v in output.items():
            if "mask" in k and override_masks_with_1s:
                v.fill_(1)
            embeddings[k + key_suffix] = v

    # Return outputs.
    embeddings = repeat_embedding(embeddings, batch_size)

    return embeddings, error_status
