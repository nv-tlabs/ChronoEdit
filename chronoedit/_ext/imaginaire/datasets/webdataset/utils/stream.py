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

import io

# PBSS
import time
from typing import Optional

import boto3
from botocore.exceptions import EndpointConnectionError
from urllib3.exceptions import ProtocolError as URLLib3ProtocolError
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError
from urllib3.exceptions import SSLError as URLLib3SSLError

from chronoedit._ext.imaginaire.utils import log


class RetryingStream:
    def __init__(self, client: boto3.client, bucket: str, key: str, retries: int = 10):  # type: ignore
        r"""Class for loading data in a streaming fashion.
        Args:
            client (boto3.client): Boto3 client
            bucket (str): Bucket where data is stored
            key (str): Key to read
            retries (int): Number of retries
        """
        self.client = client
        self.bucket = bucket
        self.key = key
        self.retries = retries
        self.content_size = self.get_length()
        self.stream, _ = self.get_stream()
        self._amount_read = 0

        self.name = f"{bucket}/{key}"

    def get_length(self) -> int:
        r"""Function for obtaining length of the bytestream"""
        head_obj = self.client.head_object(Bucket=self.bucket, Key=self.key)
        length = int(head_obj["ContentLength"])
        return length

    def get_stream(self, start_range: int = 0, end_range: Optional[int] = None) -> tuple[io.BytesIO, int]:
        r"""Function for getting stream in a range
        Args:
            start_range (int): Start index for stream
            end_range (int): End index for stream
        Returns:
            stream (bytes): Stream of data being read
            content_size (int): Length of the bytestream read
        """
        extra_args = {}
        if start_range != 0 or end_range is not None:
            end_range = "" if end_range is None else end_range - 1  # type: ignore
            # Start and end are inclusive in HTTP, convert to Python convention
            range_param = f"bytes={start_range}-{end_range}"
            extra_args["Range"] = range_param
        response = self.client.get_object(Bucket=self.bucket, Key=self.key, **extra_args)
        content_size = response["ResponseMetadata"]["HTTPHeaders"]["content-length"]
        body = response["Body"]
        stream = body._raw_stream
        return stream, content_size

    def read(self, amt: Optional[int] = None) -> bytes:
        r"""Read function for reading the data stream.
        Args:
            amt (int): Amount of data to read
        Returns:
            chunk (bytes): Bytes read
        """

        chunk = b""
        for cur_retry_idx in range(self.retries):
            try:
                chunk = self.stream.read(amt)
                if len(chunk) == 0 and self._amount_read != self.content_size:
                    raise IOError
                break
            except URLLib3ReadTimeoutError as e:
                log.warning(
                    f"[read] URLLib3ReadTimeoutError: {e} {self.name} retry: {cur_retry_idx} / {self.retries}",
                    rank0_only=False,
                )
            except URLLib3ProtocolError as e:
                log.warning(
                    f"[read] URLLib3ProtocolError: {e} {self.name} retry: {cur_retry_idx} / {self.retries}",
                    rank0_only=False,
                )
            except URLLib3SSLError as e:
                log.warning(
                    f"[read] URLLib3SSLError: {e} {self.name} retry: {cur_retry_idx} / {self.retries}", rank0_only=False
                )
            except IOError as e:
                log.warning(
                    f"[read] Premature end of stream. IOError {e}. Retrying...  {self.name} retry: {cur_retry_idx} / {self.retries}",
                    rank0_only=False,
                )
            time.sleep(1)
            try:
                self.stream, _ = self.get_stream(self._amount_read)
            except EndpointConnectionError as e:
                log.error(
                    f"[get_stream] EndpointConnectionError: {e} {self.name} retry: {cur_retry_idx} / {self.retries}",
                    rank0_only=False,
                )

        if len(chunk) == 0 and self._amount_read != self.content_size:
            log.warning(
                f"After {self.retries} retries, chunk is empty and self._amount_read != self.content_size {self._amount_read} != {self.content_size} {self.name}",
                rank0_only=False,
            )
            raise IOError

        self._amount_read += len(chunk)
        return chunk
