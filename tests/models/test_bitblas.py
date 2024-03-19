"""Compare the outputs of a GPTQ model to a BitBLAS model.

Note: GPTQ and BitBLAS do not have bitwise correctness.
As a result, in this test, we just confirm that the top selected tokens of the
BitBLAS/GPTQ models are in the top 3 selections of each other.

Note: BitBLAS internally uses locks to synchronize the threads. This can
result in very slight nondeterminism for BitBLAS. As a result, we re-run the test
up to 3 times to see if we pass.

Run `pytest tests/models/test_bitblas.py --forked`.
"""

import pytest
import torch
from dataclasses import dataclass
from vllm.model_executor.layers.quantization import (
    _QUANTIZATION_CONFIG_REGISTRY)

capability = torch.cuda.get_device_capability()
capability = capability[0] * 10 + capability[1]
bitblas_not_supported = (
    capability < _QUANTIZATION_CONFIG_REGISTRY["bitblas"].get_min_capability())


@dataclass
class ModelPair:
    model_bitblas: str
    model_gptq: str


model_pairs = [
    ModelPair(model_bitblas="/home/t-leiwang/mlc_workspace/AutoGPTQ/dist/opt-125m-4bit-128g-bitblas",
              model_gptq="/home/t-leiwang/mlc_workspace/AutoGPTQ/dist/opt-125m-4bit-128-gptq")
]


@pytest.mark.flaky(reruns=2)
@pytest.mark.skipif(bitblas_not_supported,
                    reason="BitBLAS is not supported on this GPU type.")
@pytest.mark.parametrize("model_pair", model_pairs)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [3])
def test_models(
    vllm_runner,
    example_prompts,
    model_pair: ModelPair,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    bitblas_model = vllm_runner(model_pair.model_bitblas, dtype=dtype)
    bitblas_outputs = bitblas_model.generate_greedy_logprobs(
        example_prompts, max_tokens, num_logprobs)

    # Note: not sure why, but deleting just the model on Ada Lovelace
    #   does not free the GPU memory. On Ampere, deleting the just model
    #   frees the memory.
    del bitblas_model.model.llm_engine.driver_worker
    del bitblas_model

    gptq_model = vllm_runner(model_pair.model_gptq, dtype=dtype)
    gptq_outputs = gptq_model.generate_greedy_logprobs(example_prompts,
                                                       max_tokens,
                                                       num_logprobs)

    # Note: not sure why, but deleting just the model on Ada Lovelace
    #   does not free the GPU memory. On Ampere, deleting the just model
    #   frees the memory.
    del gptq_model.model.llm_engine.driver_worker
    del gptq_model

    # loop through the prompts
    for prompt_idx in range(len(example_prompts)):
        gptq_output_ids, gptq_output_str, gptq_logprobs = gptq_outputs[
            prompt_idx]
        bitblas_output_ids, bitblas_output_str, bitblas_logprobs = bitblas_outputs[
            prompt_idx]

        for idx, (gptq_output_id, bitblas_output_id) in enumerate(
                zip(gptq_output_ids, bitblas_output_ids)):
            # If sequence is not an exact match,
            if bitblas_output_id != gptq_output_id:
                # Each predicted token must be in top 5 of the other's
                assert gptq_output_id in bitblas_logprobs[idx], (
                    f"Test{prompt_idx}:\nGPTQ:\t{gptq_output_str!r}\n"
                    f"BitBLAS:\t{bitblas_output_str!r}")
                assert bitblas_output_id in gptq_logprobs[idx], (
                    f"Test{prompt_idx}:\nGPTQ:\t{gptq_output_str!r}\n"
                    f"BitBLAS:\t{bitblas_output_str!r}")

                # Break out since sequences will now diverge.
                break

import pytest

if __name__ == "__main__":
    pytest.main(["-s", "tests/models/test_bitblas.py"])