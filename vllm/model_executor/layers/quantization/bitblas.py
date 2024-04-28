from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter
import ctypes
from functools import reduce
import operator

from vllm._C import ops
from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

try:
    import bitblas
except ImportError as e:
    bitblas_import_exception = e
    raise ValueError(
        f"Trying to use the bitblas backend, but could not import dependencies with the following error: {bitblas_import_exception}"
    )

from bitblas.ops.matmul_dequantize import (
    MatmulWeightOnlyDequantizeConfig,
    MatmulWeightOnlyDequantize,
)
import bitblas
from bitblas.utils import auto_detect_nvidia_target
from bitblas.cache import global_operator_cache

BITBLAS_TARGET = auto_detect_nvidia_target()
BITBLAS_DATABASE_PATH = ".bitblas_database"
global_operator_cache.load_from_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)


class BitBLASConfig(QuantizationConfig):
    """Config class for BitBLAS."""

    TORCH_DTYPE = torch.float16
    STORAGE_DTYPE = "int8"  # assume int8 storage
    TORCH_STORAGE_DTYPE = getattr(torch, STORAGE_DTYPE)
    ZEROS_TYPE = "quantized"  # "original" or "rescale" or "quantized"

    def __init__(
        self,
        nbits: int = 4,
        group_size: int = -1,
        fast_type_conversion: bool = True,
        weight_propagation: bool = False,
    ) -> None:
        # Group size for the quantization.
        self.group_size = group_size

        storage_dtype = self.STORAGE_DTYPE
        storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))

        self.storage_dtype = storage_dtype
        self.storage_torch_dtype = self.TORCH_STORAGE_DTYPE
        # 4 Bits packed into 32 bit datatype.
        self.pack_factor = storage_nbit // nbits

        self.nbits = nbits

        # whether weight propagation is applied
        self.fast_type_conversion = fast_type_conversion
        self.weight_propagation = weight_propagation

        # Zeros type for the quantized weights.
        self.zeros_type = self.ZEROS_TYPE

    def __repr__(self) -> str:
        return f"BitBLASConfig(group_size={self.group_size}"

    @classmethod
    def get_name(cls) -> str:
        return "bitblas"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BitBLASConfig":
        nbits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        return cls(nbits, group_size)

    def get_linear_method(self) -> "BitBLASLinearMethod":
        return BitBLASLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []


class BitBLASLinearMethod(LinearMethodBase):
    """Implements a linear method using BitBLAS for efficient computation with quantized weights.

    This class provides methods to create quantized weights and apply them to inputs using the BitBLAS
    library, which is designed for high-performance, bit-level operations on GPUs.

    Attributes:
        quant_config (BitBLASConfig): Configuration for BitBLAS quantization.
    """

    OPT_FEATURES = [1, 16, 32, 64, 128, 256, 512]
    BITBLAS_DTYPES = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.half: "float16",
        torch.int8: "int8",
    }

    def __init__(self, quant_config: BitBLASConfig):
        """Initializes the BitBLASLinearMethod with the given quantization configuration.

        Args:
            quant_config: Configuration for quantizing weights and operations.
        """
        self.quant_config = quant_config
        self.bitblas_matmul = (
            None  # Placeholder for the BitBLAS matrix multiplication operator.
        )
        self.opt_features = self.OPT_FEATURES  # Optimized features for tuning.
        self.enable_tuning = True  # Flag to enable tuning for the BitBLAS operator.
        self.target = BITBLAS_TARGET

    def create_weights(
        self,
        input_size_per_partition: int,
        output_size_per_partition: int,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        """Creates quantized weights for use in linear operations.

        The function initializes and returns a dictionary containing quantized weights, scales, and zeros
        for performing quantized matrix multiplication operations.

        Args:
            input_size_per_partition: The size of the input partition.
            output_size_per_partition: The size of the output partition.
            input_size: The total size of the input (unused).
            output_size: The total size of the output (unused).
            params_dtype: The data type of the parameters (expected to be torch.float16).

        Returns:
            A dictionary containing the quantized weights ('qweight'), scales ('scales'), and zeros ('zeros').

        Raises:
            ValueError: If `params_dtype` is not `torch.float16` or if the input size per partition
                        is not divisible by the group size in `quant_config`.
        """
        del input_size, output_size  # Unused arguments.
        if params_dtype != torch.float16:
            raise ValueError(
                f"Parameter data type must be torch.float16, but got {params_dtype}"
            )

        if (
            self.quant_config.group_size != -1
            and input_size_per_partition % self.quant_config.group_size != 0
        ):
            raise ValueError(
                f"Input size per partition ({input_size_per_partition}) must be divisible by "
                f"group size ({self.quant_config.group_size})."
            )

        # Initialize or retrieve the BitBLAS matrix multiplication operator.
        self._configure_bitblas_matmul(
            input_size_per_partition,
            output_size_per_partition,
            params_dtype,
            self.enable_tuning,
            self.quant_config.fast_type_conversion,
            bias=False,
            propagate_b=self.quant_config.weight_propagation,
            layout="nt",
            bits=self.quant_config.nbits,
        )
        # Initialize quantized weights with dimensions optimized for BitBLAS operations.

        qweight = Parameter(
            torch.empty(
                self.bitblas_matmul.retrieve_weight_shape(),
                device="cuda",
                dtype=self.quant_config.storage_torch_dtype,
            ),
            requires_grad=False,
        )
        # Attributes to help with unpacking and applying the weights later.
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "bitblas_tile_size": (
                    self.bitblas_matmul.retrieve_weight_shape()[-2]
                    if self.quant_config.weight_propagation
                    else None
                ),
                "pack_factor": self.quant_config.pack_factor,
                "weight_propagation": self.quant_config.weight_propagation,
            },
        )

        # Compute the number of input groups for channel-wise quantization.
        input_groups = (
            1
            if self.quant_config.group_size == -1
            else input_size_per_partition // self.quant_config.group_size
        )

        # Initialize scales and zeros for the quantized weights.
        scales = Parameter(
            torch.empty(
                output_size_per_partition,
                input_groups,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales, {"input_dim": None if input_groups == 1 else 1, "output_dim": 0})
        if self.quant_config.zeros_type == "quantized":
            zeros = Parameter(
                torch.empty(
                    input_groups,
                    output_size_per_partition // self.quant_config.pack_factor,
                    device="cuda",
                    dtype=self.quant_config.storage_torch_dtype,
                ),
                requires_grad=False,
            )
            # Set attributes to indicate how scales and zeros are applied.

            set_weight_attrs(
                zeros,
                {
                    "input_dim": None if input_groups == 1 else 0,
                    "output_dim": 1,
                    "packed_dim": 1,
                    "pack_factor": self.quant_config.pack_factor,
                },
            )
        else:
            zeros = Parameter(
                torch.empty(output_size_per_partition, input_groups,
                            device="cuda",
                            dtype=params_dtype),
                requires_grad=False,
            )
            # Set attributes to indicate how scales and zeros are applied.
            set_weight_attrs(scales, {"input_dim": None if input_groups == 1 else 1, "output_dim": 0})

        return {"qweight": qweight, "scales": scales, "zeros": zeros}

    def _configure_bitblas_matmul(
        self,
        infeatures,
        outfeatures,
        params_dtype,
        enable_tuning,
        fast_decoding,
        bias,
        propagate_b,
        layout,
        bits,
    ):
        # Assuming MatmulWeightOnlyDequantizeConfig and MatmulWeightOnlyDequantize are defined elsewhere
        bitblas_dtype = self.BITBLAS_DTYPES[params_dtype]
        matmul_config = MatmulWeightOnlyDequantizeConfig(
            M=self.opt_features,
            N=outfeatures,
            K=infeatures,
            in_dtype=bitblas_dtype,
            out_dtype=bitblas_dtype,
            accum_dtype="int32" if bitblas_dtype == "int8" else bitblas_dtype,
            bit=bits,
            storage_dtype=self.quant_config.STORAGE_DTYPE,
            source_format="uint",
            with_scaling=True,
            with_zeros=True,
            group_size=self.quant_config.group_size,
            fast_decoding=fast_decoding,
            with_bias=bias,
            propagate_a=False,
            propagate_b=propagate_b,
            layout=layout,
            zeros_type=self.quant_config.zeros_type,
        )
        self.bitblas_matmul = self._get_or_create_bitblas_operator(
            matmul_config, enable_tuning
        )

    def _get_or_create_bitblas_operator(self, config, enable_tuning):
        bitblas_matmul = global_operator_cache.get(config)
        if bitblas_matmul is None:
            bitblas_matmul = MatmulWeightOnlyDequantize(config, target=self.target)
            if enable_tuning:
                bitblas_matmul.hardware_aware_finetune(topk=20)
                global_operator_cache.add(config, bitblas_matmul)
                global_operator_cache.save_into_database(
                    BITBLAS_DATABASE_PATH, BITBLAS_TARGET
                )
                print(
                    "BitBLAS Tuning done, appended operator to global_operator_cache."
                )
            else:
                print("BitBLAS Operator created.")
        else:
            print("BitBLAS Operator found in global_operator_cache.")
        return bitblas_matmul

    def apply_weights(
        self,
        weights: Dict[str, Any],
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies the quantized weights to the input tensor and adds bias if provided.

        This method performs a quantized matrix multiplication operation on the input tensor using the
        pre-initialized BitBLAS operator and the quantized weights, scales, and zeros.

        Args:
            weights: A dictionary containing the quantized weights, scales, and zeros.
            x: The input tensor to which the weights will be applied.
            bias: An optional tensor to be added to the output after the weights have been applied.

        Returns:
            The output tensor after applying the quantized weights (and bias if provided).
        """
        # Reshape the input and prepare the output tensor.
        # x_2d = x.view(-1, x.shape[-1])
        # output_2d = torch.empty(
        #     x_2d.shape[:-1] + (weights["scales"].shape[0],),
        #     dtype=x_2d.dtype,
        #     device=x_2d.device,
        # )
        # # Apply the BitBLAS matrix multiplication.
        # self.bitblas_matmul(
        #     x_2d, weights["qweight"], weights["scales"], weights["zeros"], output_2d
        # )

        # # Reshape the output and apply bias if provided.
        # output = output_2d.view(x.shape[:-1] + (output_2d.shape[1],))
        # if bias is not None:
        #     output += bias

        if x.dtype != torch.float16:
            x = x.half()
        output = torch.empty(
            x.shape[:-1] + (weights["scales"].shape[0],), dtype=x.dtype, device=x.device
        )
        x_void = ctypes.c_void_p(x.data_ptr())
        qweight_void = ctypes.c_void_p(weights["qweight"].data_ptr())
        scales_void = ctypes.c_void_p(weights["scales"].data_ptr())
        zeros_void = ctypes.c_void_p(weights["zeros"].data_ptr())
        output_void = ctypes.c_void_p(output.data_ptr())

        # m is the product of the last n - 1 dimensions of A
        m = ctypes.c_int32(reduce(operator.mul, x.shape[:-1], 1))
        self.bitblas_matmul.lib.call(
            x_void , qweight_void, scales_void, zeros_void, output_void, m
        )
        if bias is not None:
            output += bias

        return output

__all__ = ["BitBLASConfig"]
