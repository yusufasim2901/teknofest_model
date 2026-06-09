#!/usr/bin/env python3
"""
TensorRT INT8 Engine Export — YOLOv8 ONNX → Optimized INT8 Engine.

This script converts a trained YOLOv8 model (exported as ONNX) into a
highly optimized NVIDIA TensorRT engine with INT8 quantization.  INT8
inference typically delivers 2–4× speedup over FP16 and 4–8× over FP32
on NVIDIA GPUs, which is critical for meeting the <100 ms per-frame
latency requirement on edge devices.

Prerequisites:
    - NVIDIA GPU with compute capability ≥ 7.0 (Volta or newer).
    - CUDA Toolkit 12.x installed.
    - TensorRT 10.x Python bindings (``pip install tensorrt``).
    - PyCUDA (``pip install pycuda``).
    - A calibration image dataset (see notes below).

Usage::

    python scripts/export_tensorrt.py \\
        --onnx models/yolov8n.onnx \\
        --output models/yolov8n_int8.engine \\
        --calib-dir data/calibration_images/ \\
        --calib-count 500 \\
        --batch-size 1 \\
        --input-shape 640 640

.. note::
    This script is intended to run on the edge device or a build server
    with the same GPU architecture as the deployment target.  TensorRT
    engines are NOT portable across different GPU architectures.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ──────────────────────────────────────────────────────────────────────
# TensorRT and PyCUDA imports
#
# These libraries require the NVIDIA CUDA Toolkit to be installed on
# the system.  They are intentionally NOT listed in requirements.txt
# because they must match the exact CUDA version on the target device.
#
# Installation:
#   pip install tensorrt pycuda numpy
# ──────────────────────────────────────────────────────────────────────
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 — Initialises the CUDA context
except ImportError as e:
    print(
        f"ERROR: {e}\n"
        "TensorRT and PyCUDA are required for this script.\n"
        "Install them with: pip install tensorrt pycuda\n"
        "Ensure CUDA Toolkit is installed on the system.",
        file=sys.stderr,
    )
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger("trt_export")

# Map Python log levels to TensorRT log levels
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


# ──────────────────────────────────────────────────────────────────────
# Constants — YOLOv8 Preprocessing
# ──────────────────────────────────────────────────────────────────────

# YOLOv8 expects images normalised to [0, 1] with channel order RGB.
# The input tensor shape is (batch, 3, H, W).
_CHANNEL_COUNT: int = 3
_PIXEL_MAX: float = 255.0


# ======================================================================
# INT8 Calibrator
# ======================================================================
#
# WHY CALIBRATION IS NEEDED:
# ──────────────────────────
# INT8 quantization maps the continuous FP32 activation range to just
# 256 discrete integer levels (−128 to +127).  To do this without
# destroying model accuracy, we need to find the optimal *scale factor*
# for each tensor in the network.
#
# The calibrator runs a forward pass on a set of *representative* input
# images (the "calibration dataset") and collects activation histograms.
# These histograms are then used to compute the scale factors that
# minimise quantization error.
#
# CALIBRATION DATASET REQUIREMENTS:
# ──────────────────────────────────
# 1. SIZE:   200–1000 images.  ~500 is a good sweet spot.  More images
#            give diminishing returns; fewer may under-represent the
#            activation distribution.
#
# 2. DOMAIN: Images MUST come from the TARGET DOMAIN — i.e. Turkish
#            roads, TOGG vehicles, various weather/lighting conditions.
#            Using generic ImageNet images will produce poor scale
#            factors and degrade detection accuracy.
#
# 3. SOURCE: Use a random subset of your VALIDATION set (not training).
#            This avoids overfitting the quantization to training data.
#
# 4. FORMAT: Standard image formats (JPEG, PNG).  They will be resized
#            and normalised to match YOLOv8's expected input.
#
# 5. DIVERSITY: Include edge cases — night scenes, rain, glare, partial
#            occlusions, empty roads, crowded intersections — so the
#            calibrator sees the full range of activations.
# ======================================================================


class INT8Calibrator(trt.IInt8EntropyCalibrator2):
    """Custom INT8 calibrator for YOLOv8 using entropy-based quantization.

    TensorRT provides several calibration algorithms.  We use
    ``IInt8EntropyCalibrator2`` because it employs KL divergence
    (Kullback–Leibler) to find the quantization threshold that
    minimises information loss between the original FP32 distribution
    and the quantized INT8 distribution.  This generally yields better
    accuracy than simpler min/max calibration, at a small increase in
    calibration time.

    Parameters:
        calib_dir:    Path to the directory of calibration images.
        calib_count:  Maximum number of images to use.
        batch_size:   Number of images per calibration batch.
        input_shape:  ``(height, width)`` expected by the model.
        cache_file:   Path to save/load the calibration table.
    """

    def __init__(
        self,
        calib_dir: Path,
        calib_count: int,
        batch_size: int,
        input_shape: tuple[int, int],
        cache_file: Path,
    ) -> None:
        super().__init__()

        self._batch_size: int = batch_size
        self._input_shape: tuple[int, int] = input_shape  # (H, W)
        self._cache_file: Path = cache_file

        # ── Discover calibration images ──────────────────────
        # We scan the directory for common image extensions.
        valid_extensions: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        all_images: list[Path] = sorted(
            p
            for p in calib_dir.iterdir()
            if p.suffix.lower() in valid_extensions
        )

        if len(all_images) == 0:
            msg = (
                f"No calibration images found in '{calib_dir}'.  "
                f"Provide at least 200 images from your target domain."
            )
            raise FileNotFoundError(msg)

        # Cap to the requested count
        self._image_paths: list[Path] = all_images[:calib_count]
        self._current_index: int = 0

        logger.info(
            "Calibrator initialised: %d images from '%s'.",
            len(self._image_paths),
            calib_dir,
        )

        # ── Allocate GPU memory for one batch ────────────────
        # The calibrator feeds images to TensorRT one batch at a time.
        # We pre-allocate a contiguous device buffer to avoid repeated
        # GPU memory allocation during calibration.
        #
        # Shape: (batch_size, 3, H, W) × 4 bytes (float32)
        h, w = self._input_shape
        self._batch_nbytes: int = (
            self._batch_size * _CHANNEL_COUNT * h * w * np.dtype(np.float32).itemsize
        )
        self._device_input: int = cuda.mem_alloc(self._batch_nbytes)

    def get_batch_size(self) -> int:
        """Return the calibration batch size."""
        return self._batch_size

    def get_batch(
        self,
        names: list[str],
        p_str: str | None = None,
    ) -> list[int] | None:
        """Load the next batch of calibration images.

        TensorRT calls this method repeatedly until it returns ``None``,
        at which point calibration is complete.

        Each image is preprocessed identically to how YOLOv8 expects
        its inference input:
          1. Read from disk and decode.
          2. Resize to the model's input dimensions (e.g. 640×640).
          3. Convert BGR → RGB (OpenCV loads as BGR).
          4. Normalise pixel values from [0, 255] to [0.0, 1.0].
          5. Transpose from HWC to CHW (channel-first for PyTorch/TRT).

        Returns:
            A list containing the device pointer, or ``None`` when all
            images have been processed.
        """
        if self._current_index >= len(self._image_paths):
            # All images processed — calibration complete.
            return None

        # ── Load and preprocess one batch ────────────────────
        import cv2  # Deferred import (heavy dependency)

        batch_images: list[np.ndarray] = []
        h, w = self._input_shape

        for _ in range(self._batch_size):
            if self._current_index >= len(self._image_paths):
                break

            img_path: Path = self._image_paths[self._current_index]
            self._current_index += 1

            # Read image from disk
            img: np.ndarray | None = cv2.imread(str(img_path))
            if img is None:
                logger.warning("Could not read image: %s — skipping.", img_path)
                continue

            # Resize to model input dimensions using bilinear interpolation.
            # YOLOv8 uses letterbox resizing in practice, but for calibration
            # purposes a simple resize is sufficient and faster.
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

            # BGR → RGB (OpenCV loads as BGR, YOLOv8 expects RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalise [0, 255] → [0.0, 1.0] and convert to float32
            img = img.astype(np.float32) / _PIXEL_MAX

            # HWC → CHW (height, width, channels → channels, height, width)
            # TensorRT and PyTorch both expect channel-first layout.
            img = np.transpose(img, (2, 0, 1))

            batch_images.append(img)

        if not batch_images:
            return None

        # Stack into a single (N, 3, H, W) array and make contiguous
        batch_array: np.ndarray = np.ascontiguousarray(
            np.stack(batch_images, axis=0),
            dtype=np.float32,
        )

        # Copy the batch from host (CPU) to device (GPU)
        cuda.memcpy_htod(self._device_input, batch_array.tobytes())

        logger.debug(
            "Calibrator batch: images %d–%d / %d.",
            self._current_index - len(batch_images),
            self._current_index,
            len(self._image_paths),
        )

        # Return the device pointer(s) — TensorRT will run a forward
        # pass on this data to collect activation statistics.
        return [int(self._device_input)]

    def read_calibration_cache(self) -> bytes | None:
        """Load a previously saved calibration table from disk.

        CALIBRATION CACHE:
        ──────────────────
        The calibration process is expensive (it runs a full forward pass
        on every batch of images).  TensorRT serialises the computed
        scale factors into a "calibration table" that can be saved to
        disk and reused in subsequent engine builds.

        If the cache file exists, calibration is skipped entirely and
        the saved scale factors are used.  This saves ~5–15 minutes on
        a typical 500-image dataset.

        Delete the cache file if you:
          - Change the calibration dataset.
          - Update the ONNX model weights.
          - Switch to a different GPU architecture.
        """
        if self._cache_file.exists():
            logger.info("Loading calibration cache from '%s'.", self._cache_file)
            return self._cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """Save the computed calibration table to disk for reuse.

        This is called automatically by TensorRT after calibration
        completes.  The cache contains the optimal INT8 scale factors
        for every tensor in the network.
        """
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache_file.write_bytes(cache)
        logger.info(
            "Calibration cache saved to '%s' (%d bytes).",
            self._cache_file,
            len(cache),
        )


# ======================================================================
# Engine Builder
# ======================================================================


def build_engine(
    onnx_path: Path,
    output_path: Path,
    calib_dir: Path,
    calib_count: int = 500,
    batch_size: int = 1,
    input_height: int = 640,
    input_width: int = 640,
    workspace_gb: float = 4.0,
) -> None:
    """Parse an ONNX model and build an INT8-optimized TensorRT engine.

    The build process:

    1. Parse the ONNX graph into TensorRT's internal representation.
    2. Configure the builder for INT8 precision with FP16 fallback.
    3. Attach the entropy calibrator for scale-factor computation.
    4. Optimise the network (layer fusion, kernel auto-tuning).
    5. Serialise the engine to a binary ``.engine`` file.

    VRAM EFFICIENCY NOTES:
    ──────────────────────
    - ``workspace_gb`` controls the maximum temporary GPU memory TensorRT
      can use during optimisation.  4 GB is a good default for edge GPUs
      (Jetson Orin).  Increase to 8–16 GB on datacenter GPUs.
    - The engine file itself is typically 5–20 MB for YOLOv8n/s models.
    - INT8 engines use ~4× less VRAM at inference time compared to FP32.

    Args:
        onnx_path:    Path to the input ONNX model.
        output_path:  Path for the output ``.engine`` file.
        calib_dir:    Directory containing calibration images.
        calib_count:  Number of calibration images to use.
        batch_size:   Calibration and inference batch size.
        input_height: Model input height in pixels.
        input_width:  Model input width in pixels.
        workspace_gb: Maximum GPU workspace memory in GB.
    """
    # ── Validate inputs ──────────────────────────────────────
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if not calib_dir.is_dir():
        raise NotADirectoryError(f"Calibration directory not found: {calib_dir}")

    logger.info("=" * 60)
    logger.info("TensorRT INT8 Engine Builder")
    logger.info("=" * 60)
    logger.info("ONNX model:       %s", onnx_path)
    logger.info("Output engine:    %s", output_path)
    logger.info("Calibration dir:  %s (%d images)", calib_dir, calib_count)
    logger.info("Input shape:      (%d, 3, %d, %d)", batch_size, input_height, input_width)
    logger.info("Workspace:        %.1f GB", workspace_gb)

    # ── Step 1: Create TensorRT builder & network ────────────
    #
    # The builder is TensorRT's top-level object for engine creation.
    # The network definition uses EXPLICIT_BATCH mode, which is required
    # for ONNX models (as opposed to the deprecated implicit batch mode).
    builder: trt.Builder = trt.Builder(TRT_LOGGER)
    network_flags: int = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network: trt.INetworkDefinition = builder.create_network(network_flags)

    # ── Step 2: Parse the ONNX model ────────────────────────
    #
    # The ONNX parser converts the platform-agnostic ONNX graph into
    # TensorRT's internal layer representation.  If parsing fails,
    # the errors are logged for debugging.
    parser: trt.OnnxParser = trt.OnnxParser(network, TRT_LOGGER)
    logger.info("Parsing ONNX model…")

    onnx_data: bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_data):
        error_messages: list[str] = []
        for i in range(parser.num_errors):
            error_messages.append(str(parser.get_error(i)))
        combined: str = "\n".join(error_messages)
        raise RuntimeError(f"ONNX parsing failed:\n{combined}")

    logger.info(
        "ONNX parsed successfully: %d layers, %d inputs, %d outputs.",
        network.num_layers,
        network.num_inputs,
        network.num_outputs,
    )

    # ── Step 3: Configure the builder ────────────────────────
    config: trt.IBuilderConfig = builder.create_builder_config()

    # Set maximum GPU workspace memory.
    # TensorRT uses this for temporary buffers during layer fusion and
    # kernel auto-tuning.  More workspace = more optimisation options.
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(workspace_gb * (1 << 30)),  # Convert GB → bytes
    )

    # ── Enable INT8 quantization with FP16 fallback ──────────
    #
    # WHY BOTH INT8 AND FP16?
    # ───────────────────────
    # Not all layers quantize well to INT8 (e.g. certain activation
    # functions, skip connections).  By also enabling FP16, TensorRT
    # can choose the best precision per-layer:
    #   - INT8 for convolutions and matrix multiplications (bulk of compute).
    #   - FP16 for layers where INT8 causes unacceptable accuracy loss.
    # This mixed-precision approach typically retains >99% of FP32 mAP.
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)  # Fallback for non-INT8 layers

    logger.info("INT8 quantization ENABLED (with FP16 fallback).")

    # ── Step 4: Attach the calibrator ────────────────────────
    #
    # The calibrator provides the representative data that TensorRT
    # uses to compute optimal scale factors for INT8 quantization.
    # Without a calibrator, INT8 mode cannot be used.
    cache_file: Path = output_path.with_suffix(".calib_cache")
    calibrator = INT8Calibrator(
        calib_dir=calib_dir,
        calib_count=calib_count,
        batch_size=batch_size,
        input_shape=(input_height, input_width),
        cache_file=cache_file,
    )
    config.int8_calibrator = calibrator
    logger.info("INT8 calibrator attached (entropy-based, KL divergence).")

    # ── Step 5: Set dynamic batch size (optimization profiles) ─
    #
    # Dynamic batching allows the same engine to handle different batch
    # sizes at runtime.  For edge deployment, we typically use batch=1
    # for real-time inference but allow up to batch=8 for throughput
    # testing or batch-processing recorded footage.
    profile: trt.IOptimizationProfile = builder.create_optimization_profile()
    input_tensor: trt.ITensor = network.get_input(0)
    input_name: str = input_tensor.name

    # (min_batch, optimal_batch, max_batch)
    profile.set_shape(
        input_name,
        min=(1, _CHANNEL_COUNT, input_height, input_width),
        opt=(batch_size, _CHANNEL_COUNT, input_height, input_width),
        max=(max(batch_size, 8), _CHANNEL_COUNT, input_height, input_width),
    )
    config.add_optimization_profile(profile)
    logger.info(
        "Optimization profile: min_batch=1, opt_batch=%d, max_batch=%d.",
        batch_size,
        max(batch_size, 8),
    )

    # ── Step 6: Build the engine ─────────────────────────────
    #
    # This is the most time-consuming step (~2–30 minutes depending on
    # model size and GPU).  TensorRT will:
    #   a) Run calibration (forward pass on all calibration batches).
    #   b) Compute optimal INT8 scale factors per tensor.
    #   c) Fuse compatible layers (conv+bn+relu → single kernel).
    #   d) Auto-tune kernel implementations for the target GPU.
    #   e) Serialise the optimised engine.
    logger.info("Building engine… (this may take several minutes)")
    serialized_engine: trt.IHostMemory | None = builder.build_serialized_network(
        network,
        config,
    )

    if serialized_engine is None:
        raise RuntimeError(
            "Engine build failed.  Check TensorRT logs above for details."
        )

    # ── Step 7: Save to disk ─────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(serialized_engine)

    engine_size_mb: float = len(serialized_engine) / (1024 * 1024)
    logger.info("=" * 60)
    logger.info("Engine saved: %s (%.1f MB)", output_path, engine_size_mb)
    logger.info("Calibration cache: %s", cache_file)
    logger.info("=" * 60)
    logger.info(
        "DEPLOYMENT NOTE: This engine is optimised for the current GPU.\n"
        "  It is NOT portable to different GPU architectures.\n"
        "  Rebuild the engine on each target device."
    )


# ======================================================================
# CLI Entry Point
# ======================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export a YOLOv8 ONNX model to an INT8-quantized TensorRT engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/export_tensorrt.py \\\n"
            "      --onnx models/yolov8n.onnx \\\n"
            "      --output models/yolov8n_int8.engine \\\n"
            "      --calib-dir data/calibration_images/ \\\n"
            "      --calib-count 500\n"
            "\n"
            "Calibration Dataset Requirements:\n"
            "  - 200–1000 images from your TARGET DOMAIN (not generic datasets).\n"
            "  - Include diverse conditions: day/night, rain, glare, traffic.\n"
            "  - Use a random subset of your validation set.\n"
            "  - Standard image formats: JPEG, PNG, BMP.\n"
        ),
    )

    parser.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help="Path to the input ONNX model file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the output .engine file.",
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        required=True,
        help="Directory containing calibration images.",
    )
    parser.add_argument(
        "--calib-count",
        type=int,
        default=500,
        help="Maximum number of calibration images to use (default: 500).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Calibration batch size (default: 1).",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("H", "W"),
        help="Model input dimensions in pixels (default: 640 640).",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=4.0,
        help="Maximum GPU workspace memory in GB (default: 4.0).",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point — parse args and build the engine."""
    args: argparse.Namespace = parse_args()

    build_engine(
        onnx_path=args.onnx,
        output_path=args.output,
        calib_dir=args.calib_dir,
        calib_count=args.calib_count,
        batch_size=args.batch_size,
        input_height=args.input_shape[0],
        input_width=args.input_shape[1],
        workspace_gb=args.workspace,
    )


if __name__ == "__main__":
    main()
