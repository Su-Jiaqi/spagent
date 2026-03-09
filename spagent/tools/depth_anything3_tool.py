"""
Depth Anything 3 Tool

A single-file SPAgent tool for monocular depth estimation.

Features:
- Mock mode for development / CI
- Real inference mode for Depth Anything V3
- Standard SPAgent tool return format
- Saves depth visualization and/or raw depth array
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class DepthAnything3Tool(Tool):
    """Tool for monocular depth estimation using Depth Anything V3."""

    def __init__(
        self,
        use_mock: bool = True,
        device: str = "cuda",
        encoder: str = "vitl",
        checkpoint_path: Optional[str] = None,
        save_dir: Optional[str] = None,
        input_size: int = 518,
    ):
        """
        Args:
            use_mock: If True, use a fake depth predictor for testing.
            device: Device for inference, e.g. "cuda" or "cpu".
            encoder: Depth Anything V3 encoder variant: "vits", "vitb", or "vitl".
            checkpoint_path: Path to model checkpoint (.pth). Required for real inference.
            save_dir: Directory to save outputs. If None, save next to input image.
            input_size: Input size for the model (used in real inference).
        """
        super().__init__(
            name="depth_anything3_tool",
            description=(
                "Estimate monocular depth from a single RGB image using Depth Anything V3. "
                "Use this tool when you need a depth map, per-pixel relative depth, or geometric "
                "scene understanding from one image."
            ),
        )
        self.use_mock = use_mock
        self.device = device
        self.encoder = encoder
        self.checkpoint_path = checkpoint_path
        self.save_dir = Path(save_dir) if save_dir else None
        self.input_size = input_size

        self.model = None
        self.torch = None

        self._init_model()

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input RGB image."
                },
                "output_format": {
                    "type": "string",
                    "enum": ["png", "npy", "both"],
                    "description": (
                        "Output format for the depth result. "
                        "'png' saves a visualized depth map, "
                        "'npy' saves the raw depth array, "
                        "'both' saves both."
                    ),
                    "default": "both"
                },
                "colormap": {
                    "type": "string",
                    "enum": ["gray", "inferno", "magma", "viridis", "plasma"],
                    "description": "Colormap for depth PNG visualization.",
                    "default": "inferno"
                },
                "normalize": {
                    "type": "boolean",
                    "description": "Whether to normalize depth before visualization.",
                    "default": True
                }
            },
            "required": ["image_path"]
        }

    def _init_model(self) -> None:
        """Initialize mock or real model."""
        if self.use_mock:
            logger.info("DepthAnything3Tool initialized in mock mode.")
            return

        try:
            import torch
            self.torch = torch

            # Adjust this import if your Depth Anything V3 package path differs.
            # Common pattern:
            # from depth_anything_v3.dpt import DepthAnythingV3
            from depth_anything_v3.dpt import DepthAnythingV3

            model_configs = {
                "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
                "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
                "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            }

            if self.encoder not in model_configs:
                raise ValueError(
                    f"Unsupported encoder: {self.encoder}. "
                    f"Expected one of {list(model_configs.keys())}."
                )

            if not self.checkpoint_path:
                raise ValueError(
                    "checkpoint_path is required when use_mock=False."
                )

            checkpoint_path = Path(self.checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

            self.model = DepthAnythingV3(**model_configs[self.encoder])
            state_dict = torch.load(str(checkpoint_path), map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device).eval()

            logger.info(
                "DepthAnything3Tool real model loaded successfully. "
                f"encoder={self.encoder}, checkpoint={self.checkpoint_path}, device={self.device}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize DepthAnything3Tool real model: {e}", exc_info=True)
            raise

    def _mock_predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a simple vertical gradient as fake depth."""
        h, w = image_bgr.shape[:2]
        depth = np.tile(
            np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1),
            (1, w)
        )
        return depth

    def _real_predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Run real Depth Anything V3 inference."""
        if self.model is None or self.torch is None:
            raise RuntimeError("Real model is not initialized.")

        # Depth Anything implementations often expect RGB input.
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Many public DA-V3 repos expose model.infer_image(image_rgb, input_size=...)
        if hasattr(self.model, "infer_image"):
            depth = self.model.infer_image(image_rgb, input_size=self.input_size)
        else:
            raise NotImplementedError(
                "Your Depth Anything V3 model does not expose infer_image(...). "
                "Please adapt _real_predict() to your local implementation."
            )

        if not isinstance(depth, np.ndarray):
            depth = np.array(depth, dtype=np.float32)

        return depth.astype(np.float32)

    @staticmethod
    def _apply_colormap(depth_uint8: np.ndarray, colormap: str) -> np.ndarray:
        cmap_dict = {
            "gray": None,
            "inferno": cv2.COLORMAP_INFERNO,
            "magma": cv2.COLORMAP_MAGMA,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "plasma": cv2.COLORMAP_PLASMA,
        }

        cmap = cmap_dict[colormap]
        if cmap is None:
            return depth_uint8
        return cv2.applyColorMap(depth_uint8, cmap)

    def _save_outputs(
        self,
        image_path: Path,
        depth: np.ndarray,
        output_format: str,
        colormap: str,
        normalize: bool,
    ) -> Dict[str, Optional[str]]:
        """Save depth outputs to disk."""
        save_root = self.save_dir if self.save_dir else image_path.parent
        save_root.mkdir(parents=True, exist_ok=True)

        stem = image_path.stem
        png_path = save_root / f"{stem}_depth.png"
        npy_path = save_root / f"{stem}_depth.npy"

        output_path = None

        if output_format in ["png", "both"]:
            depth_vis = depth.copy()

            if normalize:
                depth_vis = depth_vis - depth_vis.min()
                if depth_vis.max() > 1e-8:
                    depth_vis = depth_vis / depth_vis.max()
                depth_vis = (depth_vis * 255.0).astype(np.uint8)
            else:
                depth_vis = np.clip(depth_vis, 0, 255).astype(np.uint8)

            depth_color = self._apply_colormap(depth_vis, colormap)
            ok = cv2.imwrite(str(png_path), depth_color)
            if not ok:
                raise IOError(f"Failed to save depth PNG to {png_path}")
            output_path = str(png_path)

        if output_format in ["npy", "both"]:
            np.save(str(npy_path), depth)

        return {
            "depth_png_path": str(png_path) if output_format in ["png", "both"] else None,
            "depth_npy_path": str(npy_path) if output_format in ["npy", "both"] else None,
            "output_path": output_path,
        }

    def call(
        self,
        image_path: str,
        output_format: str = "both",
        colormap: str = "inferno",
        normalize: bool = True,
    ) -> Dict[str, Any]:
        """
        Run depth estimation.

        Returns:
            A dict with:
            - success: bool
            - result: nested dict on success
            - output_path: optional main output path
            - summary: short summary on success
            - error: error string on failure
        """
        try:
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }

            if output_format not in {"png", "npy", "both"}:
                return {
                    "success": False,
                    "error": f"Invalid output_format: {output_format}"
                }

            image_bgr = cv2.imread(str(image_path_obj))
            if image_bgr is None:
                return {
                    "success": False,
                    "error": f"Failed to read image: {image_path}"
                }

            if self.use_mock:
                depth = self._mock_predict(image_bgr)
            else:
                depth = self._real_predict(image_bgr)

            if depth.ndim != 2:
                return {
                    "success": False,
                    "error": f"Depth output must be 2D, got shape={depth.shape}"
                }

            outputs = self._save_outputs(
                image_path=image_path_obj,
                depth=depth,
                output_format=output_format,
                colormap=colormap,
                normalize=normalize,
            )

            result = {
                "depth_min": float(depth.min()),
                "depth_max": float(depth.max()),
                "shape": list(depth.shape),
                "depth_png_path": outputs["depth_png_path"],
                "depth_npy_path": outputs["depth_npy_path"],
            }

            return {
                "success": True,
                "result": result,
                "output_path": outputs["output_path"],
                "summary": (
                    f"Depth estimation completed for {image_path_obj.name}. "
                    f"Depth shape: {depth.shape[0]}x{depth.shape[1]}."
                )
            }

        except Exception as e:
            logger.error(f"DepthAnything3Tool error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }