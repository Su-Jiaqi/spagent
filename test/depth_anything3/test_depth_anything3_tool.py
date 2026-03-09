from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from spagent.tools.depth_anything3_tool import DepthAnything3Tool


def test_depth_anything3_tool_mock_on_bus_png():
    image_path = Path("/data/sjq/spagent/test/depth_anything3/assets/bus.png")
    output_dir = Path("/data/sjq/spagent/test/depth_anything3/outputs")

    assert image_path.exists(), f"Test image not found: {image_path}"

    tool = DepthAnything3Tool(
        use_mock=True,
        save_dir=str(output_dir),
    )

    result = tool.call(
        image_path=str(image_path),
        output_format="both",
        colormap="inferno",
        normalize=True,
    )

    assert result["success"] is True
    assert "result" in result

    depth_png_path = result["result"]["depth_png_path"]
    depth_npy_path = result["result"]["depth_npy_path"]

    assert depth_png_path is not None
    assert depth_npy_path is not None
    assert Path(depth_png_path).exists(), f"Depth PNG not found: {depth_png_path}"
    assert Path(depth_npy_path).exists(), f"Depth NPY not found: {depth_npy_path}"