#!/usr/bin/env python3
"""
Download and convert the strongest supported local face model for Embeddingz.

The script tries the best released checkpoint first:
1. CVLFace AdaFace ViT-Base KPRPE WebFace12M
2. CVLFace AdaFace IR101 WebFace12M

It saves the converted Core ML package into `Embeddings/` and keeps the raw
downloaded artifacts under `Models/`, which is gitignored.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "Models"
APP_MODELS_DIR = ROOT / "Embeddings"

DEPENDENCIES = [
    ("huggingface_hub", "huggingface_hub"),
    ("transformers", "transformers"),
    ("safetensors", "safetensors"),
    ("timm", "timm"),
    ("omegaconf", "omegaconf"),
    ("easydict", "easydict"),
]

MODEL_CANDIDATES = [
    {
        "repo_id": "minchul/cvlface_adaface_vit_base_kprpe_webface12m",
        "package_name": "CVLFaceViTKPRPE",
        "description": "CVLFace AdaFace ViT-Base KPRPE WebFace12M",
    },
    {
        "repo_id": "minchul/cvlface_adaface_ir101_webface12m",
        "package_name": "CVLFaceIR101",
        "description": "CVLFace AdaFace IR101 WebFace12M",
    },
]

SAFE_RPE_INIT = """from .KPRPE import kprpe_shared
import warnings

try:
    from .rpe_ops.rpe_index import RPEIndexFunction
except Exception:
    RPEIndexFunction = None
    RED_STR = "\\033[91m{}\\033[00m"
    warnings.warn(RED_STR.format("\\n[WARNING] Falling back to the pure Python KPRPE path; the optional `rpe_ops` extension was not built."),)


def build_rpe(rpe_config, head_dim, num_heads):
    if rpe_config is None:
        return None
    else:
        name = rpe_config.name
        if name == 'KPRPE_shared':
            rpe_config = kprpe_shared.get_rpe_config(
                ratio=rpe_config.ratio,
                method=rpe_config.method,
                mode=rpe_config.mode,
                shared_head=rpe_config.shared_head,
                skip=0,
                rpe_on=rpe_config.rpe_on,
            )
            return kprpe_shared.build_rpe(rpe_config, head_dim=head_dim, num_heads=num_heads)

        else:
            raise NotImplementedError(f"Unknow RPE: {name}")
"""

SAFE_FVCORE_IMPORT = """try:
    from fvcore.nn import flop_count
except Exception:
    def flop_count(*args, **kwargs):
        return ({}, {})
"""


def ensure_dependencies() -> None:
    missing: list[str] = []
    for module_name, pip_name in DEPENDENCIES:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(pip_name)

    if not missing:
        return

    print(f"Installing missing dependencies: {', '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def download_repo(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download

    target_dir = MODELS_DIR / repo_id.split("/")[-1]
    MODELS_DIR.mkdir(exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(target_dir))
    return target_dir


def patch_repo(repo_dir: Path) -> None:
    rpe_init = repo_dir / "models" / "vit_kprpe" / "RPE" / "__init__.py"
    if rpe_init.exists():
        current = rpe_init.read_text()
        if "subprocess.check_call" in current:
            rpe_init.write_text(SAFE_RPE_INIT)

    iresnet_model = repo_dir / "models" / "iresnet" / "model.py"
    if iresnet_model.exists():
        current = iresnet_model.read_text()
        if "from fvcore.nn import flop_count" in current:
            iresnet_model.write_text(
                current.replace("from fvcore.nn import flop_count", SAFE_FVCORE_IMPORT)
            )


def clear_repo_modules() -> None:
    for name in list(sys.modules):
        if name == "wrapper" or name.startswith("models"):
            del sys.modules[name]


def load_repo_model(repo_dir: Path):
    import yaml

    clear_repo_modules()
    cwd = Path.cwd()
    sys.path.insert(0, str(repo_dir))
    os.chdir(repo_dir)

    try:
        from wrapper import CVLFaceRecognitionModel, ModelConfig

        wrapper_model = CVLFaceRecognitionModel(ModelConfig())
        model = wrapper_model.model.eval()
        config = yaml.safe_load((repo_dir / "pretrained_model" / "model.yaml").read_text())
        return model, config
    finally:
        os.chdir(cwd)
        if str(repo_dir) in sys.path:
            sys.path.remove(str(repo_dir))


def make_example_keypoints() -> "torch.Tensor":
    import torch

    return torch.tensor(
        [[[0.34, 0.38], [0.66, 0.38], [0.50, 0.58], [0.40, 0.76], [0.60, 0.76]]],
        dtype=torch.float32,
    )


def convert_model(model, config: dict, package_name: str) -> Path:
    import coremltools as ct
    import torch

    class ExportWrapper(torch.nn.Module):
        def __init__(self, inner_model, expects_keypoints: bool):
            super().__init__()
            self.inner_model = inner_model
            self.expects_keypoints = expects_keypoints

        def forward(self, image, keypoints=None):
            if self.expects_keypoints:
                return self.inner_model(image, keypoints)
            return self.inner_model(image)

    input_shape = tuple(config["input_size"])
    channels, height, width = input_shape
    if channels != 3:
        raise ValueError(f"Unsupported input shape: {input_shape}")

    expects_keypoints = "kprpe" in str(config.get("yaml_path", "")).lower()
    export_model = ExportWrapper(model, expects_keypoints).eval()
    image = torch.randn(1, channels, height, width)

    if expects_keypoints:
        keypoints = make_example_keypoints()
        traced = torch.jit.trace(
            export_model,
            (image, keypoints),
            strict=False,
            check_trace=False,
        )
        inputs = [
            ct.ImageType(
                name="image",
                shape=image.shape,
                scale=1 / 127.5,
                bias=[-1.0, -1.0, -1.0],
                color_layout=ct.colorlayout.RGB,
            ),
            ct.TensorType(name="keypoints", shape=keypoints.shape),
        ]
    else:
        traced = torch.jit.trace(
            export_model,
            image,
            strict=False,
            check_trace=False,
        )
        inputs = [
            ct.ImageType(
                name="image",
                shape=image.shape,
                scale=1 / 127.5,
                bias=[-1.0, -1.0, -1.0],
                color_layout=ct.colorlayout.RGB,
            )
        ]

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=ct.target.iOS18,
    )

    output_path = APP_MODELS_DIR / f"{package_name}.mlpackage"
    if output_path.exists():
        shutil.rmtree(output_path)
    mlmodel.save(str(output_path))
    return output_path


def cleanup_partial_package(package_name: str) -> None:
    output_path = APP_MODELS_DIR / f"{package_name}.mlpackage"
    if output_path.exists():
        shutil.rmtree(output_path)


def main() -> int:
    ensure_dependencies()

    for candidate in MODEL_CANDIDATES:
        repo_id = candidate["repo_id"]
        package_name = candidate["package_name"]
        description = candidate["description"]
        print(f"Trying {description} ({repo_id})")

        try:
            repo_dir = download_repo(repo_id)
            patch_repo(repo_dir)
            model, config = load_repo_model(repo_dir)
            output_path = convert_model(model, config, package_name)
            print(f"Saved {description} to {output_path}")
            return 0
        except Exception:
            cleanup_partial_package(package_name)
            print(f"Failed to convert {description}")
            traceback.print_exc()

    print("No candidate model converted successfully.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
