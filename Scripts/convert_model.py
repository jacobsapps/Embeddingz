"""
convert_model.py — Convert FaceNet PyTorch model to CoreML format.

Usage:
    pip install facenet-pytorch coremltools torch numpy
    python convert_model.py

Produces: FaceNet.mlpackage (512-dim embedding model, 160×160 RGB input)
"""

import torch
import coremltools as ct
import numpy as np
from facenet_pytorch import InceptionResnetV1

# Load pretrained FaceNet (VGGFace2 weights)
model = InceptionResnetV1(pretrained="vggface2").eval()

# Trace with dummy input (1, 3, 160, 160)
dummy_input = torch.randn(1, 3, 160, 160)
traced_model = torch.jit.trace(model, dummy_input)

# Convert to CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="image", shape=(1, 3, 160, 160),
                         scale=1.0 / 255.0, bias=[0, 0, 0])],
    outputs=[ct.TensorType(name="embedding")],
    minimum_deployment_target=ct.target.iOS18,
)

mlmodel.author = "facenet-pytorch (VGGFace2)"
mlmodel.short_description = "FaceNet InceptionResnetV1 — 512-dim face embeddings"
mlmodel.save("FaceNet.mlpackage")
print("Saved FaceNet.mlpackage")
