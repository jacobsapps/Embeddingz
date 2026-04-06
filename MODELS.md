# Face Model Notes

This document explains the model-side changes in the app and why the current setup looks different from the original FaceNet prototype.

## Why the model changed

The original app used a FaceNet-style embedding model. That was enough to prove out the architecture, but it had two limitations for a real photo library:

- weaker separation on messy consumer photos
- a preprocessing path that only used a padded face rectangle

The current app upgrades both parts of that stack:

- stronger modern checkpoints
- landmark-aware inputs

## Preferred model

The primary bundled model is:

`CVLFace AdaFace ViT-Base KPRPE WebFace12M`

Why this one:

- it is materially stronger than the old FaceNet baseline
- it benefits from five-point face geometry instead of just the image crop
- the quality target here is “best local prototype,” not “smallest mobile checkpoint”

The fallback chain is:

1. `CVLFaceViTKPRPE`
2. `CVLFaceIR101`
3. `FaceNet`

That keeps the project usable even if the heaviest conversion path is unavailable.

## Why the app no longer uses `VNCoreMLModel`

Vision requests are convenient for image-only models. The current preferred checkpoint expects:

- an aligned face image
- normalized five-point keypoints

That makes direct `MLModel` inference the better fit. The app now pools plain `MLModel` instances behind an actor and feeds model inputs explicitly.

This change is structural, not cosmetic. Without it, the app could not drive the KPRPE model properly.

## Input preparation

For each face:

1. Vision finds the face rectangle
2. Vision landmarks estimate eyes, nose, and mouth geometry
3. The app builds an aligned square crop
4. The app normalizes five keypoints into model coordinates
5. Core ML receives `image` and, when required, `keypoints`

If landmarks are missing, the app falls back to an approximate five-point layout derived from the detected face rectangle.

## Output handling

The embedding output is normalized to unit length after inference. That gives the rest of the pipeline a clean contract:

- embeddings are directly comparable with cosine similarity
- blocked matrix multiplication can produce the similarity graph efficiently

## Installer script

The repo now uses a single setup script:

```bash
python3 Scripts/install_best_face_model.py
```

That script:

- installs missing Python dependencies if needed
- downloads the strongest supported checkpoint from Hugging Face
- patches optional upstream import paths that are irrelevant to inference
- traces the PyTorch model
- converts it to a Core ML `mlprogram`
- writes the result into `Embeddings/`

Raw downloaded repos are stored under `Models/`. Both `Models/` and generated model packages are gitignored.

## Conversion details

The preferred ViT checkpoint is exported as an iOS 18 Core ML program.

Important details:

- input image layout: RGB
- image normalization: `image / 127.5 - 1.0`
- keypoint tensor shape: `1 × 5 × 2`
- output name: `embedding`

The script also carries a fallback IR101 export path so the app can still ship a stronger CNN checkpoint if the ViT conversion path fails.

## Why this matters for clustering

Changing the model alone is not enough. A stronger embedding model changes:

- the shape of the nearest-neighbor graph
- the similarity values users should threshold on
- the failure modes when thresholds are too strict

That is why the clustering rewrite and the model rewrite were done together. The app now treats clustering sensitivity as model-version-specific instead of assuming every model behaves like FaceNet.
