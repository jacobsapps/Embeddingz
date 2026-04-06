# Embeddingz

An iOS prototype for local face detection, embedding generation, clustering, and face-space exploration using PhotoKit, Vision, Core ML, and GRDB.

## Setup

### 1. Generate a local model package

```bash
python3 Scripts/install_best_face_model.py
```

The installer:

- downloads the strongest supported checkpoint, currently `CVLFace AdaFace ViT-Base KPRPE WebFace12M`
- falls back to `CVLFace AdaFace IR101 WebFace12M` if the preferred export path fails
- converts the model to Core ML and writes the generated package into `Embeddings/`
- keeps raw downloaded artifacts under `Models/`

`Models/`, generated `.mlpackage` files, and build products are intentionally excluded from version control.

### 2. Build and run

Open `Embeddings.xcodeproj`, run on a device with a photo library, grant photo access, and start a scan from the Faces tab.

## Project layout

- `Embeddings/Models/` contains the scan pipeline, database layer, and photo-library integration
- `Embeddings/Views/` contains the Photos, Faces, and embedding-browser UI
- `Scripts/install_best_face_model.py` installs the local Core ML model used at runtime
