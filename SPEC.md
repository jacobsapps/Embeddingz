# Embeddingz — App Specification

**Building an image face tagging system using Core ML and embeddings.**

An iOS app that replicates Apple's face identification system using the Vision framework, CoreML face embeddings, and local vector storage.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    EmbeddingzApp                     │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ PhotoMgr │  │ FaceProc │  │   FaceDatabase    │  │
│  │(PHAssets)│──│(pipeline)│──│ (GRDB / SQLite)   │  │
│  └──────────┘  └──────────┘  └───────────────────┘  │
│       │              │                  │            │
│  ┌─────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │PhotosTab│  │ FacesTab │  │  EmbeddingsTab    │  │
│  │ (grid)  │  │(clusters)│  │ (face explorer)   │  │
│  └─────────┘  └──────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Processing Pipeline

```
PHAsset
  │
  ▼
PHImageManager (1024×1024)
  │
  ▼
VNDetectFaceRectanglesRequest
  │
  ▼
VNDetectFaceLandmarksRequest
  │
  ▼
Aligned face crop + normalized 5-point landmarks
  │
  ▼
Best available local model
(CVLFace ViT KPRPE → IR101 → FaceNet fallback)
  │
  ▼
L2 normalize embedding
  │
  ▼
Blocked cosine similarity graph
  + sparse label propagation clustering
  + fragment repair
  + consistent-cluster merge
  + batch `personId` write
  ▼
Store in GRDB with `embeddingVersion`
```

## Tab Structure

### Tab 1: Photos
- Displays all photos from the user's library in a 3-column `LazyVGrid`
- Uses `PHCachingImageManager` for performant thumbnail loading
- Handles photo library authorization flow
- Fetches the most recent 1,000 photos

### Tab 2: Faces
- Shows discovered face clusters in a grid layout
- Each cluster displays a person icon, ID, and face count
- Header shows scan progress, clustering progress, status text, ETA, and provisional state
- Tap the scan button (Face ID icon) to start processing
- Tap a cluster to navigate to all photos containing that person

### Tab 3: Embeddings
- Browse all detected faces for the active embedding version
- "All Faces" card shows a global PCA map of every face embedding
- Selected-person card shows a local PCA map for one cluster plus face thumbnails
- 2D mode supports tap selection; 3D mode is an overview
- PCA projection computed via LAPACK (`ssyev_`) eigen decomposition with explained-variance labels

## Data Model

### FaceRecord (SQLite via GRDB)

| Column      | Type    | Description                              |
|-------------|---------|------------------------------------------|
| id          | INTEGER | Auto-incrementing primary key            |
| assetId     | TEXT    | `PHAsset.localIdentifier`                |
| boundingBox | BLOB    | JSON-encoded `CGRect` from Vision        |
| embedding   | BLOB    | 512 × Float32 = 2048 bytes raw data     |
| embeddingVersion | INTEGER | Active model / preprocessing version |
| personId    | INTEGER | Cluster assignment ID                    |

### PersonCluster (in-memory)

| Field    | Type    | Description                     |
|----------|---------|---------------------------------|
| id       | Int     | Unique cluster identifier       |
| centroid | [Float] | Normalized centroid for previews |
| count    | Int     | Number of faces in cluster      |

## Key Technical Details

### Face Embedding Model
- **Preferred model**: CVLFace AdaFace ViT-Base KPRPE WebFace12M
- **Fallbacks**: CVLFace AdaFace IR101, then FaceNet
- **Inputs**: aligned RGB face crop plus normalized five-point landmarks when the model requires them
- **Output**: unit-normalized embedding vector
- **Conversion**: one-shot installer + Core ML export via `Scripts/install_best_face_model.py`

### Clustering Algorithm
Sparse exact-similarity graph with adaptive label propagation:
1. Compute cosine similarities in matrix tiles with Accelerate
2. Retain the strongest neighbors per face, with a larger budget on small libraries
3. Run weighted sparse label propagation across a short threshold schedule
4. Score all threshold passes instead of stopping at the first acceptable one
5. Run a tiny-fragment attach pass
6. Run a consistent-cluster merge pass for obvious split identities
7. Batch-write final `personId`s to SQLite

### PCA Projection
- Computes the covariance matrix of centered embeddings using `cblas_sgemm`
- Eigen decomposition via LAPACK `ssyev_` (ascending eigenvalues)
- Projects onto the top 2 or 3 principal components for 2D/3D visualization
- Rendered as an interactive face explorer using Swift Charts / Chart3D
- Uses the same face-level unit as clustering rather than plotting whole photos

### Concurrency Model
- Swift 6 strict concurrency with `MainActor` default isolation
- `PhotoManager` and `FaceProcessor` are `@MainActor` `ObservableObject`s
- Heavy computation (face detection, CoreML inference, PCA) runs via `Task.detached` and `nonisolated` static methods
- `FaceDatabase` is `Sendable` (thread-safe GRDB `DatabaseQueue`)

## Dependencies

| Dependency | Source | Purpose |
|------------|--------|---------|
| GRDB.swift | SPM    | SQLite database for face records + embeddings |
| Photos     | Apple  | Photo library access and caching |
| Vision     | Apple  | Face detection + landmarks |
| CoreML     | Apple  | Local face embedding inference |
| Accelerate | Apple  | BLAS/LAPACK for PCA, vDSP for vector ops |
| Charts     | Apple  | Scatter plot visualization |

## File Structure

```
Embeddingz/
├── EmbeddingzApp.swift              # App entry point, dependency injection
├── ContentView.swift                # TabView with 3 tabs
├── Models/
│   ├── FaceDatabase.swift           # GRDB schema, data types, CRUD
│   ├── PhotoManager.swift           # PHAsset fetching & caching
│   └── FaceProcessor.swift          # Vision + CoreML + clustering pipeline
├── Utilities/
│   └── PCAProjection.swift          # Accelerate-based PCA
├── Views/
│   ├── PhotosViewController.swift   # Tab 1: photo grid
│   ├── FacesViewController.swift    # Tab 2: face clusters + progress
│   ├── FaceClusterCell.swift        # Reusable cluster preview cell
│   ├── PersonPhotosViewController.swift # Filtered photos for one person
│   ├── PhotoBrowserViewController.swift # Full-screen photo browsing
│   ├── EmbeddingsViewController.swift # Tab 3: embeddings browser
│   └── EmbeddingExplorerView.swift  # Swift Charts face-space explorer
└── Assets.xcassets/
Scripts/
├── convert_model.py                 # Legacy FaceNet converter
└── install_best_face_model.py       # Best-model downloader + CoreML conversion
```

## Setup Instructions

1. Run the one-shot installer:
   ```bash
   python3 Scripts/install_best_face_model.py
   ```
2. Open `Embeddingz.xcodeproj`
3. Build and run on a device with photos
4. Grant photo library access when prompted
5. Navigate to the Faces tab and tap the scan button
