# Embeddingz — Technical Overview

## Architecture

Embeddingz is an iOS app that scans a photo library, detects faces, generates embeddings locally, clusters them into identities, and visualizes the embedding space.

The app is built around three main subsystems:

- `PhotoManager` for PhotoKit access and caching
- `FaceProcessor` for face detection, embedding generation, clustering, and progress reporting
- `FaceDatabase` for local persistence through GRDB / SQLite

## Face Detection

The app uses Apple Vision for face detection:

- `VNDetectFaceRectanglesRequest` finds candidate faces
- very small or low-confidence detections are discarded before embedding
- `VNDetectFaceLandmarksRequest` is then used to improve crop normalization before model inference

The important change in the current pipeline is that embedding inputs are now landmark-aware rather than being based purely on a padded bounding box.

## Embedding Model

The app now prefers the strongest supported local model in the bundle:

- primary: `CVLFace AdaFace ViT-Base KPRPE WebFace12M`
- fallback: `CVLFace AdaFace IR101 WebFace12M`
- legacy fallback: `FaceNet`

The current inference path no longer goes through `VNCoreMLModel`. The processor keeps a pool of plain `MLModel` instances and calls Core ML directly so it can feed both:

- the aligned face image
- normalized five-point facial keypoints for the KPRPE model variant

All embeddings are L2-normalized after inference so cosine similarity reduces to a dot product.

## Storage

Face records are stored in SQLite with GRDB. Each row includes:

- `assetId`
- `boundingBox`
- `embedding`
- `embeddingVersion`
- `personId`

`embeddingVersion` is used to prevent old and new embedding pipelines from being mixed together silently. For this proof-of-concept, a pipeline version change triggers a rebuild instead of a complicated live migration.

## Clustering

The clustering path was rewritten around a sparse graph model with an adaptive threshold sweep.

### Old shape

The old app built a dense similarity graph, ran Chinese Whispers, then tried to merge clusters again with a centroid-based HAC pass.

### Current shape

The current app:

1. Loads embeddings for the active version
2. Computes similarities in blocks with Accelerate/BLAS
3. Keeps the strongest neighbors for each face above a relaxed candidate floor
4. Runs a sparse weighted label-propagation pass across several threshold candidates
5. Scores every threshold candidate instead of stopping at the first acceptable one
6. Runs a conservative tiny-fragment attach pass
7. Runs a consistent-cluster merge pass for obvious split identities
8. Batch-writes final `personId`s

This is intentionally less brittle than the previous mutual-only component pass:

- exact blocked similarity still provides the evidence
- sparse storage keeps the graph bounded
- label propagation lets repeated local agreement reinforce identities
- the threshold sweep prevents the app from getting stuck in an over-strict early pass
- the repair merge only fires when bridge evidence keeps pointing at the same target

## Progress Reporting

The processor now exposes a richer status model instead of a single generic clustering progress bar.

Each update includes:

- `phase`
- `completedUnits`
- `totalUnits`
- `statusText`
- `etaSeconds`
- `isProvisional`

That supports UI states like:

- scanning photos
- loading embeddings
- building the neighbor graph
- forming identity clusters
- finalizing assignments

The second-stage ETA is derived from processed similarity blocks rather than guessed from face count alone.

## Provisional Results

During graph construction the app can emit provisional clusters from the partial graph. Those clusters are visible in the UI before the final database write, which makes long-running clustering feel alive instead of stalled.

The provisional state is explicitly marked so the app does not pretend those assignments are final identities yet.

The Faces screen also now keeps the UI populated while the work is happening:

- recent scanned face crops stay visible at the top of the grid
- the progress header breaks work into scan, graph, merge, and save rows
- provisional cluster updates animate into place during merge-heavy passes

That makes the second stage feel like active work rather than a frozen placeholder.

## Photo Loading And Performance

The slow-feeling parts of the app were not just about clustering.

The main load-path fixes are now:

- `PhotoManager` keeps identifier and index lookups in memory so collection cells do not linearly search the full asset array
- the photo grids page more items into the diffable snapshot as the user scrolls instead of eagerly rendering the full fetched list
- grid cells request fast cell-sized PhotoKit images
- the full-screen browser requests roughly screen-sized images with a zoom budget instead of defaulting toward full-resolution assets
- the browser thumbnail strip only reloads the changed selection cells instead of reloading the entire strip on every move
- face thumbnails are cropped from smaller source requests and cached with a memory budget

Those changes do not turn PhotoKit into a database-backed infinite scroller, but they remove several avoidable sources of UI churn and oversized image work.

## Visualization

The Graphs tab is now a face-space explorer rather than a generic scatter view.

The important semantic choice is that the plotted unit is a detected face crop, not a photo. The graph is built from stored face embeddings for the active embedding version and colored by `personId`.

The tab now exposes:

- a global overview built from all assigned faces
- a 3D SceneKit cloud for broad spatial context
- a 2D Swift Charts mode for precise selection
- a selected-person detail chart that uses the same global coordinates
- a thumbnail scrubber for the chosen identity

The projection math is still PCA-based:

- mean-center embeddings
- compute covariance with Accelerate
- run eigen decomposition with LAPACK
- project onto the leading principal components

The UI intentionally hides raw component labels now. The graph is meant to be useful first, mathematically explicit second.

## Why this matters

The big technical shift in the app is not just “swap in a bigger model.” It is the combination of:

- better normalized inputs
- stronger embedding checkpoints
- versioned embeddings
- sparse graph construction
- adaptive identity formation
- honest progress reporting

That makes the system more defensible both as software and as a story you can explain in a technical writeup.
