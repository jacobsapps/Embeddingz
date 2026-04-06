# Face Pipeline Walkthrough

This document describes the current end-to-end pipeline in the app after the model and clustering rewrite.

## 1. Photo enumeration

Photos are loaded from PhotoKit through `PhotoManager`. The fetch limit remains configurable so the app can be used on:

- a small test library
- a fixed-size proof-of-concept dataset
- the user’s full library

The Faces tab starts from `PHAsset`s, not files on disk, so the whole pipeline stays inside the normal iOS Photos model.

The photo-facing screens no longer try to stand up the whole visible library at once. The photo grids now:

- keep identifier and index lookups in memory so cells do not scan the full asset array repeatedly
- page more assets into the diffable snapshot as the user scrolls
- request cell-sized images instead of oversized thumbnails

## 2. Parallel scan stage

The first stage is still the fast part.

For each asset:

1. Request a reasonably sized image from PhotoKit
2. Run `VNDetectFaceRectanglesRequest`
3. Drop low-quality faces before embedding
4. Run landmark detection for each remaining face
5. Generate an embedding
6. Insert that face into SQLite through GRDB

The work runs with a bounded task group using roughly:

`activeProcessorCount - 1`

workers so the app can keep the UI responsive while still using the device.

## 3. Landmark-aware face preparation

The biggest preprocessing change is between face detection and embedding generation.

The old pipeline mostly used a padded bounding box crop. The current pipeline tries to derive a more stable input from the face geometry:

1. Run `VNDetectFaceLandmarksRequest` inside the face ROI
2. Extract a five-point layout from the eyes, nose, and mouth when possible
3. Build an aligned square crop around that geometry
4. Normalize those keypoints into model input coordinates
5. Fall back to an approximate face-rectangle layout if landmarks are unavailable

This is not full 3D frontalization. It is a pragmatic alignment step that materially reduces crop drift from photo to photo.

## 4. Embedding model selection

The app now chooses the best bundled local model at runtime:

1. `CVLFaceViTKPRPE`
2. `CVLFaceIR101`
3. `FaceNet`

That fallback chain lets the project support a strong research-grade model while still keeping a smaller legacy path available.

## 5. Direct Core ML inference

The processor no longer uses `VNCoreMLRequest` for embeddings. Instead, it keeps a pool of plain `MLModel` instances and calls Core ML directly.

That change matters because the preferred model expects two inputs:

- `image`
- `keypoints`

The KPRPE variant gets both the aligned crop and the normalized five-point landmarks. The image-only fallbacks ignore keypoints.

After inference, the app L2-normalizes the embedding so cosine similarity is just a dot product.

## 6. Versioned storage

Each stored face now includes:

- `assetId`
- `boundingBox`
- `embedding`
- `embeddingVersion`
- `personId`

`embeddingVersion` exists so the app never mixes vectors from incompatible pipelines inside the same clustering run.

For this prototype, the migration rule is intentionally blunt:

- if the database contains only an older embedding version, clear and rebuild

That keeps the graph internally consistent and makes the behavior easy to explain.

## 7. Honest second-stage status

When scan completes, the app does not jump into an opaque “clustering” bucket anymore.

It exposes explicit phases:

- `loadingEmbeddings`
- `buildingGraph`
- `clustering`
- `finalizing`

Each update includes:

- `completedUnits`
- `totalUnits`
- `statusText`
- `etaSeconds`
- `isProvisional`

That gives the UI enough structure to tell the truth about long-running work.

## 8. Blocked exact similarity

The expensive step is now phrased explicitly as graph construction.

Embeddings are stacked into a dense matrix and compared in tiles using Accelerate/BLAS:

```text
similarityBlock = A × Bᵀ
```

Because embeddings are unit-normalized, each value is already a cosine similarity.

The processor then:

- skips self-comparisons
- keeps only similarities above a relaxed candidate floor
- retains only the strongest neighbors per face

The neighbor budget is adaptive:

- small libraries keep far more local evidence because the storage cost is trivial
- larger libraries fall back to a capped top-`k` budget so memory stays bounded

This is still exact pairwise similarity work, but it avoids storing or reasoning about a dense graph afterward.

## 9. Provisional graph updates

Graph construction is long enough to deserve visible partial output.

Every few tiles, the processor:

1. snapshots the current sparse candidate graph
2. runs a provisional clustering pass
3. converts those labels into `PersonCluster` previews
4. publishes them to the Faces tab

That gives the user a more honest story:

- photo scanning is done
- graph construction is still running
- the visible people groups are provisional until the full graph is available

## 10. Sparse clustering with threshold scoring

Once a candidate graph exists, the processor does not rely on one hard-coded threshold anymore.

Instead it:

1. tries a small schedule of thresholds around the selected sensitivity
2. runs sparse weighted label propagation at each threshold
3. scores every pass instead of stopping at the first acceptable one
4. keeps the best pass based on grouped-face coverage, cluster strength, and overall cluster count

This matters because the new embedding models do not use the same cosine scale as the old FaceNet setup, and a threshold that is too strict can leave the user with dozens of duplicate “people.”

## 11. Conservative repair passes

After the main pass, the app runs two narrow cleanup stages.

### 11a. Tiny-fragment attach

Tiny fragments, currently clusters of size `<= 2`, can be attached to a larger target only when:

- every member has a strong best external match
- those best matches all point to the same target cluster
- the target cluster is larger than the fragment

### 11b. Consistent-cluster merge

The app also repairs obvious split identities when:

- several faces in the smaller cluster bridge to the same larger target
- the larger target points back with real external support
- the cluster centroids still agree

This keeps the post-pass useful without reintroducing an open-ended merge loop.

## 12. Finalization and UI handoff

During finalization, the processor:

- preserves person names where possible by majority overlap with previous assignments
- batch-writes final `personId`s in one transaction
- swaps the UI from provisional to final clusters

The Faces screen now keeps the current stage visible and distinct instead of collapsing everything into one long generic mapping bar. The collection view also stays alive during processing:

- recent scanned faces stay pinned at the top while the scan is running
- provisional cluster updates arrive during graph build and merge passes
- visible cluster cells animate when provisional merges collapse identities

## 13. Face-space visualization

The Graphs tab now works from the same unit as clustering: detected faces.

The app builds:

- a global PCA overview using all assigned faces for the active embedding version
- a SceneKit 3D cloud for high-level structure
- a Swift Charts 2D map for precise picking
- a selected-person detail map that reuses the same global coordinates instead of recomputing a local projection
- thumbnail strips for both person selection and per-face scrubbing

That makes the graph useful for debugging split identities without breaking the connection back to the real face crops and source photos.

## 14. Why this pipeline is blog-friendly

The current architecture is easier to explain because each step has a clean job:

- Vision finds faces
- landmark processing stabilizes the crop
- the best available local model produces embeddings
- blocked exact similarity builds a sparse candidate graph
- weighted label propagation groups identities
- narrow repair passes fix only obvious leftovers and split clusters
- a real status model explains the second stage honestly

That is a much more defensible story than “run some clustering, then wait and hope.”
