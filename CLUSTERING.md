# Face Clustering Rewrite

## Why the old second stage felt broken

The original app had two separate identity-grouping ideas layered on top of each other:

1. Build a dense similarity graph by comparing every face against every other face
2. Run Chinese Whispers on that graph
3. Run a second centroid-based merge pass to glue clusters back together

That design had three practical problems.

### 1. The user experience was opaque

The expensive part was the pairwise graph build, but the UI did not expose it honestly. The user mostly saw a generic “clustering” progress bar with no block count, no ETA, and no sense of whether the app was still working or had stalled.

### 2. The algorithm was fighting itself

Chinese Whispers works when local graph edges are meaningful. The old HAC-style merge pass reintroduced centroid trust after the graph stage had already separated identities.

In practice, that meant the app could first split cautiously and then merge too aggressively anyway.

### 3. The early rewrite was still too brittle

The first pass of the rewrite removed HAC, but it still leaned too hard on reciprocal-only edges and a FaceNet-era threshold scale. With newer models, that could leave the graph too sparse and produce the exact failure mode users notice immediately:

- the same person appears many times
- the second bar finishes without a satisfying explanation
- clustering technically ran, but the result still looked broken

## What the current clustering stage does

The shipped implementation is now a sparse exact-similarity pipeline with adaptive threshold selection and a repair pass for obvious split identities.

## Step 1. Build an exact candidate graph, but keep more neighbors when the library is small

Every embedding is still compared exactly, but not with nested Swift loops. Embeddings are stacked into a matrix and compared in tiles using Accelerate/BLAS.

For each tile pair, the processor:

- computes a cosine-similarity block in one batched operation
- ignores weak values below a relaxed candidate floor
- retains only the strongest neighbors for each face

The important tuning change is that the neighbor budget is now adaptive:

- for a few hundred faces, the processor can keep essentially all meaningful neighbors
- for larger libraries, it falls back to a capped top-`k` budget

That matters because under-merge often comes from throwing away too much local evidence, not from the embedding model alone.

## Step 2. Run sparse weighted label propagation

Once the candidate graph exists, the processor runs a sparse Chinese-Whispers-style pass:

- each node starts with its own label
- neighbors vote with weights derived from similarity
- reciprocal evidence gets a small bonus
- labels propagate until the graph stabilizes or hits an iteration cap

This is more forgiving than strict reciprocal connected components. A face no longer has to survive a hard mutual-edge filter just to join an obviously related group.

## Step 3. Score several thresholds instead of stopping at the first acceptable one

The current embedding models do not share the same similarity scale as the older FaceNet pipeline.

So the processor does not trust one fixed threshold blindly. It tries a short schedule around the selected clustering sensitivity:

- requested threshold
- slightly lower thresholds
- a guarded lower floor

The current implementation now evaluates the whole schedule and keeps the strongest result seen. It does not stop at the first “good enough” pass anymore.

The selection logic currently prefers:

- more grouped faces
- a larger strongest cluster
- fewer total clusters when the grouped coverage is otherwise equal

That change is specifically aimed at the user-visible failure mode where the same person still appears as four or five separate “people” even though the model itself is stronger.

## Step 4. Repair both tiny fragments and obvious split identities

Two narrow cleanup passes remain.

### 4a. Tiny-fragment attach

Tiny fragments, currently size `<= 2`, can attach to a larger cluster only when:

- every member has a strong best external match
- those matches unanimously point at the same target cluster
- the target cluster is larger than the fragment

### 4b. Consistent-cluster merge

The app also repairs larger split clusters when there is strong repeated evidence that they belong together.

Two clusters can merge only when:

- several faces in the smaller cluster bridge strongly to the same target cluster
- the target cluster also points back with real external support
- the normalized centroids still agree
- the strongest bridge edges clear a guarded threshold

This is still not a free-form agglomerative merge loop. It is a conservative repair step for identities that were clearly split by the earlier pass.

## Why this works better for face embeddings

The pipeline now leans into the strengths of the embedding space:

- exact local similarity evidence
- repeated agreement across a sparse graph
- threshold selection matched to the current model family
- narrow repair passes only where the graph keeps pointing at the same answer

It still avoids the weakest part of the original design:

- unconstrained centroid-only identity decisions after graph clustering

## Progress and provisional results

The app now exposes second-stage work as real pipeline phases:

- `loadingEmbeddings`
- `buildingGraph`
- `clustering`
- `finalizing`

During graph construction, progress is reported in block counts such as:

`Building neighbor graph... 18/55 blocks`

ETA is estimated from observed block throughput rather than guessed from face count alone.

The app also emits provisional clusters during graph build so the Faces tab can show temporary structure before final database assignment.

Two UX details matter here:

- the user can see grouping structure before the job is done
- the second progress bar stays visible at completion instead of disappearing without explanation

## Why versioned thresholds matter too

One subtle bug in this kind of system is carrying a user’s old clustering threshold into a new embedding model.

That is exactly how you end up with a stronger model but worse apparent clustering: the math changed, but the slider value did not.

The app now treats clustering sensitivity as model-version-specific:

- FaceNet keeps its older threshold band
- the newer CVLFace models use a lower default range

That prevents the new model from inheriting an over-strict threshold from the old pipeline.

## The practical outcome

The current second stage is designed to be:

- easier to explain
- less brittle
- more honest in the UI
- better matched to the current embedding models

The philosophical change is straightforward:

- old pipeline: “compare everything, cluster it, then merge mistakes away”
- current pipeline: “keep more local evidence, score several graph passes, and only merge clusters when the graph repeatedly supports that merge”
