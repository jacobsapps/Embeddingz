import Photos
import Vision
import CoreML
import CoreImage
import CoreVideo
import Accelerate
import Observation

// MARK: - Model Pool Actor

/// Actor-based pool of MLModel instances for thread-safe concurrent embedding.
private actor ModelPool {
    private var available: [MLModel]
    private var waiters: [CheckedContinuation<MLModel, Never>] = []

    init(url: URL, count: Int) throws {
        available = []
        for _ in 0..<count {
            available.append(try MLModel(contentsOf: url))
        }
    }

    func borrow() async -> MLModel {
        if let model = available.popLast() {
            return model
        }
        return await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    func release(_ model: MLModel) {
        if let waiter = waiters.first {
            waiters.removeFirst()
            waiter.resume(returning: model)
        } else {
            available.append(model)
        }
    }
}

// MARK: - Processing Status

enum ProcessingPhase: String, Sendable, Equatable {
    case idle
    case scanning
    case loadingEmbeddings
    case buildingGraph
    case clustering
    case finalizing
}

struct ProcessingStatus: Sendable, Equatable {
    var phase: ProcessingPhase
    var completedUnits: Int
    var totalUnits: Int
    var statusText: String
    var etaSeconds: Double?
    var isProvisional: Bool

    static let idle = ProcessingStatus(
        phase: .idle,
        completedUnits: 0,
        totalUnits: 0,
        statusText: "",
        etaSeconds: nil,
        isProvisional: false
    )

    var fractionCompleted: Float {
        guard totalUnits > 0 else { return 0 }
        return Float(completedUnits) / Float(totalUnits)
    }

    var etaText: String? {
        guard completedUnits >= 2,
              let etaSeconds,
              etaSeconds.isFinite,
              etaSeconds > 1 else { return nil }
        let rounded = Int(etaSeconds.rounded())
        if rounded >= 3600 {
            return "\(rounded / 3600)h \(rounded % 3600 / 60)m"
        }
        if rounded >= 60 {
            return "\(rounded / 60)m \(rounded % 60)s"
        }
        return "\(rounded)s"
    }
}

// MARK: - Scanned Face Info

struct ScannedFace: Sendable, Equatable {
    let id: Int64
    let assetId: String
    let boundingBox: CGRect
}

// MARK: - Face Processor

@Observable
@MainActor
final class FaceProcessor {
    private struct NeighborCandidate: Sendable, Equatable {
        let neighbor: Int
        let similarity: Float
    }

    private struct PreparedFaceInput: Sendable {
        let image: CGImage
        let normalizedKeypoints: [CGPoint]?
    }

    private struct EmbeddingModelConfiguration: Sendable, Equatable {
        let resourceName: String
        let embeddingVersion: Int
        let inputSize: Int
        let imageInputName: String
        let keypointInputName: String?
        let outputName: String
        let requiresKeypoints: Bool

        static let cvlFaceViTKPRPE = EmbeddingModelConfiguration(
            resourceName: "CVLFaceViTKPRPE",
            embeddingVersion: 4,
            inputSize: 112,
            imageInputName: "image",
            keypointInputName: "keypoints",
            outputName: "embedding",
            requiresKeypoints: true
        )

        static let cvlFaceIR101 = EmbeddingModelConfiguration(
            resourceName: "CVLFaceIR101",
            embeddingVersion: 3,
            inputSize: 112,
            imageInputName: "image",
            keypointInputName: nil,
            outputName: "embedding",
            requiresKeypoints: false
        )

        static let faceNet = EmbeddingModelConfiguration(
            resourceName: "FaceNet",
            embeddingVersion: 2,
            inputSize: 160,
            imageInputName: "image",
            keypointInputName: nil,
            outputName: "embedding",
            requiresKeypoints: false
        )
    }

    private struct ClusteringSummary: Sendable, Equatable {
        let clusterCount: Int
        let singletonCount: Int
        let nonSingletonCount: Int
        let largestClusterSize: Int

        var nonSingletonCoverage: Double {
            let total = singletonCount + nonSingletonCount
            guard total > 0 else { return 0 }
            return Double(nonSingletonCount) / Double(total)
        }
    }

    private struct ClusteringPassResult: Sendable {
        let labels: [Int]
        let threshold: Float
        let summary: ClusteringSummary
    }

    private let database: FaceDatabase
    private let photoManager: PhotoManager

    var processedCount = 0
    var totalCount = 0
    var isProcessing = false
    var clusters: [PersonCluster] = []
    var detectedFaceCount = 0
    var scannedFaces: [ScannedFace] = []
    var status: ProcessingStatus = .idle

    /// Lower = more aggressive grouping (fewer clusters).
    /// Higher = more conservative (more precise identities).
    var similarityThreshold: Double {
        get {
            let configuration = Self.activeEmbeddingModelConfiguration()
            let value = UserDefaults.standard.double(
                forKey: Self.clusterThresholdDefaultsKey(for: configuration)
            )
            if value > 0 {
                return value
            }

            // Preserve the old single-key setting only for the legacy FaceNet path.
            if configuration == .faceNet {
                let legacyValue = UserDefaults.standard.double(forKey: "clusterThreshold")
                if legacyValue > 0 {
                    return legacyValue
                }
            }

            return Self.defaultSimilarityThreshold(for: configuration)
        }
        set {
            let configuration = Self.activeEmbeddingModelConfiguration()
            UserDefaults.standard.set(
                newValue,
                forKey: Self.clusterThresholdDefaultsKey(for: configuration)
            )
        }
    }

    var clusteringSensitivityOptions: [(String, Double)] {
        Self.clusteringSensitivityOptions(for: Self.activeEmbeddingModelConfiguration())
    }

    init(database: FaceDatabase, photoManager: PhotoManager) {
        self.database = database
        self.photoManager = photoManager
    }

    func resetState() {
        processedCount = 0
        totalCount = 0
        detectedFaceCount = 0
        isProcessing = false
        clusters = []
        scannedFaces = []
        status = .idle
    }

    func startProcessing() {
        guard !isProcessing else { return }
        isProcessing = true
        clusters = []
        scannedFaces = []
        processedCount = 0
        detectedFaceCount = 0
        totalCount = photoManager.assets.count
        status = ProcessingStatus(
            phase: .scanning,
            completedUnits: 0,
            totalUnits: totalCount,
            statusText: "0/\(totalCount) photos",
            etaSeconds: nil,
            isProvisional: false
        )

        let assets = photoManager.assets
        let totalAssetCount = assets.count
        let threshold = Float(similarityThreshold)
        let maxConcurrency = max(1, ProcessInfo.processInfo.activeProcessorCount - 1)
        let modelConfiguration = Self.activeEmbeddingModelConfiguration()

        Task.detached { [database, photoManager] in
            do {
                try Self.prepareDatabase(
                    database,
                    forEmbeddingVersion: modelConfiguration.embeddingVersion
                )

                let processedIds = try database.processedAssetIds(
                    embeddingVersion: modelConfiguration.embeddingVersion
                )

                var unprocessed: [PHAsset] = []
                var skipCount = 0
                for asset in assets {
                    if processedIds.contains(asset.localIdentifier) {
                        skipCount += 1
                    } else {
                        unprocessed.append(asset)
                    }
                }

                let existingFaceCount = (try? database.faceCount(
                    embeddingVersion: modelConfiguration.embeddingVersion
                )) ?? 0
                let initialProcessedCount = skipCount

                await MainActor.run {
                    self.processedCount = initialProcessedCount
                    self.detectedFaceCount = existingFaceCount
                    self.status = ProcessingStatus(
                        phase: .scanning,
                        completedUnits: initialProcessedCount,
                        totalUnits: self.totalCount,
                        statusText: "\(initialProcessedCount)/\(self.totalCount) photos",
                        etaSeconds: nil,
                        isProvisional: false
                    )
                }

                guard let modelURL = Bundle.main.url(
                    forResource: modelConfiguration.resourceName,
                    withExtension: "mlmodelc"
                ) else {
                    await MainActor.run {
                        self.status = .idle
                        self.isProcessing = false
                    }
                    return
                }

                let poolSize = min(maxConcurrency, max(1, unprocessed.count))
                let pool = try ModelPool(url: modelURL, count: poolSize)

                let scanStart = Date()
                await withTaskGroup(of: (String, [(CGRect, [Float])]).self) { group in
                    var iterator = unprocessed.makeIterator()
                    var inFlight = 0

                    while inFlight < maxConcurrency, let asset = iterator.next() {
                        group.addTask {
                            await Self.processAsset(
                                asset,
                                pool: pool,
                                photoManager: photoManager,
                                modelConfiguration: modelConfiguration
                            )
                        }
                        inFlight += 1
                    }

                    var completed = skipCount
                    for await (assetId, detectedFaces) in group {
                        var newFaces: [ScannedFace] = []
                        for (boundingBox, embedding) in detectedFaces {
                            var record = FaceRecord(
                                assetId: assetId,
                                boundingBox: FaceRecord.encodeBoundingBox(boundingBox),
                                embedding: FaceRecord.encodeEmbedding(embedding),
                                embeddingVersion: modelConfiguration.embeddingVersion,
                                personId: nil
                            )
                            try? database.insert(&record)
                            if let faceId = record.id {
                                newFaces.append(
                                    ScannedFace(
                                        id: faceId,
                                        assetId: assetId,
                                        boundingBox: boundingBox
                                    )
                                )
                            }
                        }

                        completed += 1
                        let elapsed = Date().timeIntervalSince(scanStart)
                        let eta = Self.eta(
                            elapsed: elapsed,
                            completed: completed,
                            total: max(totalAssetCount, 1)
                        )
                        let completedCount = completed
                        let newFaceBatch = newFaces

                        await MainActor.run {
                            self.processedCount = completedCount
                            self.detectedFaceCount += newFaceBatch.count
                            self.scannedFaces = Array(
                                (self.scannedFaces + newFaceBatch).suffix(24)
                            )
                            self.status = ProcessingStatus(
                                phase: .scanning,
                                completedUnits: completedCount,
                                totalUnits: self.totalCount,
                                statusText: "\(completedCount)/\(self.totalCount) photos",
                                etaSeconds: eta,
                                isProvisional: false
                            )
                        }

                        if let asset = iterator.next() {
                            group.addTask {
                                await Self.processAsset(
                                    asset,
                                    pool: pool,
                                    photoManager: photoManager,
                                    modelConfiguration: modelConfiguration
                                )
                            }
                        }
                    }
                }

                try await Self.performClustering(
                    on: self,
                    database: database,
                    threshold: threshold,
                    embeddingVersion: modelConfiguration.embeddingVersion
                )
            } catch {
                await MainActor.run {
                    self.status = .idle
                    self.isProcessing = false
                }
            }
        }
    }

    func loadExistingClusters() {
        do {
            let preferredVersion = Self.activeEmbeddingModelConfiguration().embeddingVersion
            guard let version = try Self.preferredEmbeddingVersion(
                in: database,
                preferredVersion: preferredVersion
            ) else {
                resetState()
                return
            }
            clusters = try Self.loadClusters(from: database, embeddingVersion: version)
            detectedFaceCount = try database.faceCount(embeddingVersion: version)
            processedCount = detectedFaceCount
            status = clusters.isEmpty
                ? .idle
                : ProcessingStatus(
                    phase: .idle,
                    completedUnits: 0,
                    totalUnits: 0,
                    statusText: "\(clusters.count) people found",
                    etaSeconds: nil,
                    isProvisional: false
                )
        } catch {
            resetState()
        }
    }

    func reclusterExisting() {
        guard !isProcessing else { return }
        let preferredVersion = (try? Self.preferredEmbeddingVersion(
            in: database,
            preferredVersion: Self.activeEmbeddingModelConfiguration().embeddingVersion
        )) ?? nil
        guard let version = preferredVersion else { return }

        isProcessing = true
        status = ProcessingStatus(
            phase: .loadingEmbeddings,
            completedUnits: 0,
            totalUnits: 1,
            statusText: "Loading embeddings...",
            etaSeconds: nil,
            isProvisional: false
        )

        Task.detached { [database] in
            do {
                try await Self.performClustering(
                    on: self,
                    database: database,
                    threshold: Float(self.similarityThreshold),
                    embeddingVersion: version
                )
            } catch {
                await MainActor.run {
                    self.status = .idle
                    self.isProcessing = false
                }
            }
        }
    }

    // MARK: - Clustering Pipeline

    nonisolated
    private static func performClustering(
        on processor: FaceProcessor,
        database: FaceDatabase,
        threshold: Float,
        embeddingVersion: Int
    ) async throws {
        let allFaces = try database.allFaces(embeddingVersion: embeddingVersion)
        guard !allFaces.isEmpty else {
            await MainActor.run {
                processor.status = .idle
                processor.isProcessing = false
            }
            return
        }

        let embeddings = allFaces.map(\.embeddingVector)
        guard let dim = embeddings.first?.count else {
            await MainActor.run {
                processor.status = .idle
                processor.isProcessing = false
            }
            return
        }

        await MainActor.run {
            processor.detectedFaceCount = allFaces.count
            processor.status = ProcessingStatus(
                phase: .loadingEmbeddings,
                completedUnits: 1,
                totalUnits: 1,
                statusText: "\(allFaces.count) faces ready",
                etaSeconds: nil,
                isProvisional: false
            )
        }

        let baseThreshold = max(0.18, threshold)
        let candidateFloor = max(0.10, baseThreshold - 0.20)
        let graphResult = await buildSparseNeighborGraph(
            embeddings: embeddings,
            dimension: dim,
            minimumCandidateSimilarity: candidateFloor,
            topK: recommendedNeighborCount(for: embeddings.count),
            tileSize: 128
        ) { processedTiles, totalTiles, etaSeconds, candidates in
            let statusText = "\(processedTiles)/\(totalTiles) blocks"
            await MainActor.run {
                processor.status = ProcessingStatus(
                    phase: .buildingGraph,
                    completedUnits: processedTiles,
                    totalUnits: totalTiles,
                    statusText: statusText,
                    etaSeconds: etaSeconds,
                    isProvisional: processedTiles > 0 && !processor.clusters.isEmpty
                )
            }

            guard let candidates else { return }

            let provisionalPass = await selectClusteringPass(
                from: candidates,
                count: embeddings.count,
                baseThreshold: baseThreshold
            )
            let provisionalLabels = provisionalPass.labels
            guard hasNonSingletonCluster(provisionalLabels) else { return }
            let provisionalClusters = previewClusters(
                faces: allFaces,
                embeddings: embeddings,
                labels: provisionalLabels
            )

            await MainActor.run {
                processor.clusters = provisionalClusters
                processor.status = ProcessingStatus(
                    phase: .buildingGraph,
                    completedUnits: processedTiles,
                    totalUnits: totalTiles,
                    statusText: "\(processedTiles)/\(totalTiles) blocks",
                    etaSeconds: etaSeconds,
                    isProvisional: true
                )
            }
        }

        let clusteringTotalUnits = thresholdCandidates(from: baseThreshold).count + 5
        let thresholdPassCount = thresholdCandidates(from: baseThreshold).count
        let selectedPass = await selectClusteringPass(
            from: graphResult,
            count: embeddings.count,
            baseThreshold: baseThreshold
        ) { completedPasses, totalPasses, result in
            let previewClusters = previewClusters(
                faces: allFaces,
                embeddings: embeddings,
                labels: result.labels
            )
            await MainActor.run {
                processor.clusters = previewClusters
                processor.status = ProcessingStatus(
                    phase: .clustering,
                    completedUnits: completedPasses,
                    totalUnits: clusteringTotalUnits,
                    statusText: "Pass \(completedPasses)/\(totalPasses)",
                    etaSeconds: nil,
                    isProvisional: true
                )
            }
        }
        var labels = selectedPass.labels
        labels = attachSmallFragments(
            labels: labels,
            candidates: graphResult,
            threshold: selectedPass.threshold
        )
        let cleanedLabels = labels
        await MainActor.run {
            processor.clusters = previewClusters(
                faces: allFaces,
                embeddings: embeddings,
                labels: cleanedLabels
            )
            processor.status = ProcessingStatus(
                phase: .clustering,
                completedUnits: thresholdPassCount + 1,
                totalUnits: clusteringTotalUnits,
                statusText: "Cleaning fragments",
                etaSeconds: nil,
                isProvisional: true
            )
        }
        labels = await mergeConsistentClusters(
            labels: labels,
            candidates: graphResult,
            embeddings: embeddings,
            threshold: selectedPass.threshold
        ) { mergePass, maxPasses, updatedLabels in
            let previewClusters = previewClusters(
                faces: allFaces,
                embeddings: embeddings,
                labels: updatedLabels
            )
            await MainActor.run {
                processor.clusters = previewClusters
                processor.status = ProcessingStatus(
                    phase: .clustering,
                    completedUnits: thresholdPassCount + 1 + mergePass,
                    totalUnits: clusteringTotalUnits,
                    statusText: "Merge pass \(mergePass)/\(maxPasses)",
                    etaSeconds: nil,
                    isProvisional: true
                )
            }
        }
        let finalSummary = clusteringSummary(for: labels)
        let assignments = sequentialAssignments(from: labels)
        let finalClusters = buildClusters(
            faces: allFaces,
            embeddings: embeddings,
            personIds: assignments
        )

        await MainActor.run {
            processor.clusters = finalClusters
            processor.status = ProcessingStatus(
                phase: .finalizing,
                completedUnits: 1,
                totalUnits: 1,
                statusText: "\(finalClusters.count) people",
                etaSeconds: nil,
                isProvisional: false
            )
        }

        let existingNames = try database.allNames()
        if !existingNames.isEmpty {
            var newToOldCounts: [Int: [Int: Int]] = [:]
            for (index, face) in allFaces.enumerated() {
                guard let oldId = face.personId else { continue }
                let newId = assignments[index]
                newToOldCounts[newId, default: [:]][oldId, default: 0] += 1
            }
            for (newId, counts) in newToOldCounts {
                if let bestOldId = counts.max(by: { $0.value < $1.value })?.key,
                   let name = existingNames[bestOldId] {
                    try database.setName(forPerson: newId, name: name)
                }
            }
        }

        let updates = zip(allFaces, assignments).compactMap { face, personId -> (Int64, Int)? in
            guard let faceId = face.id else { return nil }
            return (faceId, personId)
        }
        try database.batchUpdatePersonIds(updates)

        await MainActor.run {
            processor.clusters = finalClusters
            processor.scannedFaces = []
            processor.status = ProcessingStatus(
                phase: .idle,
                completedUnits: 0,
                totalUnits: 0,
                statusText: "\(finalClusters.count) people • \(finalSummary.nonSingletonCount)/\(allFaces.count) faces",
                etaSeconds: nil,
                isProvisional: false
            )
            processor.isProcessing = false
        }
    }

    nonisolated
    private static func buildSparseNeighborGraph(
        embeddings: [[Float]],
        dimension dim: Int,
        minimumCandidateSimilarity: Float,
        topK: Int,
        tileSize: Int,
        progress: @escaping @Sendable (Int, Int, Double?, [[NeighborCandidate]]?) async -> Void
    ) async -> [[NeighborCandidate]] {
        let n = embeddings.count
        guard n > 1 else { return Array(repeating: [], count: n) }

        var flat = [Float](repeating: 0, count: n * dim)
        for (row, embedding) in embeddings.enumerated() {
            let offset = row * dim
            flat.replaceSubrange(offset..<(offset + dim), with: embedding)
        }

        let tileCount = Int(ceil(Double(n) / Double(tileSize)))
        let totalTiles = tileCount * (tileCount + 1) / 2
        let refreshStride: Int
        switch totalTiles {
        case ..<12:
            refreshStride = 1
        case ..<48:
            refreshStride = 2
        default:
            refreshStride = max(4, totalTiles / 12)
        }
        var processedTiles = 0
        var candidates = Array(repeating: [NeighborCandidate](), count: n)
        let buildStart = Date()

        for tileRow in 0..<tileCount {
            let rowStart = tileRow * tileSize
            let rowEnd = min(rowStart + tileSize, n)
            let rowCount = rowEnd - rowStart

            for tileCol in tileRow..<tileCount {
                let colStart = tileCol * tileSize
                let colEnd = min(colStart + tileSize, n)
                let colCount = colEnd - colStart

                var similarities = [Float](repeating: 0, count: rowCount * colCount)
                flat.withUnsafeBufferPointer { buffer in
                    guard let base = buffer.baseAddress else { return }
                    let aPtr = base.advanced(by: rowStart * dim)
                    let bPtr = base.advanced(by: colStart * dim)

                    similarities.withUnsafeMutableBufferPointer { simBuffer in
                        guard let simPtr = simBuffer.baseAddress else { return }
                        cblas_sgemm(
                            CblasRowMajor,
                            CblasNoTrans,
                            CblasTrans,
                            Int32(rowCount),
                            Int32(colCount),
                            Int32(dim),
                            1.0,
                            aPtr,
                            Int32(dim),
                            bPtr,
                            Int32(dim),
                            0.0,
                            simPtr,
                            Int32(colCount)
                        )
                    }
                }

                for localRow in 0..<rowCount {
                    let globalRow = rowStart + localRow
                    for localCol in 0..<colCount {
                        let globalCol = colStart + localCol
                        if globalRow >= globalCol { continue }

                        let similarity = similarities[localRow * colCount + localCol]
                        guard similarity >= minimumCandidateSimilarity else { continue }

                        insertCandidate(
                            into: &candidates[globalRow],
                            neighbor: globalCol,
                            similarity: similarity,
                            topK: topK
                        )
                        insertCandidate(
                            into: &candidates[globalCol],
                            neighbor: globalRow,
                            similarity: similarity,
                            topK: topK
                        )
                    }
                }

                processedTiles += 1
                let etaSeconds = eta(
                    elapsed: Date().timeIntervalSince(buildStart),
                    completed: processedTiles,
                    total: totalTiles
                )
                let provisionalCandidates: [[NeighborCandidate]]? =
                    (processedTiles == totalTiles || processedTiles % refreshStride == 0)
                    ? candidates
                    : nil
                await progress(processedTiles, totalTiles, etaSeconds, provisionalCandidates)
            }
        }

        return candidates
    }

    nonisolated
    private static func insertCandidate(
        into candidates: inout [NeighborCandidate],
        neighbor: Int,
        similarity: Float,
        topK: Int
    ) {
        if let index = candidates.firstIndex(where: { $0.neighbor == neighbor }) {
            if similarity > candidates[index].similarity {
                candidates[index] = NeighborCandidate(neighbor: neighbor, similarity: similarity)
            }
        } else {
            candidates.append(NeighborCandidate(neighbor: neighbor, similarity: similarity))
        }

        candidates.sort { lhs, rhs in
            if lhs.similarity == rhs.similarity {
                return lhs.neighbor < rhs.neighbor
            }
            return lhs.similarity > rhs.similarity
        }
        if candidates.count > topK {
            candidates.removeSubrange(topK..<candidates.count)
        }
    }

    nonisolated
    private static func selectClusteringPass(
        from candidates: [[NeighborCandidate]],
        count: Int,
        baseThreshold: Float,
        progress: (@Sendable (Int, Int, ClusteringPassResult) async -> Void)? = nil
    ) async -> ClusteringPassResult {
        let thresholds = thresholdCandidates(from: baseThreshold)
        var bestResult: ClusteringPassResult?

        for (index, threshold) in thresholds.enumerated() {
            let labels = sparseChineseWhispers(
                from: candidates,
                count: count,
                threshold: threshold
            )
            let summary = clusteringSummary(for: labels)
            let result = ClusteringPassResult(
                labels: labels,
                threshold: threshold,
                summary: summary
            )

            if bestResult == nil || isBetter(result, than: bestResult!) {
                bestResult = result
            }

            if let progress {
                await progress(index + 1, thresholds.count, result)
            }
        }

        return bestResult ?? ClusteringPassResult(
            labels: Array(0..<count),
            threshold: baseThreshold,
            summary: clusteringSummary(for: Array(0..<count))
        )
    }

    nonisolated
    private static func sparseChineseWhispers(
        from candidates: [[NeighborCandidate]],
        count: Int,
        threshold: Float
    ) -> [Int] {
        guard count > 0 else { return [] }
        guard count > 1 else { return [0] }

        var adjacency = Array(repeating: [NeighborCandidate](), count: count)
        let candidateMaps = candidates.map { Dictionary(uniqueKeysWithValues: $0.map { ($0.neighbor, $0.similarity) }) }

        for node in 0..<count {
            for candidate in candidates[node] where candidate.similarity >= threshold {
                let reciprocal = candidateMaps[candidate.neighbor][node]
                let weight = reciprocal.map { max(candidate.similarity, $0) + 0.02 } ?? candidate.similarity
                insertCandidate(
                    into: &adjacency[node],
                    neighbor: candidate.neighbor,
                    similarity: weight,
                    topK: candidates[node].count
                )
                insertCandidate(
                    into: &adjacency[candidate.neighbor],
                    neighbor: node,
                    similarity: weight,
                    topK: candidates[candidate.neighbor].count + 1
                )
            }
        }

        var labels = Array(0..<count)
        let maxIterations = 20

        for _ in 0..<maxIterations {
            var changed = false
            let order = Array(0..<count).shuffled()

            for node in order {
                let neighbors = adjacency[node]
                guard !neighbors.isEmpty else { continue }

                var labelScores: [Int: Float] = [:]
                var labelSupports: [Int: Int] = [:]

                for neighbor in neighbors {
                    let neighborLabel = labels[neighbor.neighbor]
                    labelScores[neighborLabel, default: 0] += neighbor.similarity
                    labelSupports[neighborLabel, default: 0] += 1
                }

                let currentLabel = labels[node]
                let currentScore = labelScores[currentLabel] ?? 0
                let currentSupport = labelSupports[currentLabel] ?? 0
                guard let bestLabel = labelScores.keys.max(by: { lhs, rhs in
                    let leftScore = labelScores[lhs] ?? 0
                    let rightScore = labelScores[rhs] ?? 0
                    if leftScore == rightScore {
                        let leftSupport = labelSupports[lhs] ?? 0
                        let rightSupport = labelSupports[rhs] ?? 0
                        if leftSupport == rightSupport {
                            return lhs > rhs
                        }
                        return leftSupport < rightSupport
                    }
                    return leftScore < rightScore
                }) else {
                    continue
                }

                let bestScore = labelScores[bestLabel] ?? 0
                let bestSupport = labelSupports[bestLabel] ?? 0
                let shouldSwitch = bestLabel != currentLabel
                    && (
                        bestScore > currentScore + 0.0001
                        || (
                            abs(bestScore - currentScore) <= 0.0001
                            && bestSupport > currentSupport
                        )
                    )

                if shouldSwitch {
                    labels[node] = bestLabel
                    changed = true
                }
            }

            if !changed {
                break
            }
        }

        return normalizedLabels(labels)
    }

    nonisolated
    private static func thresholdCandidates(from baseThreshold: Float) -> [Float] {
        let floor = max(0.18, baseThreshold - 0.16)
        let values = [
            baseThreshold,
            baseThreshold - 0.04,
            baseThreshold - 0.08,
            baseThreshold - 0.12,
            floor
        ]
        var unique: [Float] = []
        for value in values {
            let clamped = min(0.90, max(0.18, value))
            if !unique.contains(where: { abs($0 - clamped) < 0.0001 }) {
                unique.append(clamped)
            }
        }
        return unique
    }

    nonisolated
    private static func recommendedNeighborCount(for faceCount: Int) -> Int {
        guard faceCount > 1 else { return 1 }

        switch faceCount {
        case ..<512:
            return faceCount - 1
        case ..<1_024:
            return min(faceCount - 1, 192)
        case ..<4_096:
            return min(faceCount - 1, 128)
        default:
            return 96
        }
    }

    nonisolated
    private static func clusteringSummary(for labels: [Int]) -> ClusteringSummary {
        let counts = Dictionary(grouping: labels, by: { $0 }).mapValues(\.count)
        let singletonCount = counts.values.filter { $0 == 1 }.count
        let nonSingletonCount = counts.values.filter { $0 > 1 }.reduce(0, +)
        let largestClusterSize = counts.values.max() ?? 0
        return ClusteringSummary(
            clusterCount: counts.count,
            singletonCount: singletonCount,
            nonSingletonCount: nonSingletonCount,
            largestClusterSize: largestClusterSize
        )
    }

    nonisolated
    private static func isBetter(
        _ lhs: ClusteringPassResult,
        than rhs: ClusteringPassResult
    ) -> Bool {
        if lhs.summary.nonSingletonCount != rhs.summary.nonSingletonCount {
            return lhs.summary.nonSingletonCount > rhs.summary.nonSingletonCount
        }
        if lhs.summary.largestClusterSize != rhs.summary.largestClusterSize {
            return lhs.summary.largestClusterSize > rhs.summary.largestClusterSize
        }
        if lhs.summary.clusterCount != rhs.summary.clusterCount {
            return lhs.summary.clusterCount < rhs.summary.clusterCount
        }
        return lhs.threshold > rhs.threshold
    }

    nonisolated
    private static func attachSmallFragments(
        labels: [Int],
        candidates: [[NeighborCandidate]],
        threshold: Float
    ) -> [Int] {
        guard !labels.isEmpty else { return [] }

        var updated = labels
        let attachThreshold = min(0.92, threshold + 0.08)
        let fragmentMaxSize = 2
        let membersByLabel = Dictionary(grouping: Array(updated.indices), by: { updated[$0] })
        let clusterSizes = membersByLabel.mapValues(\.count)

        for (label, members) in membersByLabel.sorted(by: { $0.value.count < $1.value.count }) {
            guard members.count <= fragmentMaxSize else { continue }

            var unanimousTarget: Int?
            var shouldAttach = true

            for member in members {
                let bestExternal = candidates[member]
                    .filter { updated[$0.neighbor] != label && $0.similarity >= attachThreshold }
                    .max(by: { $0.similarity < $1.similarity })

                guard let bestExternal else {
                    shouldAttach = false
                    break
                }

                let target = updated[bestExternal.neighbor]
                guard clusterSizes[target, default: 0] > members.count else {
                    shouldAttach = false
                    break
                }

                if let unanimousTarget, unanimousTarget != target {
                    shouldAttach = false
                    break
                }

                unanimousTarget = target
            }

            guard shouldAttach, let target = unanimousTarget else { continue }
            for member in members {
                updated[member] = target
            }
        }

        return normalizedLabels(updated)
    }

    nonisolated
    private static func mergeConsistentClusters(
        labels: [Int],
        candidates: [[NeighborCandidate]],
        embeddings: [[Float]],
        threshold: Float,
        progress: (@Sendable (Int, Int, [Int]) async -> Void)? = nil
    ) async -> [Int] {
        guard !labels.isEmpty else { return [] }
        guard let dim = embeddings.first?.count, dim > 0 else { return labels }

        var updated = labels
        let centroidThreshold = max(0.32, threshold)
        let bridgeThreshold = max(0.24, threshold - 0.08)
        let maxPasses = 4

        for passIndex in 0..<maxPasses {
            let membersByLabel = Dictionary(grouping: Array(updated.indices), by: { updated[$0] })
            let clusterSizes = membersByLabel.mapValues(\.count)
            let centroids = membersByLabel.mapValues { indices in
                computeNormalizedCentroid(
                    embeddings: embeddings,
                    indices: indices,
                    dim: dim
                )
            }

            var plannedMerges: [Int: Int] = [:]

            for (label, members) in membersByLabel.sorted(by: { lhs, rhs in
                if lhs.value.count == rhs.value.count {
                    return lhs.key < rhs.key
                }
                return lhs.value.count < rhs.value.count
            }) {
                guard plannedMerges[label] == nil else { continue }

                var supportByTarget: [Int: Int] = [:]
                var bestBridgeByTarget: [Int: Float] = [:]

                for member in members {
                    guard let bestExternal = candidates[member]
                        .filter({ updated[$0.neighbor] != label && $0.similarity >= bridgeThreshold })
                        .max(by: { $0.similarity < $1.similarity }) else {
                        continue
                    }

                    let target = updated[bestExternal.neighbor]
                    supportByTarget[target, default: 0] += 1
                    bestBridgeByTarget[target] = max(
                        bestBridgeByTarget[target] ?? 0,
                        bestExternal.similarity
                    )
                }

                guard let bestTarget = supportByTarget.max(by: { lhs, rhs in
                    if lhs.value == rhs.value {
                        return (bestBridgeByTarget[lhs.key] ?? 0) < (bestBridgeByTarget[rhs.key] ?? 0)
                    }
                    return lhs.value < rhs.value
                }) else {
                    continue
                }

                let target = bestTarget.key
                let sourceSupport = bestTarget.value
                guard target != label else { continue }
                guard plannedMerges[target] == nil else { continue }
                guard clusterSizes[target, default: 0] >= members.count else { continue }

                let requiredSourceSupport = max(2, Int(ceil(Double(members.count) * 0.30)))
                guard sourceSupport >= requiredSourceSupport else { continue }

                let reverseSupport = reverseBridgeSupport(
                    members: membersByLabel[target] ?? [],
                    sourceLabel: label,
                    currentLabel: target,
                    labels: updated,
                    candidates: candidates,
                    minimumSimilarity: bridgeThreshold
                )
                let requiredReverseSupport = (clusterSizes[target, default: 0] >= 4) ? 2 : 1
                guard reverseSupport >= requiredReverseSupport else { continue }

                let centroidSimilarity = dot(
                    centroids[label] ?? [],
                    centroids[target] ?? []
                )
                let bestBridge = bestBridgeByTarget[target] ?? 0
                guard centroidSimilarity >= centroidThreshold else { continue }
                guard bestBridge >= max(bridgeThreshold, threshold - 0.02) else { continue }

                plannedMerges[label] = target
            }

            guard !plannedMerges.isEmpty else { break }

            for (source, target) in plannedMerges {
                for index in updated.indices where updated[index] == source {
                    updated[index] = target
                }
            }
            updated = normalizedLabels(updated)

            if let progress {
                await progress(passIndex + 1, maxPasses, updated)
            }
        }

        return normalizedLabels(updated)
    }

    nonisolated
    private static func reverseBridgeSupport(
        members: [Int],
        sourceLabel: Int,
        currentLabel: Int,
        labels: [Int],
        candidates: [[NeighborCandidate]],
        minimumSimilarity: Float
    ) -> Int {
        var support = 0

        for member in members {
            guard let bestExternal = candidates[member]
                .filter({ labels[$0.neighbor] != currentLabel && $0.similarity >= minimumSimilarity })
                .max(by: { $0.similarity < $1.similarity }) else {
                continue
            }

            if labels[bestExternal.neighbor] == sourceLabel {
                support += 1
            }
        }

        return support
    }

    nonisolated
    private static func dot(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count, !lhs.isEmpty else { return 0 }
        var result: Float = 0
        for (left, right) in zip(lhs, rhs) {
            result += left * right
        }
        return result
    }

    nonisolated
    private static func sequentialAssignments(from labels: [Int]) -> [Int] {
        var map: [Int: Int] = [:]
        var nextId = 0
        return labels.map { label in
            if let existing = map[label] {
                return existing
            }
            map[label] = nextId
            defer { nextId += 1 }
            return map[label]!
        }
    }

    nonisolated
    private static func normalizedLabels(_ labels: [Int]) -> [Int] {
        sequentialAssignments(from: labels)
    }

    nonisolated
    private static func hasNonSingletonCluster(_ labels: [Int]) -> Bool {
        let counts = Dictionary(grouping: labels, by: { $0 }).mapValues(\.count)
        return counts.values.contains(where: { $0 > 1 })
    }

    nonisolated
    private static func buildClusters(
        faces: [FaceRecord],
        embeddings: [[Float]],
        personIds: [Int]
    ) -> [PersonCluster] {
        guard !faces.isEmpty else { return [] }

        let groups = Dictionary(grouping: Array(faces.indices), by: { personIds[$0] })

        return groups.map { personId, indices in
            let centroid = computeNormalizedCentroid(
                embeddings: embeddings,
                indices: indices,
                dim: embeddings.first?.count ?? 0
            )
            let sampleIndex = indices.max { lhs, rhs in
                let left = faces[lhs].boundingRect
                let right = faces[rhs].boundingRect
                return (left.width * left.height) < (right.width * right.height)
            } ?? indices[0]
            let sampleFace = faces[sampleIndex]

            return PersonCluster(
                id: personId,
                centroid: centroid,
                count: indices.count,
                sampleAssetId: sampleFace.assetId,
                sampleBoundingBox: sampleFace.boundingRect
            )
        }
        .sorted { lhs, rhs in
            if lhs.count == rhs.count {
                return lhs.id < rhs.id
            }
            return lhs.count > rhs.count
        }
    }

    nonisolated
    private static func previewClusters(
        faces: [FaceRecord],
        embeddings: [[Float]],
        labels: [Int]
    ) -> [PersonCluster] {
        let assignments = sequentialAssignments(from: labels)
        return buildClusters(
            faces: faces,
            embeddings: embeddings,
            personIds: assignments
        )
    }

    nonisolated
    private static func computeNormalizedCentroid(
        embeddings: [[Float]],
        indices: [Int],
        dim: Int
    ) -> [Float] {
        guard dim > 0, !indices.isEmpty else { return [] }

        var centroid = [Float](repeating: 0, count: dim)
        for index in indices {
            for component in 0..<dim {
                centroid[component] += embeddings[index][component]
            }
        }

        let scale = 1 / Float(indices.count)
        var mutableScale = scale
        vDSP_vsmul(centroid, 1, &mutableScale, &centroid, 1, vDSP_Length(dim))

        var norm: Float = 0
        vDSP_svesq(centroid, 1, &norm, vDSP_Length(dim))
        norm = sqrt(norm)
        if norm > 0 {
            vDSP_vsdiv(centroid, 1, &norm, &centroid, 1, vDSP_Length(dim))
        }

        return centroid
    }

    // MARK: - Database Helpers

    nonisolated
    private static func activeEmbeddingModelConfiguration() -> EmbeddingModelConfiguration {
        if Bundle.main.url(forResource: EmbeddingModelConfiguration.cvlFaceViTKPRPE.resourceName, withExtension: "mlmodelc") != nil {
            return .cvlFaceViTKPRPE
        }
        if Bundle.main.url(forResource: EmbeddingModelConfiguration.cvlFaceIR101.resourceName, withExtension: "mlmodelc") != nil {
            return .cvlFaceIR101
        }
        return .faceNet
    }

    nonisolated
    private static func clusterThresholdDefaultsKey(
        for configuration: EmbeddingModelConfiguration
    ) -> String {
        "clusterThreshold.v\(configuration.embeddingVersion)"
    }

    nonisolated
    private static func defaultSimilarityThreshold(
        for configuration: EmbeddingModelConfiguration
    ) -> Double {
        if configuration == .cvlFaceViTKPRPE {
            return 0.36
        }
        if configuration == .cvlFaceIR101 {
            return 0.40
        }
        return 0.58
    }

    nonisolated
    private static func clusteringSensitivityOptions(
        for configuration: EmbeddingModelConfiguration
    ) -> [(String, Double)] {
        if configuration == .cvlFaceViTKPRPE || configuration == .cvlFaceIR101 {
            return [
                ("Aggressive — fewest clusters", 0.28),
                ("Moderate", 0.32),
                ("Balanced (recommended)", 0.36),
                ("Conservative", 0.40),
                ("Strict", 0.44),
                ("Very Strict — most clusters", 0.48),
            ]
        }
        return [
            ("Aggressive — fewest clusters", 0.48),
            ("Moderate", 0.54),
            ("Balanced (recommended)", 0.58),
            ("Conservative", 0.62),
            ("Strict", 0.66),
            ("Very Strict — most clusters", 0.72),
        ]
    }

    nonisolated
    private static func prepareDatabase(
        _ database: FaceDatabase,
        forEmbeddingVersion embeddingVersion: Int
    ) throws {
        let versions = try database.availableEmbeddingVersions()
        guard !versions.isEmpty else { return }
        let uniqueVersions = Set(versions)
        if uniqueVersions != Set([embeddingVersion]) {
            _ = try database.deleteAll()
        }
    }

    nonisolated
    private static func preferredEmbeddingVersion(
        in database: FaceDatabase,
        preferredVersion: Int
    ) throws -> Int? {
        let versions = try database.availableEmbeddingVersions()
        if versions.contains(preferredVersion) {
            return preferredVersion
        }
        return versions.first
    }

    nonisolated
    private static func loadClusters(
        from database: FaceDatabase,
        embeddingVersion: Int
    ) throws -> [PersonCluster] {
        let allFaces = try database.allFaces(embeddingVersion: embeddingVersion)
        let clusteredFaces = allFaces.filter { $0.personId != nil }
        guard !clusteredFaces.isEmpty else { return [] }
        let embeddings = clusteredFaces.map(\.embeddingVector)
        let personIds = clusteredFaces.compactMap(\.personId)
        return buildClusters(faces: clusteredFaces, embeddings: embeddings, personIds: personIds)
    }

    // MARK: - Parallel Asset Processing

    nonisolated
    private static func processAsset(
        _ asset: PHAsset,
        pool: ModelPool,
        photoManager: PhotoManager,
        modelConfiguration: EmbeddingModelConfiguration
    ) async -> (String, [(CGRect, [Float])]) {
        let assetId = asset.localIdentifier

        guard let cgImage = await photoManager.loadFullImage(for: asset) else {
            return (assetId, [])
        }

        guard let faces = try? await detectFaces(in: cgImage), !faces.isEmpty else {
            return (assetId, [])
        }

        let imageWidth = CGFloat(cgImage.width)
        let imageHeight = CGFloat(cgImage.height)
        let minFacePixels: CGFloat = 40
        let qualityFaces = faces.filter { observation in
            let faceWidth = observation.boundingBox.width * imageWidth
            let faceHeight = observation.boundingBox.height * imageHeight
            return faceWidth >= minFacePixels
                && faceHeight >= minFacePixels
                && observation.confidence >= 0.5
        }
        guard !qualityFaces.isEmpty else { return (assetId, []) }

        let model = await pool.borrow()
        var results: [(CGRect, [Float])] = []
        for observation in qualityFaces {
            if let embedding = try? await computeEmbedding(
                face: observation,
                image: cgImage,
                model: model,
                modelConfiguration: modelConfiguration
            ) {
                results.append((observation.boundingBox, embedding))
            }
        }
        await pool.release(model)

        return (assetId, results)
    }

    // MARK: - Face Detection

    nonisolated
    private static func detectFaces(in image: CGImage) async throws -> [VNFaceObservation] {
        try await withCheckedThrowingContinuation { continuation in
            let request = VNDetectFaceRectanglesRequest { request, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                continuation.resume(returning: request.results as? [VNFaceObservation] ?? [])
            }
            let handler = VNImageRequestHandler(cgImage: image, options: [:])
            do {
                try handler.perform([request])
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }

    nonisolated
    private static func detectLandmarks(
        for observation: VNFaceObservation,
        in image: CGImage
    ) async throws -> VNFaceObservation? {
        try await withCheckedThrowingContinuation { continuation in
            let request = VNDetectFaceLandmarksRequest { request, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                let observations = request.results as? [VNFaceObservation] ?? []
                continuation.resume(returning: observations.first)
            }
            request.regionOfInterest = observation.boundingBox
            let handler = VNImageRequestHandler(cgImage: image, options: [:])
            do {
                try handler.perform([request])
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }

    // MARK: - Embedding Computation

    nonisolated
    private static func computeEmbedding(
        face: VNFaceObservation,
        image: CGImage,
        model: MLModel,
        modelConfiguration: EmbeddingModelConfiguration
    ) async throws -> [Float]? {
        let preparedFace = try await prepareFaceInput(
            face: face,
            image: image,
            modelConfiguration: modelConfiguration
        )
        guard let preparedFace else { return nil }

        let pixelBuffer = try makePixelBuffer(
            from: preparedFace.image,
            size: modelConfiguration.inputSize
        )
        var features: [String: Any] = [
            modelConfiguration.imageInputName: MLFeatureValue(pixelBuffer: pixelBuffer)
        ]

        if let keypointInputName = modelConfiguration.keypointInputName {
            guard let normalizedKeypoints = preparedFace.normalizedKeypoints,
                  let keypointArray = try? makeKeypointArray(from: normalizedKeypoints) else {
                return nil
            }
            features[keypointInputName] = MLFeatureValue(multiArray: keypointArray)
        }

        let inputProvider = try MLDictionaryFeatureProvider(dictionary: features)
        let prediction = try await model.prediction(from: inputProvider)
        let featureValue = prediction.featureValue(for: modelConfiguration.outputName)
            ?? prediction.featureNames.sorted().compactMap { prediction.featureValue(for: $0) }.first
        guard let multiArray = featureValue?.multiArrayValue else { return nil }

        let count = multiArray.count
        var embedding = [Float](repeating: 0, count: count)
        let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        for index in 0..<count {
            embedding[index] = ptr[index]
        }

        var norm: Float = 0
        vDSP_svesq(embedding, 1, &norm, vDSP_Length(count))
        norm = sqrt(norm)
        if norm > 0 {
            vDSP_vsdiv(embedding, 1, &norm, &embedding, 1, vDSP_Length(count))
        }

        return embedding
    }

    nonisolated
    private static func prepareFaceInput(
        face: VNFaceObservation,
        image: CGImage,
        modelConfiguration: EmbeddingModelConfiguration
    ) async throws -> PreparedFaceInput? {
        let landmarkedObservation = (try? await detectLandmarks(for: face, in: image)) ?? face
        let imageSize = CGSize(width: image.width, height: image.height)
        let faceRect = imageRect(for: face.boundingBox, imageSize: imageSize)
        let keypoints = fivePointLandmarks(for: landmarkedObservation, imageSize: imageSize)
        let cropRect = alignedCropRect(faceRect: faceRect, keypoints: keypoints, imageSize: imageSize)
            ?? cropRect(for: faceRect, imageSize: imageSize)

        guard let cropRect, let croppedImage = image.cropping(to: cropRect) else { return nil }

        let normalizedKeypoints: [CGPoint]?
        if modelConfiguration.requiresKeypoints {
            normalizedKeypoints = keypoints.map { point in
                normalize(point: point, in: cropRect)
            }
        } else {
            normalizedKeypoints = nil
        }

        return PreparedFaceInput(
            image: croppedImage,
            normalizedKeypoints: normalizedKeypoints
        )
    }

    nonisolated
    private static func makePixelBuffer(
        from image: CGImage,
        size: Int
    ) throws -> CVPixelBuffer {
        let attributes: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            size,
            size,
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer else {
            throw NSError(domain: "FaceProcessor", code: Int(status))
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw NSError(domain: "FaceProcessor", code: -1)
        }

        guard let context = CGContext(
            data: baseAddress,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else {
            throw NSError(domain: "FaceProcessor", code: -2)
        }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
        return pixelBuffer
    }

    nonisolated
    private static func makeKeypointArray(from keypoints: [CGPoint]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, 5, 2], dataType: .float32)
        for (index, point) in keypoints.prefix(5).enumerated() {
            array[[NSNumber(value: 0), NSNumber(value: index), NSNumber(value: 0)]] = NSNumber(value: Float(point.x))
            array[[NSNumber(value: 0), NSNumber(value: index), NSNumber(value: 1)]] = NSNumber(value: Float(point.y))
        }
        return array
    }

    nonisolated
    private static func alignedCropRect(
        faceRect: CGRect,
        keypoints: [CGPoint],
        imageSize: CGSize
    ) -> CGRect? {
        guard keypoints.count >= 3 else {
            return cropRect(for: faceRect, imageSize: imageSize)
        }

        let leftEye = keypoints[0]
        let rightEye = keypoints[1]
        let nose = keypoints[2]
        let eyeMidpoint = CGPoint(
            x: (leftEye.x + rightEye.x) * 0.5,
            y: (leftEye.y + rightEye.y) * 0.5
        )
        let centerY = (eyeMidpoint.y * 0.55) + (nose.y * 0.45)
        let center = CGPoint(x: eyeMidpoint.x, y: centerY)
        let eyeDistance = hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y)
        let side = max(faceRect.width, faceRect.height, eyeDistance * 2.8)
        let desiredRect = CGRect(
            x: center.x - side * 0.5,
            y: center.y - side * 0.45,
            width: side,
            height: side
        )
        return clamp(rect: desiredRect, to: imageSize)
    }

    nonisolated
    private static func cropFace(observation: VNFaceObservation, from image: CGImage) -> CGImage? {
        let imageSize = CGSize(width: image.width, height: image.height)
        let faceRect = imageRect(for: observation.boundingBox, imageSize: imageSize)
        guard let cropRect = cropRect(for: faceRect, imageSize: imageSize) else { return nil }
        return image.cropping(to: cropRect)
    }

    nonisolated
    private static func cropRect(for faceRect: CGRect, imageSize: CGSize) -> CGRect? {
        let padding: CGFloat = 0.3
        let width = faceRect.width * (1 + padding * 2)
        let height = faceRect.height * (1 + padding * 2)
        let origin = CGPoint(
            x: faceRect.minX - faceRect.width * padding,
            y: faceRect.minY - faceRect.height * padding
        )
        return clamp(rect: CGRect(origin: origin, size: CGSize(width: width, height: height)), to: imageSize)
    }

    nonisolated
    private static func imageRect(for boundingBox: CGRect, imageSize: CGSize) -> CGRect {
        CGRect(
            x: boundingBox.origin.x * imageSize.width,
            y: (1 - boundingBox.origin.y - boundingBox.height) * imageSize.height,
            width: boundingBox.width * imageSize.width,
            height: boundingBox.height * imageSize.height
        )
    }

    nonisolated
    private static func fivePointLandmarks(
        for observation: VNFaceObservation,
        imageSize: CGSize
    ) -> [CGPoint] {
        let faceRect = imageRect(for: observation.boundingBox, imageSize: imageSize)
        let leftEye = averagePoint(for: observation.landmarks?.leftEye?.normalizedPoints, in: observation, imageSize: imageSize)
            ?? CGPoint(x: faceRect.minX + faceRect.width * 0.32, y: faceRect.minY + faceRect.height * 0.38)
        let rightEye = averagePoint(for: observation.landmarks?.rightEye?.normalizedPoints, in: observation, imageSize: imageSize)
            ?? CGPoint(x: faceRect.minX + faceRect.width * 0.68, y: faceRect.minY + faceRect.height * 0.38)
        let nose = averagePoint(for: observation.landmarks?.nose?.normalizedPoints, in: observation, imageSize: imageSize)
            ?? CGPoint(x: faceRect.midX, y: faceRect.minY + faceRect.height * 0.58)
        let mouthCorners = mouthCorners(for: observation, imageSize: imageSize)
        let leftMouth = mouthCorners?.0
            ?? CGPoint(x: faceRect.minX + faceRect.width * 0.38, y: faceRect.minY + faceRect.height * 0.78)
        let rightMouth = mouthCorners?.1
            ?? CGPoint(x: faceRect.minX + faceRect.width * 0.62, y: faceRect.minY + faceRect.height * 0.78)

        return [leftEye, rightEye, nose, leftMouth, rightMouth]
    }

    nonisolated
    private static func mouthCorners(
        for observation: VNFaceObservation,
        imageSize: CGSize
    ) -> (CGPoint, CGPoint)? {
        let mouthPoints = observation.landmarks?.outerLips?.normalizedPoints
            ?? observation.landmarks?.innerLips?.normalizedPoints
        guard let mouthPoints, !mouthPoints.isEmpty else { return nil }

        let converted = mouthPoints.map { point in
            CGPoint(
                x: (observation.boundingBox.origin.x + point.x * observation.boundingBox.width) * imageSize.width,
                y: (1 - (observation.boundingBox.origin.y + point.y * observation.boundingBox.height)) * imageSize.height
            )
        }
        let sorted = converted.sorted { lhs, rhs in
            if lhs.x == rhs.x {
                return lhs.y < rhs.y
            }
            return lhs.x < rhs.x
        }
        let sampleCount = max(1, sorted.count / 4)
        let leftPoints = Array(sorted.prefix(sampleCount))
        let rightPoints = Array(sorted.suffix(sampleCount))
        let left = average(of: leftPoints)
        let right = average(of: rightPoints)
        return (left, right)
    }

    nonisolated
    private static func averagePoint(
        for normalizedPoints: [CGPoint]?,
        in observation: VNFaceObservation,
        imageSize: CGSize
    ) -> CGPoint? {
        guard let normalizedPoints, !normalizedPoints.isEmpty else { return nil }

        let converted = normalizedPoints.map { point in
            CGPoint(
                x: (observation.boundingBox.origin.x + point.x * observation.boundingBox.width) * imageSize.width,
                y: (1 - (observation.boundingBox.origin.y + point.y * observation.boundingBox.height)) * imageSize.height
            )
        }

        return average(of: converted)
    }

    nonisolated
    private static func average(of points: [CGPoint]) -> CGPoint {
        let sum = points.reduce(CGPoint.zero) { partial, point in
            CGPoint(x: partial.x + point.x, y: partial.y + point.y)
        }
        return CGPoint(
            x: sum.x / CGFloat(max(points.count, 1)),
            y: sum.y / CGFloat(max(points.count, 1))
        )
    }

    nonisolated
    private static func normalize(point: CGPoint, in rect: CGRect) -> CGPoint {
        CGPoint(
            x: min(1, max(0, (point.x - rect.minX) / rect.width)),
            y: min(1, max(0, (point.y - rect.minY) / rect.height))
        )
    }

    nonisolated
    private static func clamp(rect: CGRect, to imageSize: CGSize) -> CGRect? {
        let x = max(0, rect.origin.x)
        let y = max(0, rect.origin.y)
        let width = min(imageSize.width - x, rect.width)
        let height = min(imageSize.height - y, rect.height)
        guard width > 1, height > 1 else { return nil }
        return CGRect(x: x, y: y, width: width, height: height).integral
    }

    // MARK: - Helpers

    nonisolated
    private static func eta(elapsed: TimeInterval, completed: Int, total: Int) -> Double? {
        guard completed > 0, total > completed, elapsed > 0 else { return nil }
        let rate = elapsed / Double(completed)
        return rate * Double(total - completed)
    }
}
