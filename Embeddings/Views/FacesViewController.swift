import UIKit

// MARK: - Face Item (diffable data source identity)

private enum FaceItem: Hashable {
    case shimmer(Int)
    case cluster(Int)
    case detectedFace(Int64)
}

// MARK: - Shimmer Cell

private final class ShimmerCell: UICollectionViewCell {
    static let reuseId = "ShimmerCell"
    private let shimmerLayer = CAGradientLayer()

    override init(frame: CGRect) {
        super.init(frame: frame)
        contentView.backgroundColor = .secondarySystemFill
        contentView.clipsToBounds = true

        shimmerLayer.colors = [
            UIColor.secondarySystemFill.cgColor,
            UIColor.tertiarySystemFill.cgColor,
            UIColor.secondarySystemFill.cgColor,
        ]
        shimmerLayer.locations = [0, 0.5, 1]
        shimmerLayer.startPoint = CGPoint(x: 0, y: 0.5)
        shimmerLayer.endPoint = CGPoint(x: 1, y: 0.5)
        contentView.layer.addSublayer(shimmerLayer)
    }

    required init?(coder: NSCoder) { fatalError() }

    override func layoutSubviews() {
        super.layoutSubviews()
        shimmerLayer.frame = CGRect(
            x: -contentView.bounds.width, y: 0,
            width: contentView.bounds.width * 3, height: contentView.bounds.height
        )
    }

    func startAnimating() {
        guard shimmerLayer.animation(forKey: "shimmer") == nil else { return }
        let animation = CABasicAnimation(keyPath: "locations")
        animation.fromValue = [-1.0, -0.5, 0.0] as [NSNumber]
        animation.toValue = [1.0, 1.5, 2.0] as [NSNumber]
        animation.duration = 1.5
        animation.repeatCount = .infinity
        shimmerLayer.add(animation, forKey: "shimmer")
    }
}

// MARK: - Faces View Controller

final class FacesViewController: UIViewController {
    private enum Constants {
        static let columns: CGFloat = 3
        static let spacing: CGFloat = 2
        static let inset: CGFloat = 2
    }

    private struct ClusterDigest: Equatable {
        let id: Int
        let count: Int
        let sampleAssetId: String
        let sampleBoundingBox: CGRect
    }

    private struct ObservationSignature: Equatable {
        let clusters: [ClusterDigest]
        let isProcessing: Bool
        let processed: Int
        let total: Int
        let detectedFaces: Int
        let status: ProcessingStatus
        let scannedFaceIds: [Int64]
    }

    private struct MergeAnimationSource {
        let snapshotView: UIView
        let item: FaceItem
        let oldIndex: Int
        let startFrame: CGRect
    }

    private let faceProcessor: FaceProcessor
    private let photoManager: PhotoManager

    private var collectionView: UICollectionView!
    private var dataSource: UICollectionViewDiffableDataSource<Int, FaceItem>!
    private var pipelineHeaderView: ProcessingPipelineHeaderView!
    private var emptyLabel: UILabel!

    private var personNames: [Int: String] = [:]
    private var clusterById: [Int: PersonCluster] = [:]
    private var scannedFaceById: [Int64: ScannedFace] = [:]
    private var lastObservationSignature: ObservationSignature?
    private var observationTask: Task<Void, Never>?
    private var lastMergeAnimationTime: CFTimeInterval = 0

    private var gridItemWidth: CGFloat {
        let totalSpacing = Constants.spacing * (Constants.columns - 1) + Constants.inset * 2
        let availableWidth = max(view.bounds.width, 320)
        return floor((availableWidth - totalSpacing) / Constants.columns)
    }

    init(faceProcessor: FaceProcessor, photoManager: PhotoManager) {
        self.faceProcessor = faceProcessor
        self.photoManager = photoManager
        super.init(nibName: nil, bundle: nil)
        title = "Faces"
    }

    required init?(coder: NSCoder) { fatalError() }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground
        setupNavBar()
        setupHeader()
        setupCollectionView()
        setupEmptyState()
        setupDataSource()
        loadData()
        startObserving()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        updateLayoutMetrics()
    }

    deinit { observationTask?.cancel() }

    // MARK: - Nav Bar

    private func setupNavBar() {
        let scanButton = UIBarButtonItem(
            image: UIImage(systemName: "faceid"), style: .plain,
            target: self, action: #selector(startScan)
        )
        let gearButton = UIBarButtonItem(
            image: UIImage(systemName: "gearshape"), style: .plain,
            target: self, action: #selector(showSettings)
        )
        navigationItem.rightBarButtonItems = [gearButton, scanButton]
    }

    @objc private func startScan() {
        faceProcessor.startProcessing()
    }

    @objc private func showSettings() {
        let alert = UIAlertController(
            title: "Settings",
            message: nil,
            preferredStyle: .actionSheet
        )

        alert.addAction(UIAlertAction(title: "Change Clustering Sensitivity", style: .default) { [weak self] _ in
            self?.showThresholdPicker()
        })

        alert.addAction(UIAlertAction(title: "Recluster Now", style: .default) { [weak self] _ in
            self?.faceProcessor.reclusterExisting()
        })

        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
        present(alert, animated: true)
    }

    private func showThresholdPicker() {
        let alert = UIAlertController(
            title: "Clustering Sensitivity",
            message: "Controls how aggressively faces are grouped as the same person for the current embedding model. Tap \"Recluster Now\" after changing to apply.",
            preferredStyle: .actionSheet
        )
        let current = faceProcessor.similarityThreshold
        let options = faceProcessor.clusteringSensitivityOptions

        for (label, value) in options {
            let isCurrent = abs(value - current) < 0.01
            let title = isCurrent ? "\(label) \u{2713}" : label
            alert.addAction(UIAlertAction(title: title, style: .default) { [weak self] _ in
                self?.faceProcessor.similarityThreshold = value
            })
        }
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
        present(alert, animated: true)
    }

    // MARK: - Header (progress)

    private func setupHeader() {
        pipelineHeaderView = ProcessingPipelineHeaderView()
        pipelineHeaderView.isHidden = true
        view.addSubview(pipelineHeaderView)

        NSLayoutConstraint.activate([
            pipelineHeaderView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 8),
            pipelineHeaderView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            pipelineHeaderView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),
        ])
    }

    // MARK: - Collection View

    private func setupCollectionView() {
        let layout = UICollectionViewFlowLayout()
        layout.itemSize = CGSize(width: gridItemWidth, height: gridItemWidth)
        layout.minimumInteritemSpacing = Constants.spacing
        layout.minimumLineSpacing = Constants.spacing
        layout.sectionInset = UIEdgeInsets(
            top: Constants.inset,
            left: Constants.inset,
            bottom: Constants.inset,
            right: Constants.inset
        )

        collectionView = UICollectionView(frame: .zero, collectionViewLayout: layout)
        collectionView.translatesAutoresizingMaskIntoConstraints = false
        collectionView.backgroundColor = .systemBackground
        collectionView.delegate = self
        collectionView.register(FaceClusterCell.self, forCellWithReuseIdentifier: FaceClusterCell.reuseId)
        collectionView.register(ShimmerCell.self, forCellWithReuseIdentifier: ShimmerCell.reuseId)
        view.addSubview(collectionView)

        NSLayoutConstraint.activate([
            collectionView.topAnchor.constraint(equalTo: pipelineHeaderView.bottomAnchor, constant: 10),
            collectionView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            collectionView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            collectionView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])
    }

    private func updateLayoutMetrics() {
        guard let layout = collectionView.collectionViewLayout as? UICollectionViewFlowLayout else { return }
        let updatedSize = CGSize(width: gridItemWidth, height: gridItemWidth)
        guard layout.itemSize != updatedSize else { return }
        layout.itemSize = updatedSize
        layout.invalidateLayout()
    }

    private func setupEmptyState() {
        emptyLabel = UILabel()
        emptyLabel.text = "Scan your photo library to detect and cluster faces."
        emptyLabel.textColor = .secondaryLabel
        emptyLabel.textAlignment = .center
        emptyLabel.numberOfLines = 0
        emptyLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(emptyLabel)

        NSLayoutConstraint.activate([
            emptyLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            emptyLabel.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            emptyLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 40),
            emptyLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -40),
        ])
    }

    private func setupDataSource() {
        dataSource = UICollectionViewDiffableDataSource<Int, FaceItem>(
            collectionView: collectionView
        ) { [weak self] collectionView, indexPath, item in
            guard let self else {
                return collectionView.dequeueReusableCell(
                    withReuseIdentifier: ShimmerCell.reuseId, for: indexPath
                )
            }

            switch item {
            case .shimmer:
                let cell = collectionView.dequeueReusableCell(
                    withReuseIdentifier: ShimmerCell.reuseId, for: indexPath
                ) as! ShimmerCell
                cell.startAnimating()
                return cell

            case .detectedFace(let faceId):
                let cell = collectionView.dequeueReusableCell(
                    withReuseIdentifier: FaceClusterCell.reuseId, for: indexPath
                ) as! FaceClusterCell
                cell.configureDetectedFacePreview()
                guard let face = self.scannedFaceById[faceId] else { return cell }

                self.loadFaceImage(
                    into: cell,
                    expectedItem: .detectedFace(faceId),
                    assetId: face.assetId,
                    boundingBox: face.boundingBox,
                    cacheKey: "scannedFace-\(faceId)"
                )
                return cell

            case .cluster(let clusterId):
                let cell = collectionView.dequeueReusableCell(
                    withReuseIdentifier: FaceClusterCell.reuseId, for: indexPath
                ) as! FaceClusterCell

                guard let cluster = self.clusterById[clusterId] else { return cell }

                let name = self.personNames[clusterId] ?? "Person \(clusterId)"
                cell.configureCluster(name: name, count: cluster.count)
                cell.clusterId = clusterId

                self.loadFaceImage(
                    into: cell,
                    expectedItem: .cluster(clusterId),
                    assetId: cluster.sampleAssetId,
                    boundingBox: cluster.sampleBoundingBox,
                    cacheKey: "face-\(clusterId)-\(cluster.sampleAssetId)"
                )

                return cell
            }
        }
    }

    // MARK: - Data Loading

    private func loadData() {
        faceProcessor.loadExistingClusters()
        loadNames()
    }

    private func loadNames() {
        personNames = (try? FaceDatabase().allNames()) ?? [:]
    }

    // MARK: - Observation

    private func startObserving() {
        observationTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let snapshot = withObservationTracking {
                    ObservationSignature(
                        clusters: self.faceProcessor.clusters.map {
                            ClusterDigest(
                                id: $0.id,
                                count: $0.count,
                                sampleAssetId: $0.sampleAssetId,
                                sampleBoundingBox: $0.sampleBoundingBox
                            )
                        },
                        isProcessing: self.faceProcessor.isProcessing,
                        processed: self.faceProcessor.processedCount,
                        total: self.faceProcessor.totalCount,
                        detectedFaces: self.faceProcessor.detectedFaceCount,
                        status: self.faceProcessor.status,
                        scannedFaceIds: self.faceProcessor.scannedFaces.map(\.id)
                    )
                } onChange: { }

                await MainActor.run {
                    guard snapshot != self.lastObservationSignature else { return }
                    self.lastObservationSignature = snapshot
                    self.updateUI(
                        clusters: self.faceProcessor.clusters,
                        isProcessing: snapshot.isProcessing,
                        processed: snapshot.processed,
                        total: snapshot.total,
                        detectedFaces: snapshot.detectedFaces,
                        status: snapshot.status,
                        scannedFaces: self.faceProcessor.scannedFaces
                    )
                }
                try? await Task.sleep(for: .milliseconds(120))
            }
        }
    }

    private func updateUI(
        clusters: [PersonCluster], isProcessing: Bool,
        processed: Int, total: Int, detectedFaces: Int,
        status: ProcessingStatus,
        scannedFaces: [ScannedFace] = []
    ) {
        let phase = status.phase
        clusterById = Dictionary(uniqueKeysWithValues: clusters.map { ($0.id, $0) })
        scannedFaceById = Dictionary(uniqueKeysWithValues: scannedFaces.map { ($0.id, $0) })

        pipelineHeaderView.apply(
            buildHeaderPresentation(
                phase: phase,
                processed: processed,
                total: total,
                detectedFaces: detectedFaces,
                clusters: clusters,
                status: status
            )
        )

        emptyLabel.isHidden = isProcessing || !clusters.isEmpty

        let items = buildSnapshotItems(
            phase: phase,
            detectedFaces: detectedFaces,
            clusters: clusters,
            scannedFaces: scannedFaces
        )
        let mergeAnimationSources = prepareMergeAnimation(
            from: dataSource.snapshot().itemIdentifiers,
            to: items,
            phase: phase,
            status: status
        )
        var snapshot = NSDiffableDataSourceSnapshot<Int, FaceItem>()
        snapshot.appendSections([0])
        snapshot.appendItems(items)
        dataSource.apply(
            snapshot,
            animatingDifferences: shouldAnimateSnapshot(for: phase)
        ) { [weak self] in
            self?.runMergeAnimations(mergeAnimationSources, newItems: items)
        }

        // Disable scan button while processing (scan is second item)
        navigationItem.rightBarButtonItems?[1].isEnabled = phase == .idle
    }

    private func loadFaceImage(
        into cell: FaceClusterCell,
        expectedItem: FaceItem,
        assetId: String,
        boundingBox: CGRect,
        cacheKey: String
    ) {
        Task {
            let image = await FaceThumbnailCache.shared.loadFaceThumbnail(
                assetId: assetId,
                boundingBox: boundingBox,
                cacheKey: cacheKey
            )
            await MainActor.run {
                guard let visibleIndexPath = collectionView.indexPath(for: cell),
                      dataSource.itemIdentifier(for: visibleIndexPath) == expectedItem else {
                    return
                }
                cell.setFaceImage(image)
            }
        }
    }

    private func buildSnapshotItems(
        phase: ProcessingPhase,
        detectedFaces: Int,
        clusters: [PersonCluster],
        scannedFaces: [ScannedFace]
    ) -> [FaceItem] {
        let liveFaceItems = recentScannedFaceItems(for: phase, scannedFaces: scannedFaces)
        let clusterItems = clusters
            .sorted { lhs, rhs in
                if lhs.count == rhs.count {
                    return lhs.id < rhs.id
                }
                return lhs.count > rhs.count
            }
            .map { FaceItem.cluster($0.id) }

        if !liveFaceItems.isEmpty || !clusterItems.isEmpty {
            return liveFaceItems + clusterItems
        }

        guard phase != .idle else { return [] }
        let shimmerCount = min(max(detectedFaces, 6), 18)
        return (0..<shimmerCount).map { FaceItem.shimmer($0) }
    }

    private func recentScannedFaceItems(
        for phase: ProcessingPhase,
        scannedFaces: [ScannedFace]
    ) -> [FaceItem] {
        guard phase != .idle else { return [] }
        return Array(scannedFaces.suffix(12).reversed()).map { FaceItem.detectedFace($0.id) }
    }

    private func shouldAnimateSnapshot(for phase: ProcessingPhase) -> Bool {
        phase != .scanning
    }

    private func prepareMergeAnimation(
        from oldItems: [FaceItem],
        to newItems: [FaceItem],
        phase: ProcessingPhase,
        status: ProcessingStatus
    ) -> [MergeAnimationSource] {
        guard phase == .clustering, status.isProvisional else { return [] }
        let oldClusterItems = oldItems.compactMap { item -> FaceItem? in
            if case .cluster = item { return item }
            return nil
        }
        let newClusterItems = newItems.compactMap { item -> FaceItem? in
            if case .cluster = item { return item }
            return nil
        }
        guard newClusterItems.count < oldClusterItems.count else { return [] }

        let now = CACurrentMediaTime()
        guard now - lastMergeAnimationTime > 0.22 else { return [] }

        let visibleSources = collectionView.indexPathsForVisibleItems
            .sorted()
            .compactMap { indexPath -> MergeAnimationSource? in
                guard let item = dataSource.itemIdentifier(for: indexPath),
                      case .cluster = item,
                      let cell = collectionView.cellForItem(at: indexPath),
                      let snapshotView = cell.snapshotView(afterScreenUpdates: false),
                      let oldIndex = oldClusterItems.firstIndex(of: item) else {
                    return nil
                }

                return MergeAnimationSource(
                    snapshotView: snapshotView,
                    item: item,
                    oldIndex: oldIndex,
                    startFrame: view.convert(cell.frame, from: collectionView)
                )
            }

        guard !visibleSources.isEmpty else { return [] }

        let newSet = Set(newClusterItems)
        let disappearingSources = visibleSources.filter { !newSet.contains($0.item) }
        let candidates = Array(
            (disappearingSources.isEmpty
             ? visibleSources.suffix(min(4, max(1, oldClusterItems.count - newClusterItems.count)))
             : disappearingSources.prefix(4))
        )
        guard !candidates.isEmpty else { return [] }

        lastMergeAnimationTime = now
        return candidates
    }

    private func runMergeAnimations(
        _ sources: [MergeAnimationSource],
        newItems: [FaceItem]
    ) {
        guard !sources.isEmpty else { return }

        let newClusterItems = newItems.compactMap { item -> FaceItem? in
            if case .cluster = item { return item }
            return nil
        }
        guard !newClusterItems.isEmpty else { return }

        let visibleTargetFrames = collectionView.indexPathsForVisibleItems
            .sorted()
            .compactMap { indexPath -> (FaceItem, CGRect)? in
                guard let item = dataSource.itemIdentifier(for: indexPath),
                      case .cluster = item,
                      let cell = collectionView.cellForItem(at: indexPath) else {
                    return nil
                }
                return (item, view.convert(cell.frame, from: collectionView))
            }
        let targetFrameByItem = Dictionary(uniqueKeysWithValues: visibleTargetFrames)

        for (offset, source) in sources.enumerated() {
            let targetIndex = min(source.oldIndex, newClusterItems.count - 1)
            let targetItem = newClusterItems[targetIndex]
            let targetFrame = targetFrameByItem[targetItem] ?? source.startFrame.offsetBy(dx: 0, dy: -24)

            let snapshotView = source.snapshotView
            snapshotView.frame = source.startFrame
            view.addSubview(snapshotView)

            if let targetCellIndexPath = dataSource.indexPath(for: targetItem),
               let targetCell = collectionView.cellForItem(at: targetCellIndexPath) {
                targetCell.transform = CGAffineTransform(scaleX: 0.94, y: 0.94)
                UIView.animate(
                    withDuration: 0.28,
                    delay: 0.02 + Double(offset) * 0.04,
                    usingSpringWithDamping: 0.72,
                    initialSpringVelocity: 0.2,
                    options: [.beginFromCurrentState, .allowUserInteraction]
                ) {
                    targetCell.transform = .identity
                }
            }

            UIView.animate(
                withDuration: 0.38,
                delay: Double(offset) * 0.04,
                options: [.curveEaseInOut, .beginFromCurrentState]
            ) {
                snapshotView.frame = targetFrame.insetBy(
                    dx: targetFrame.width * 0.18,
                    dy: targetFrame.height * 0.18
                )
                snapshotView.alpha = 0
                snapshotView.transform = CGAffineTransform(scaleX: 0.28, y: 0.28)
            } completion: { _ in
                snapshotView.removeFromSuperview()
            }
        }
    }

    private func buildHeaderPresentation(
        phase: ProcessingPhase,
        processed: Int,
        total: Int,
        detectedFaces: Int,
        clusters: [PersonCluster],
        status: ProcessingStatus
    ) -> ProcessingPipelineHeaderPresentation {
        let steps = ProcessingPipelineStep.allCases.map { step in
            ProcessingPipelineStepState(
                title: step.title,
                detail: detailText(
                    for: step,
                    phase: phase,
                    processed: processed,
                    total: total,
                    detectedFaces: detectedFaces,
                    status: status
                ),
                progress: progressValue(for: step, phase: phase, status: status, processed: processed, total: total, hasClusters: !clusters.isEmpty),
                tintColor: step.tintColor,
                isDimmed: isDimmed(step: step, for: phase, hasClusters: !clusters.isEmpty)
            )
        }

        let badgeText = status.isProvisional ? "Preview" : nil
        let summaryText: String?
        if phase == .idle, !clusters.isEmpty {
            summaryText = status.statusText
        } else if phase == .scanning || phase == .buildingGraph,
                  let eta = status.etaText {
            summaryText = "ETA \(eta)"
        } else {
            summaryText = nil
        }

        return ProcessingPipelineHeaderPresentation(
            isHidden: phase == .idle && clusters.isEmpty,
            steps: steps,
            badgeText: badgeText,
            summaryText: summaryText
        )
    }

    private func detailText(
        for step: ProcessingPipelineStep,
        phase: ProcessingPhase,
        processed: Int,
        total: Int,
        detectedFaces: Int,
        status: ProcessingStatus
    ) -> String? {
        switch step {
        case .scan:
            if phase == .scanning {
                return "\(processed)/\(max(total, 1)) • \(detectedFaces) faces"
            }
            return hasCompleted(step: .scan, for: phase) ? "Done" : nil

        case .graph:
            switch phase {
            case .loadingEmbeddings:
                return "\(detectedFaces) faces"
            case .buildingGraph:
                return [status.statusText, status.etaText.map { "ETA \($0)" }]
                    .compactMap { $0 }
                    .joined(separator: " • ")
            default:
                return hasCompleted(step: .graph, for: phase) ? "Done" : nil
            }

        case .merge:
            if phase == .clustering {
                return status.statusText
            }
            return hasCompleted(step: .merge, for: phase) ? "Done" : nil

        case .save:
            if phase == .finalizing {
                return "Writing"
            }
            return hasCompleted(step: .save, for: phase) ? "Done" : nil
        }
    }

    private func progressValue(
        for step: ProcessingPipelineStep,
        phase: ProcessingPhase,
        status: ProcessingStatus,
        processed: Int,
        total: Int,
        hasClusters: Bool
    ) -> Float {
        switch step {
        case .scan:
            if phase == .scanning {
                return total > 0 ? Float(processed) / Float(total) : 0
            }
            return hasCompleted(step: .scan, for: phase, hasClusters: hasClusters) ? 1 : 0

        case .graph:
            switch phase {
            case .loadingEmbeddings:
                return 0.05
            case .buildingGraph:
                return status.fractionCompleted
            default:
                return hasCompleted(step: .graph, for: phase, hasClusters: hasClusters) ? 1 : 0
            }

        case .merge:
            if phase == .clustering {
                return status.fractionCompleted
            }
            return hasCompleted(step: .merge, for: phase, hasClusters: hasClusters) ? 1 : 0

        case .save:
            if phase == .finalizing {
                return status.fractionCompleted
            }
            return hasCompleted(step: .save, for: phase, hasClusters: hasClusters) ? 1 : 0
        }
    }

    private func isDimmed(
        step: ProcessingPipelineStep,
        for phase: ProcessingPhase,
        hasClusters: Bool
    ) -> Bool {
        !hasCompleted(step: step, for: phase, hasClusters: hasClusters)
            && !isCurrent(step: step, for: phase)
    }

    private func isCurrent(step: ProcessingPipelineStep, for phase: ProcessingPhase) -> Bool {
        switch (step, phase) {
        case (.scan, .scanning):
            return true
        case (.graph, .loadingEmbeddings), (.graph, .buildingGraph):
            return true
        case (.merge, .clustering):
            return true
        case (.save, .finalizing):
            return true
        default:
            return false
        }
    }

    private func hasCompleted(
        step: ProcessingPipelineStep,
        for phase: ProcessingPhase,
        hasClusters: Bool = false
    ) -> Bool {
        switch step {
        case .scan:
            return phase != .scanning && (phase != .idle || hasClusters)
        case .graph:
            return phase == .clustering || phase == .finalizing || (phase == .idle && hasClusters)
        case .merge:
            return phase == .finalizing || (phase == .idle && hasClusters)
        case .save:
            return phase == .idle && hasClusters
        }
    }
}

// MARK: - UICollectionViewDelegate

extension FacesViewController: UICollectionViewDelegate {
    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        guard faceProcessor.status.phase == .idle, !faceProcessor.status.isProvisional else { return }
        guard let item = dataSource.itemIdentifier(for: indexPath),
              case .cluster(let clusterId) = item else { return }
        let vc = PersonPhotosViewController(personId: clusterId, photoManager: photoManager)
        navigationController?.pushViewController(vc, animated: true)
    }
}
