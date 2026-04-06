import UIKit
import SwiftUI

final class EmbeddingsViewController: UIViewController {
    private let faceProcessor: FaceProcessor
    private let photoManager: PhotoManager
    private let explorerStore = EmbeddingExplorerStore()

    private var hostingController: UIHostingController<EmbeddingExplorerView>!
    private var observationTask: Task<Void, Never>?
    private var lastClusterSignature: [String] = []

    init(faceProcessor: FaceProcessor, photoManager: PhotoManager) {
        self.faceProcessor = faceProcessor
        self.photoManager = photoManager
        super.init(nibName: nil, bundle: nil)
        title = "Graphs"
    }

    required init?(coder: NSCoder) { fatalError() }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground
        setupUI()
        faceProcessor.loadExistingClusters()
        startObserving()
    }

    deinit {
        observationTask?.cancel()
    }

    private func setupUI() {
        let rootView = EmbeddingExplorerView(
            store: explorerStore,
            onOpenPerson: { [weak self] personId in
                self?.openPerson(personId)
            }
        )

        hostingController = UIHostingController(rootView: rootView)
        addChild(hostingController)
        hostingController.view.translatesAutoresizingMaskIntoConstraints = false
        hostingController.view.backgroundColor = .clear
        view.addSubview(hostingController.view)

        NSLayoutConstraint.activate([
            hostingController.view.topAnchor.constraint(equalTo: view.topAnchor),
            hostingController.view.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            hostingController.view.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            hostingController.view.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])

        hostingController.didMove(toParent: self)
    }

    private func startObserving() {
        observationTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let (clusters, isProcessing) = withObservationTracking {
                    (self.faceProcessor.clusters, self.faceProcessor.isProcessing)
                } onChange: { }

                await MainActor.run {
                    self.explorerStore.isLoading = isProcessing
                    if !isProcessing {
                        self.rebuildIfNeeded(clusters: clusters)
                    }
                }

                try? await Task.sleep(for: .milliseconds(200))
            }
        }
    }

    private func rebuildIfNeeded(clusters: [PersonCluster]) {
        let signature = clusters
            .sorted { lhs, rhs in lhs.id < rhs.id }
            .map { "\($0.id):\($0.count)" }
        guard signature != lastClusterSignature else { return }
        lastClusterSignature = signature

        guard !clusters.isEmpty else {
            explorerStore.apply(snapshot: .empty)
            return
        }

        explorerStore.isLoading = true

        Task.detached {
            guard let database = try? FaceDatabase() else {
                await MainActor.run {
                    self.explorerStore.isLoading = false
                    self.explorerStore.apply(snapshot: .empty)
                }
                return
            }

            let embeddingVersion = (try? database.availableEmbeddingVersions().first) ?? nil
            let allFaces = (try? database.allFaces(embeddingVersion: embeddingVersion)) ?? []
            let assignedFaces = allFaces.compactMap { face -> (FaceRecord, Int64, Int)? in
                guard let faceId = face.id,
                      let personId = face.personId else { return nil }
                return (face, faceId, personId)
            }

            guard !assignedFaces.isEmpty else {
                await MainActor.run {
                    self.explorerStore.isLoading = false
                    self.explorerStore.apply(snapshot: .empty)
                }
                return
            }

            let personNames = (try? database.allNames()) ?? [:]

            let overviewProjection = PCAProjection.project3D(
                embeddings: assignedFaces.map { ($0.0.embeddingVector, $0.2) }
            )

            let overviewFaces = zip(assignedFaces, overviewProjection.points).map { entry, point in
                EmbeddingExplorerFace(
                    id: entry.1,
                    assetId: entry.0.assetId,
                    boundingBox: entry.0.boundingRect,
                    personId: entry.2,
                    x: point.x,
                    y: point.y,
                    z: point.z
                )
            }

            let groupedFaces = Dictionary(grouping: overviewFaces, by: \.personId)
            let people = groupedFaces.compactMap { personId, faces -> EmbeddingExplorerPerson? in
                guard let representativeFace = faces.max(by: { $0.boundingArea < $1.boundingArea }) else {
                    return nil
                }

                return EmbeddingExplorerPerson(
                    id: personId,
                    displayName: personNames[personId] ?? "Person \(personId)",
                    count: faces.count,
                    representativeFaceId: representativeFace.id,
                    representativeAssetId: representativeFace.assetId,
                    representativeBoundingBox: representativeFace.boundingBox
                )
            }
            .sorted { lhs, rhs in
                if lhs.count == rhs.count {
                    return lhs.id < rhs.id
                }
                return lhs.count > rhs.count
            }

            let snapshot = EmbeddingExplorerSnapshot(
                faces: overviewFaces,
                people: people
            )

            await MainActor.run {
                self.explorerStore.isLoading = false
                self.explorerStore.apply(snapshot: snapshot)
            }
        }
    }

    private func openPerson(_ personId: Int) {
        let viewController = PersonPhotosViewController(
            personId: personId,
            photoManager: photoManager
        )
        navigationController?.pushViewController(viewController, animated: true)
    }
}
