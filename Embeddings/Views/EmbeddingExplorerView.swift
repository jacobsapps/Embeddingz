import SwiftUI
import Charts
import Combine
import UIKit

struct EmbeddingExplorerFace: Identifiable, Hashable, Equatable, Sendable {
    let id: Int64
    let assetId: String
    let boundingBox: CGRect
    let personId: Int
    let x: Double
    let y: Double
    let z: Double

    var boundingArea: Double {
        Double(boundingBox.width * boundingBox.height)
    }
}

struct EmbeddingExplorerPerson: Identifiable, Hashable, Equatable, Sendable {
    let id: Int
    let displayName: String
    let count: Int
    let representativeFaceId: Int64
    let representativeAssetId: String
    let representativeBoundingBox: CGRect
}

struct EmbeddingExplorerSnapshot: Equatable, Sendable {
    let faces: [EmbeddingExplorerFace]
    let people: [EmbeddingExplorerPerson]

    static let empty = EmbeddingExplorerSnapshot(
        faces: [],
        people: []
    )
}

@MainActor
final class EmbeddingExplorerStore: ObservableObject {
    enum ProjectionMode: String, CaseIterable, Identifiable {
        case cloud3D = "3D"
        case map2D = "2D"

        var id: String { rawValue }
    }

    @Published var isLoading = false
    @Published var snapshot: EmbeddingExplorerSnapshot = .empty
    @Published var selectedPersonId: Int?
    @Published var selectedFaceId: Int64?
    @Published var focusedFaceId: Int64?
    @Published var projectionMode: ProjectionMode = .cloud3D

    var selectedPerson: EmbeddingExplorerPerson? {
        snapshot.people.first(where: { $0.id == selectedPersonId })
    }

    var selectedFace: EmbeddingExplorerFace? {
        snapshot.faces.first(where: { $0.id == selectedFaceId })
    }

    var selectedPersonFaces: [EmbeddingExplorerFace] {
        guard let selectedPersonId else { return [] }
        return snapshot.faces
            .filter { $0.personId == selectedPersonId }
            .sorted { lhs, rhs in
                if lhs.boundingArea == rhs.boundingArea {
                    return lhs.id < rhs.id
                }
                return lhs.boundingArea > rhs.boundingArea
            }
    }

    func apply(snapshot: EmbeddingExplorerSnapshot) {
        self.snapshot = snapshot

        if let selectedPersonId,
           snapshot.people.contains(where: { $0.id == selectedPersonId }) {
            if let selectedFaceId,
               snapshot.faces.contains(where: { $0.id == selectedFaceId }) {
                if let focusedFaceId,
                   !snapshot.faces.contains(where: { $0.id == focusedFaceId }) {
                    self.focusedFaceId = nil
                }
                return
            }

            self.selectedFaceId = bestFaceId(for: selectedPersonId, in: snapshot)
            self.focusedFaceId = nil
            return
        }

        selectedPersonId = snapshot.people.first?.id
        if let selectedPersonId {
            selectedFaceId = bestFaceId(for: selectedPersonId, in: snapshot)
        } else {
            selectedFaceId = nil
        }
        focusedFaceId = nil
    }

    func selectPerson(_ personId: Int) {
        guard snapshot.people.contains(where: { $0.id == personId }) else { return }
        selectedPersonId = personId
        selectedFaceId = bestFaceId(for: personId, in: snapshot)
        focusedFaceId = nil
    }

    func selectFace(_ faceId: Int64) {
        guard let face = snapshot.faces.first(where: { $0.id == faceId }) else { return }
        selectedPersonId = face.personId
        selectedFaceId = faceId
        focusedFaceId = faceId
    }

    private func bestFaceId(
        for personId: Int,
        in snapshot: EmbeddingExplorerSnapshot
    ) -> Int64? {
        snapshot.faces
            .filter { $0.personId == personId }
            .max { lhs, rhs in
                if lhs.boundingArea == rhs.boundingArea {
                    return lhs.id > rhs.id
                }
                return lhs.boundingArea < rhs.boundingArea
            }?
            .id
    }
}

struct EmbeddingExplorerView: View {
    @ObservedObject var store: EmbeddingExplorerStore
    let onOpenPerson: (Int) -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                if store.snapshot.faces.isEmpty, !store.isLoading {
                    emptyState
                } else {
                    overviewCard
                    peopleStrip

                    if let selectedPerson = store.selectedPerson {
                        selectedPersonCard(selectedPerson)
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 12)
            .padding(.bottom, 24)
        }
        .background(Color(.systemBackground))
        .overlay {
            if store.isLoading {
                ProgressView("Refreshing graphs…")
                    .padding(20)
                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            }
        }
    }

    private var overviewCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("All Faces")
                        .font(.title3.weight(.semibold))
                    Text("\(store.snapshot.faces.count) face crops • \(store.snapshot.people.count) people")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Picker("Projection", selection: $store.projectionMode) {
                    Text("3D").tag(EmbeddingExplorerStore.ProjectionMode.cloud3D)
                    Text("2D").tag(EmbeddingExplorerStore.ProjectionMode.map2D)
                }
                .pickerStyle(.segmented)
                .frame(width: 140)
            }

            Group {
                switch store.projectionMode {
                case .cloud3D:
                    FaceCloudSceneContainer(
                        faces: store.snapshot.faces,
                        selectedPersonId: store.selectedPersonId,
                        selectedFaceId: store.focusedFaceId,
                        onSelectFace: store.selectFace
                    )
                case .map2D:
                    OverviewFaceMap2D(
                        faces: store.snapshot.faces,
                        selectedPersonId: store.selectedPersonId,
                        selectedFaceId: store.focusedFaceId,
                        onSelectFace: store.selectFace
                    )
                }
            }
            .frame(height: 360)
            .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 24, style: .continuous))
    }

    private var peopleStrip: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("People")
                .font(.headline)

            ScrollView(.horizontal, showsIndicators: false) {
                LazyHStack(spacing: 12) {
                    ForEach(store.snapshot.people) { person in
                        PersonThumbnailStripCell(
                            person: person,
                            isSelected: person.id == store.selectedPersonId
                        ) {
                            store.selectPerson(person.id)
                        }
                    }
                }
                .padding(.vertical, 4)
            }
        }
    }

    @ViewBuilder
    private func selectedPersonCard(_ person: EmbeddingExplorerPerson) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(person.displayName)
                        .font(.title3.weight(.semibold))
                    Text("\(person.count) faces")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Button("Open Grid") {
                    onOpenPerson(person.id)
                }
                .buttonStyle(.borderedProminent)
            }

            SelectedPersonMap2D(
                allFaces: store.snapshot.faces,
                selectedFaces: store.selectedPersonFaces,
                selectedFaceId: store.focusedFaceId,
                personId: person.id,
                onSelectFace: store.selectFace
            )
            .frame(height: 250)

            ScrollView(.horizontal, showsIndicators: false) {
                LazyHStack(spacing: 10) {
                    ForEach(store.selectedPersonFaces) { face in
                        FaceThumbnailStripCell(
                            face: face,
                            isSelected: face.id == store.selectedFaceId
                        ) {
                            store.selectFace(face.id)
                        }
                    }
                }
                .padding(.vertical, 4)
            }
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 24, style: .continuous))
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("No face embeddings yet.")
                .font(.headline)
            Text("Scan in the Faces tab first. Once the app has detected faces and assigned people, the face graphs will show up here.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
    }
}

private struct OverviewFaceMap2D: View {
    let faces: [EmbeddingExplorerFace]
    let selectedPersonId: Int?
    let selectedFaceId: Int64?
    let onSelectFace: (Int64) -> Void

    var body: some View {
        Chart {
            ForEach(faces) { face in
                PointMark(
                    x: .value("x", face.x),
                    y: .value("y", face.y)
                )
                .symbolSize(face.id == selectedFaceId ? 120 : 42)
                .foregroundStyle(clusterColor(for: face.personId))
                .opacity(opacity(for: face))
            }
        }
        .chartLegend(.hidden)
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
        .chartOverlay { proxy in
            FaceMapOverlay(
                faces: faces,
                proxy: proxy,
                onSelectFace: onSelectFace
            )
        }
        .background(Color.black.opacity(0.06))
    }

    private func opacity(for face: EmbeddingExplorerFace) -> Double {
        if let selectedFaceId {
            if face.id == selectedFaceId {
                return 1
            }
            if face.personId == selectedPersonId {
                return 0.16
            }
            return 0.05
        }

        guard let selectedPersonId else { return 0.9 }
        return face.personId == selectedPersonId ? 0.95 : 0.10
    }
}

private struct SelectedPersonMap2D: View {
    let allFaces: [EmbeddingExplorerFace]
    let selectedFaces: [EmbeddingExplorerFace]
    let selectedFaceId: Int64?
    let personId: Int
    let onSelectFace: (Int64) -> Void

    var body: some View {
        Chart {
            ForEach(allFaces) { face in
                PointMark(
                    x: .value("x", face.x),
                    y: .value("y", face.y)
                )
                .symbolSize(28)
                .foregroundStyle(Color.white.opacity(selectedFaceId == nil ? 0.08 : 0.03))
            }

            ForEach(selectedFaces) { face in
                PointMark(
                    x: .value("x", face.x),
                    y: .value("y", face.y)
                )
                .symbolSize(face.id == selectedFaceId ? 130 : 52)
                .foregroundStyle(clusterColor(for: personId))
                .opacity(face.id == selectedFaceId ? 1 : (selectedFaceId == nil ? 0.9 : 0.12))
            }
        }
        .chartLegend(.hidden)
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
        .chartOverlay { proxy in
            FaceMapOverlay(
                faces: selectedFaces,
                proxy: proxy,
                onSelectFace: onSelectFace
            )
        }
        .background(Color.black.opacity(0.06))
    }
}

private struct FaceMapOverlay: View {
    let faces: [EmbeddingExplorerFace]
    let proxy: ChartProxy
    let onSelectFace: (Int64) -> Void

    var body: some View {
        GeometryReader { geometry in
            Rectangle()
                .fill(.clear)
                .contentShape(Rectangle())
                .gesture(
                    SpatialTapGesture()
                        .onEnded { value in
                            selectNearestFace(
                                at: value.location,
                                geometry: geometry
                            )
                        }
                )
        }
    }

    private func selectNearestFace(
        at location: CGPoint,
        geometry: GeometryProxy
    ) {
        guard let plotFrameAnchor = proxy.plotFrame else { return }
        let plotFrame = geometry[plotFrameAnchor]
        let localPoint = CGPoint(
            x: location.x - plotFrame.origin.x,
            y: location.y - plotFrame.origin.y
        )

        guard localPoint.x >= 0,
              localPoint.y >= 0,
              localPoint.x <= plotFrame.width,
              localPoint.y <= plotFrame.height else {
            return
        }

        var nearestFaceId: Int64?
        var nearestDistance = CGFloat.greatestFiniteMagnitude

        for face in faces {
            guard let xPosition = proxy.position(forX: face.x),
                  let yPosition = proxy.position(forY: face.y) else {
                continue
            }

            let dx = xPosition - localPoint.x
            let dy = yPosition - localPoint.y
            let distance = dx * dx + dy * dy

            if distance < nearestDistance {
                nearestDistance = distance
                nearestFaceId = face.id
            }
        }

        guard let nearestFaceId,
              nearestDistance <= 1_600 else { return }
        onSelectFace(nearestFaceId)
    }
}

private struct PersonThumbnailStripCell: View {
    let person: EmbeddingExplorerPerson
    let isSelected: Bool
    let action: () -> Void

    @State private var image: UIImage?

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 6) {
                ZStack(alignment: .bottomLeading) {
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(Color(.tertiarySystemBackground))

                    if let image {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFill()
                    } else {
                        ProgressView()
                            .controlSize(.small)
                    }

                    Text("\(person.count)")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.black.opacity(0.62), in: Capsule())
                        .padding(8)
                }
                .frame(width: 84, height: 84)
                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 3)
                }

                Text(person.displayName)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
            }
            .frame(width: 84, alignment: .leading)
        }
        .buttonStyle(.plain)
        .task(id: person.id) {
            guard image == nil else { return }
            image = await FaceThumbnailCache.shared.loadFaceThumbnail(
                assetId: person.representativeAssetId,
                boundingBox: person.representativeBoundingBox,
                cacheKey: "person-strip-\(person.id)"
            )
        }
    }
}

private struct FaceThumbnailStripCell: View {
    let face: EmbeddingExplorerFace
    let isSelected: Bool
    let action: () -> Void

    @State private var image: UIImage?

    var body: some View {
        Button(action: action) {
            ZStack {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Color(.tertiarySystemBackground))

                if let image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                } else {
                    ProgressView()
                        .controlSize(.small)
                }
            }
            .frame(width: 82, height: 82)
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 3)
            }
        }
        .buttonStyle(.plain)
        .task(id: face.id) {
            guard image == nil else { return }
            image = await FaceThumbnailCache.shared.loadFaceThumbnail(
                assetId: face.assetId,
                boundingBox: face.boundingBox,
                cacheKey: "embedding-explorer-\(face.id)"
            )
        }
    }
}

func clusterColor(for personId: Int) -> Color {
    let hue = Double((personId * 37) % 360) / 360.0
    return Color(hue: hue, saturation: 0.72, brightness: 0.95)
}
