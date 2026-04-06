import SwiftUI
import SceneKit

struct FaceCloudSceneContainer: UIViewRepresentable {
    let faces: [EmbeddingExplorerFace]
    let selectedPersonId: Int?
    let selectedFaceId: Int64?
    let onSelectFace: (Int64) -> Void

    func makeUIView(context: Context) -> FaceCloudSceneView {
        let view = FaceCloudSceneView(frame: .zero)
        view.onSelectFace = onSelectFace
        return view
    }

    func updateUIView(_ uiView: FaceCloudSceneView, context: Context) {
        uiView.onSelectFace = onSelectFace
        uiView.apply(
            faces: faces,
            selectedPersonId: selectedPersonId,
            selectedFaceId: selectedFaceId
        )
    }
}

final class FaceCloudSceneView: SCNView {
    var onSelectFace: ((Int64) -> Void)?

    private let cloudNode = SCNNode()
    private var renderedSignature: [Int64] = []
    private var currentSelectedPersonId: Int?
    private var currentSelectedFaceId: Int64?

    override init(frame: CGRect, options: [String: Any]? = nil) {
        super.init(frame: frame, options: options)
        backgroundColor = .black
        allowsCameraControl = true
        autoenablesDefaultLighting = false
        defaultCameraController.inertiaEnabled = true
        setupBaseScene()

        let tapRecognizer = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        addGestureRecognizer(tapRecognizer)
    }

    required init?(coder: NSCoder) { fatalError() }

    func apply(
        faces: [EmbeddingExplorerFace],
        selectedPersonId: Int?,
        selectedFaceId: Int64?
    ) {
        let signature = faces.map(\.id)
        currentSelectedPersonId = selectedPersonId
        currentSelectedFaceId = selectedFaceId

        if signature != renderedSignature {
            renderedSignature = signature
            rebuildScene(with: faces)
        } else {
            updateHighlighting()
        }
    }

    private func setupBaseScene() {
        let scene = SCNScene()
        self.scene = scene

        let ambientNode = SCNNode()
        ambientNode.light = SCNLight()
        ambientNode.light?.type = .ambient
        ambientNode.light?.intensity = 400
        scene.rootNode.addChildNode(ambientNode)

        let keyLight = SCNNode()
        keyLight.light = SCNLight()
        keyLight.light?.type = .omni
        keyLight.light?.intensity = 1_100
        keyLight.position = SCNVector3(4, 4, 6)
        scene.rootNode.addChildNode(keyLight)

        let fillLight = SCNNode()
        fillLight.light = SCNLight()
        fillLight.light?.type = .omni
        fillLight.light?.intensity = 500
        fillLight.position = SCNVector3(-4, -3, 5)
        scene.rootNode.addChildNode(fillLight)

        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.zNear = 0.01
        cameraNode.position = SCNVector3(0, 0, 3.2)
        scene.rootNode.addChildNode(cameraNode)

        cloudNode.name = "cloud"
        scene.rootNode.addChildNode(cloudNode)
    }

    private func rebuildScene(with faces: [EmbeddingExplorerFace]) {
        currentFaces = faces
        cloudNode.childNodes.forEach { $0.removeFromParentNode() }
        guard !faces.isEmpty else { return }

        let xs = faces.map(\.x)
        let ys = faces.map(\.y)
        let zs = faces.map(\.z)

        let xCenter = ((xs.min() ?? 0) + (xs.max() ?? 0)) / 2
        let yCenter = ((ys.min() ?? 0) + (ys.max() ?? 0)) / 2
        let zCenter = ((zs.min() ?? 0) + (zs.max() ?? 0)) / 2

        let xSpan = max((xs.max() ?? 0) - (xs.min() ?? 0), 0.001)
        let ySpan = max((ys.max() ?? 0) - (ys.min() ?? 0), 0.001)
        let zSpan = max((zs.max() ?? 0) - (zs.min() ?? 0), 0.001)
        let maxSpan = max(xSpan, ySpan, zSpan)

        let radius: CGFloat
        switch faces.count {
        case ..<400:
            radius = 0.018
        case ..<1_200:
            radius = 0.013
        default:
            radius = 0.009
        }

        for face in faces {
            let x = Float((face.x - xCenter) / maxSpan * 2.0)
            let y = Float((face.y - yCenter) / maxSpan * 2.0)
            let z = Float((face.z - zCenter) / maxSpan * 2.0)

            let sphere = SCNSphere(radius: radius)
            sphere.segmentCount = 10
            sphere.firstMaterial?.lightingModel = .physicallyBased
            sphere.firstMaterial?.metalness.contents = 0.05
            sphere.firstMaterial?.roughness.contents = 0.4
            sphere.firstMaterial?.diffuse.contents = clusterUIColor(for: face.personId)

            let node = SCNNode(geometry: sphere)
            node.name = String(face.id)
            node.position = SCNVector3(x, y, z)
            cloudNode.addChildNode(node)
        }

        updateHighlighting()
    }

    private func updateHighlighting() {
        for node in cloudNode.childNodes {
            guard let faceIdText = node.name,
                  let faceId = Int64(faceIdText),
                  let face = face(for: faceId) else {
                continue
            }

            if let currentSelectedFaceId {
                if face.id == currentSelectedFaceId {
                    node.opacity = 1
                    node.scale = SCNVector3(1.45, 1.45, 1.45)
                } else if face.personId == currentSelectedPersonId {
                    node.opacity = 0.18
                    node.scale = SCNVector3(0.98, 0.98, 0.98)
                } else {
                    node.opacity = 0.05
                    node.scale = SCNVector3(0.86, 0.86, 0.86)
                }
            } else {
                let isHighlighted = currentSelectedPersonId == nil || face.personId == currentSelectedPersonId
                node.opacity = isHighlighted ? 1 : 0.10
                node.scale = isHighlighted ? SCNVector3(1, 1, 1) : SCNVector3(0.92, 0.92, 0.92)
            }
        }
    }

    @objc
    private func handleTap(_ recognizer: UITapGestureRecognizer) {
        let location = recognizer.location(in: self)
        let hits = hitTest(location, options: nil)
        guard let faceIdText = hits.first?.node.name,
              let faceId = Int64(faceIdText) else {
            return
        }
        onSelectFace?(faceId)
    }

    private func face(for faceId: Int64) -> EmbeddingExplorerFace? {
        currentFaces.first(where: { $0.id == faceId })
    }

    private var currentFaces: [EmbeddingExplorerFace] = []

    private func clusterUIColor(for personId: Int) -> UIColor {
        let hue = CGFloat((personId * 37) % 360) / 360
        return UIColor(hue: hue, saturation: 0.72, brightness: 0.98, alpha: 1)
    }
}
