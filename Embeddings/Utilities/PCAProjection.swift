import Accelerate

enum PCAProjection {
    struct Point2D: Identifiable, Sendable, Equatable {
        let id: Int
        let x: Double
        let y: Double
        let label: Int
    }

    struct Point3D: Identifiable, Sendable, Equatable {
        let id: Int
        let x: Double
        let y: Double
        let z: Double
        let label: Int
    }

    struct Projection2D: Sendable, Equatable {
        let points: [Point2D]
        let explainedVariance: [Double]
    }

    struct Projection3D: Sendable, Equatable {
        let points: [Point3D]
        let explainedVariance: [Double]
    }

    static func project2D(embeddings: [(vector: [Float], label: Int)]) -> Projection2D {
        guard let decomposition = eigenDecompose(embeddings: embeddings) else {
            return Projection2D(points: [], explainedVariance: [])
        }

        let components = topComponents(
            from: decomposition.eigenvectors,
            dimension: decomposition.dimension,
            count: min(2, decomposition.dimension)
        )
        let projections = project(
            centered: decomposition.centered,
            dimension: decomposition.dimension,
            componentVectors: components
        )
        let explainedVariance = explainedVariance(
            from: decomposition.eigenvalues,
            count: 2
        )

        let padded = padded(projections, count: 2, itemCount: decomposition.sampleCount)
        let points = (0..<decomposition.sampleCount).map { index in
            Point2D(
                id: index,
                x: Double(padded[0][index]),
                y: Double(padded[1][index]),
                label: embeddings[index].label
            )
        }

        return Projection2D(points: points, explainedVariance: explainedVariance)
    }

    static func project3D(embeddings: [(vector: [Float], label: Int)]) -> Projection3D {
        guard let decomposition = eigenDecompose(embeddings: embeddings) else {
            return Projection3D(points: [], explainedVariance: [])
        }

        let components = topComponents(
            from: decomposition.eigenvectors,
            dimension: decomposition.dimension,
            count: min(3, decomposition.dimension)
        )
        let projections = project(
            centered: decomposition.centered,
            dimension: decomposition.dimension,
            componentVectors: components
        )
        let explainedVariance = explainedVariance(
            from: decomposition.eigenvalues,
            count: 3
        )

        let padded = padded(projections, count: 3, itemCount: decomposition.sampleCount)
        let points = (0..<decomposition.sampleCount).map { index in
            Point3D(
                id: index,
                x: Double(padded[0][index]),
                y: Double(padded[1][index]),
                z: Double(padded[2][index]),
                label: embeddings[index].label
            )
        }

        return Projection3D(points: points, explainedVariance: explainedVariance)
    }

    private struct EigenDecomposition {
        let centered: [Float]
        let eigenvectors: [Float]
        let eigenvalues: [Float]
        let dimension: Int
        let sampleCount: Int
    }

    private static func eigenDecompose(
        embeddings: [(vector: [Float], label: Int)]
    ) -> EigenDecomposition? {
        guard let dimension = embeddings.first?.vector.count,
              dimension > 0,
              embeddings.count > 0 else {
            return nil
        }

        let sampleCount = embeddings.count
        if sampleCount == 1 {
            return EigenDecomposition(
                centered: embeddings[0].vector,
                eigenvectors: identityMatrix(size: dimension),
                eigenvalues: [Float](repeating: 0, count: dimension),
                dimension: dimension,
                sampleCount: sampleCount
            )
        }

        var mean = [Float](repeating: 0, count: dimension)
        for embedding in embeddings {
            for component in 0..<dimension {
                mean[component] += embedding.vector[component]
            }
        }

        var inverseCount = 1.0 / Float(sampleCount)
        vDSP_vsmul(mean, 1, &inverseCount, &mean, 1, vDSP_Length(dimension))

        var centered = [Float](repeating: 0, count: sampleCount * dimension)
        for row in 0..<sampleCount {
            for component in 0..<dimension {
                centered[row * dimension + component] =
                    embeddings[row].vector[component] - mean[component]
            }
        }

        var covariance = [Float](repeating: 0, count: dimension * dimension)
        centered.withUnsafeBufferPointer { centeredBuffer in
            covariance.withUnsafeMutableBufferPointer { covarianceBuffer in
                guard let centeredBase = centeredBuffer.baseAddress,
                      let covarianceBase = covarianceBuffer.baseAddress else { return }

                cblas_sgemm(
                    CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    Int32(dimension),
                    Int32(dimension),
                    Int32(sampleCount),
                    1.0 / Float(sampleCount),
                    centeredBase,
                    Int32(dimension),
                    centeredBase,
                    Int32(dimension),
                    0.0,
                    covarianceBase,
                    Int32(dimension)
                )
            }
        }

        var jobz = Int8(UnicodeScalar("V").value)
        var uplo = Int8(UnicodeScalar("U").value)
        var dimN = Int32(dimension)
        var eigenvalues = [Float](repeating: 0, count: dimension)
        var workSize: Float = 0
        var lwork = Int32(-1)
        var info = Int32(0)

        ssyev_(
            &jobz,
            &uplo,
            &dimN,
            &covariance,
            &dimN,
            &eigenvalues,
            &workSize,
            &lwork,
            &info
        )
        guard info == 0 else { return nil }

        lwork = max(1, Int32(workSize.rounded(.up)))
        var work = [Float](repeating: 0, count: Int(lwork))
        ssyev_(
            &jobz,
            &uplo,
            &dimN,
            &covariance,
            &dimN,
            &eigenvalues,
            &work,
            &lwork,
            &info
        )

        guard info == 0 else { return nil }

        return EigenDecomposition(
            centered: centered,
            eigenvectors: covariance,
            eigenvalues: eigenvalues,
            dimension: dimension,
            sampleCount: sampleCount
        )
    }

    private static func identityMatrix(size: Int) -> [Float] {
        var matrix = [Float](repeating: 0, count: size * size)
        for index in 0..<size {
            matrix[index * size + index] = 1
        }
        return matrix
    }

    private static func topComponents(
        from eigenvectors: [Float],
        dimension: Int,
        count: Int
    ) -> [[Float]] {
        guard dimension > 0, count > 0 else { return [] }
        return (0..<count).map { componentIndex in
            let start = (dimension - 1 - componentIndex) * dimension
            return Array(eigenvectors[start..<(start + dimension)])
        }
    }

    private static func project(
        centered: [Float],
        dimension: Int,
        componentVectors: [[Float]]
    ) -> [[Float]] {
        let sampleCount = centered.count / max(dimension, 1)
        guard sampleCount > 0 else { return [] }

        return componentVectors.map { component in
            var values = [Float](repeating: 0, count: sampleCount)
            for sampleIndex in 0..<sampleCount {
                var projection: Float = 0
                centered.withUnsafeBufferPointer { centeredBuffer in
                    component.withUnsafeBufferPointer { componentBuffer in
                        guard let centeredBase = centeredBuffer.baseAddress,
                              let componentBase = componentBuffer.baseAddress else { return }
                        let rowBase = centeredBase.advanced(by: sampleIndex * dimension)
                        vDSP_dotpr(
                            rowBase,
                            1,
                            componentBase,
                            1,
                            &projection,
                            vDSP_Length(dimension)
                        )
                    }
                }
                values[sampleIndex] = projection
            }
            return values
        }
    }

    private static func explainedVariance(from eigenvalues: [Float], count: Int) -> [Double] {
        let total = max(eigenvalues.reduce(0, +), .ulpOfOne)
        return (0..<count).map { componentIndex in
            let eigenvalueIndex = max(0, eigenvalues.count - 1 - componentIndex)
            return Double(eigenvalues[eigenvalueIndex] / total)
        }
    }

    private static func padded(
        _ projections: [[Float]],
        count: Int,
        itemCount: Int
    ) -> [[Float]] {
        var padded = projections
        while padded.count < count {
            padded.append([Float](repeating: 0, count: itemCount))
        }
        return padded
    }
}
