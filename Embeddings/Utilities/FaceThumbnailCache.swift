import UIKit
import Photos

final class FaceThumbnailCache: @unchecked Sendable {
    static let shared = FaceThumbnailCache()

    private let cache = NSCache<NSString, UIImage>()
    private let imageManager = PHCachingImageManager()

    private init() {
        cache.countLimit = 500
        cache.totalCostLimit = 64 * 1024 * 1024
    }

    func image(forKey key: String) -> UIImage? {
        cache.object(forKey: key as NSString)
    }

    func setImage(_ image: UIImage, forKey key: String) {
        cache.setObject(image, forKey: key as NSString)
    }

    func removeImage(forKey key: String) {
        cache.removeObject(forKey: key as NSString)
    }

    /// Load a cropped face thumbnail, using cache if available.
    func loadFaceThumbnail(
        assetId: String,
        boundingBox: CGRect,
        cacheKey: String
    ) async -> UIImage? {
        if let cached = image(forKey: cacheKey) {
            return cached
        }

        let fetchResult = PHAsset.fetchAssets(withLocalIdentifiers: [assetId], options: nil)
        guard let asset = fetchResult.firstObject else { return nil }

        let targetSize = CGSize(width: 384, height: 384)
        guard let cgImage = await loadImage(for: asset, targetSize: targetSize) else { return nil }

        let w = CGFloat(cgImage.width)
        let h = CGFloat(cgImage.height)
        let padding: CGFloat = 0.3
        let faceWidth = boundingBox.width * w
        let faceHeight = boundingBox.height * h
        let padX = faceWidth * padding
        let padY = faceHeight * padding

        let x = max(0, boundingBox.origin.x * w - padX)
        let y = max(0, (1 - boundingBox.origin.y - boundingBox.height) * h - padY)
        let cropW = min(w - x, faceWidth + 2 * padX)
        let cropH = min(h - y, faceHeight + 2 * padY)

        let rect = CGRect(x: x, y: y, width: cropW, height: cropH)
        guard let cropped = cgImage.cropping(to: rect) else { return nil }

        let result = UIImage(cgImage: cropped)
        let estimatedCost = Int(result.size.width * result.size.height * result.scale * result.scale * 4)
        cache.setObject(result, forKey: cacheKey as NSString, cost: estimatedCost)
        return result
    }

    private func loadImage(for asset: PHAsset, targetSize: CGSize) async -> CGImage? {
        await withCheckedContinuation { continuation in
            let options = PHImageRequestOptions()
            options.deliveryMode = .highQualityFormat
            options.isNetworkAccessAllowed = true
            options.isSynchronous = false
            imageManager.requestImage(
                for: asset,
                targetSize: targetSize,
                contentMode: .aspectFit,
                options: options
            ) { image, _ in
                continuation.resume(returning: image?.cgImage)
            }
        }
    }
}
