import Photos
import UIKit
import Observation

@Observable
@MainActor
final class PhotoManager {
    var assets: [PHAsset] = []
    private(set) var assetsById: [String: PHAsset] = [:]
    private(set) var assetIndicesById: [String: Int] = [:]
    var authorizationStatus: PHAuthorizationStatus = .notDetermined

    let cachingImageManager = PHCachingImageManager()

    var fetchLimit: Int {
        get {
            let defaults = UserDefaults.standard
            return defaults.object(forKey: "photoFetchLimit") == nil
                ? 1_000
                : defaults.integer(forKey: "photoFetchLimit")
        }
        set {
            UserDefaults.standard.set(newValue, forKey: "photoFetchLimit")
        }
    }

    func requestAuthorization() {
        PHPhotoLibrary.requestAuthorization(for: .readWrite) { status in
            Task { @MainActor in
                self.authorizationStatus = status
                if status == .authorized || status == .limited {
                    self.fetchAssets()
                }
            }
        }
    }

    func checkAuthorization() {
        let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        authorizationStatus = status
        if status == .authorized || status == .limited {
            fetchAssets()
        } else if status == .notDetermined {
            requestAuthorization()
        }
    }

    func refetchAssets() {
        cachingImageManager.stopCachingImagesForAllAssets()
        fetchAssets()
    }

    func asset(for identifier: String) -> PHAsset? {
        assetsById[identifier]
    }

    func index(of identifier: String) -> Int? {
        assetIndicesById[identifier]
    }

    private func fetchAssets() {
        let options = PHFetchOptions()
        options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        if fetchLimit > 0 {
            options.fetchLimit = fetchLimit
        }
        options.predicate = NSPredicate(format: "mediaType == %d", PHAssetMediaType.image.rawValue)

        let collections = PHAssetCollection.fetchAssetCollections(
            with: .smartAlbum, subtype: .smartAlbumUserLibrary, options: nil
        )
        let result: PHFetchResult<PHAsset>
        if let cameraRoll = collections.firstObject {
            result = PHAsset.fetchAssets(in: cameraRoll, options: options)
        } else {
            result = PHAsset.fetchAssets(with: options)
        }

        var fetched: [PHAsset] = []
        result.enumerateObjects { asset, _, _ in
            fetched.append(asset)
        }
        assets = fetched
        assetsById = Dictionary(uniqueKeysWithValues: fetched.map { ($0.localIdentifier, $0) })
        assetIndicesById = Dictionary(
            uniqueKeysWithValues: fetched.enumerated().map { ($1.localIdentifier, $0) }
        )
        startCaching(fetched)
    }

    private func startCaching(_ assets: [PHAsset]) {
        guard !assets.isEmpty else { return }
        let size = CGSize(width: 200, height: 200)
        let options = PHImageRequestOptions()
        options.deliveryMode = .fastFormat
        options.resizeMode = .fast
        cachingImageManager.startCachingImages(
            for: Array(assets.prefix(240)),
            targetSize: size,
            contentMode: .aspectFill,
            options: options
        )
    }

    nonisolated func loadFullImage(for asset: PHAsset) async -> CGImage? {
        await withCheckedContinuation { continuation in
            let options = PHImageRequestOptions()
            options.deliveryMode = .highQualityFormat
            options.isNetworkAccessAllowed = true
            options.isSynchronous = true
            let manager = PHImageManager.default()
            manager.requestImage(
                for: asset,
                targetSize: CGSize(width: 1024, height: 1024),
                contentMode: .aspectFit,
                options: options
            ) { image, _ in
                continuation.resume(returning: image?.cgImage)
            }
        }
    }
}
