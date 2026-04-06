import UIKit
import Photos

final class PersonPhotosViewController: UIViewController {
    private enum Constants {
        static let columns: CGFloat = 3
        static let spacing: CGFloat = 2
        static let pageSize = 180
        static let loadMoreThreshold = 45
    }

    private let personId: Int
    private let photoManager: PhotoManager

    private var collectionView: UICollectionView!
    private var dataSource: UICollectionViewDiffableDataSource<Int, String>!
    private var assetIds: Set<String> = []
    private var matchingAssets: [PHAsset] = []
    private var personName = ""
    private var displayedAssetCount = 0

    private var cellSize: CGFloat {
        let availableWidth = max(view.bounds.width, 320)
        return floor(
            (availableWidth - Constants.spacing * (Constants.columns - 1))
            / Constants.columns
        )
    }

    private var gridImageTargetSize: CGSize {
        let scale = collectionView.window?.windowScene?.screen.scale ?? traitCollection.displayScale
        return CGSize(width: cellSize * scale, height: cellSize * scale)
    }

    init(personId: Int, photoManager: PhotoManager) {
        self.personId = personId
        self.photoManager = photoManager
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder: NSCoder) { fatalError() }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground
        setupCollectionView()
        setupDataSource()
        loadData()

        navigationItem.rightBarButtonItem = UIBarButtonItem(
            image: UIImage(systemName: "pencil"), style: .plain,
            target: self, action: #selector(renamePerson)
        )
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        updateLayoutMetrics()
    }

    @objc private func renamePerson() {
        let alert = UIAlertController(title: "Rename Person", message: nil, preferredStyle: .alert)
        alert.addTextField { tf in tf.text = self.personName; tf.placeholder = "Name" }
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
        alert.addAction(UIAlertAction(title: "Save", style: .default) { [weak self] _ in
            guard let self, let name = alert.textFields?.first?.text, !name.isEmpty else { return }
            self.personName = name
            self.title = name
            try? FaceDatabase().setName(forPerson: self.personId, name: name)
        })
        present(alert, animated: true)
    }

    // MARK: - Setup

    private func setupCollectionView() {
        let layout = UICollectionViewFlowLayout()
        layout.itemSize = CGSize(width: cellSize, height: cellSize)
        layout.minimumInteritemSpacing = Constants.spacing
        layout.minimumLineSpacing = Constants.spacing

        collectionView = UICollectionView(frame: view.bounds, collectionViewLayout: layout)
        collectionView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        collectionView.backgroundColor = .systemBackground
        collectionView.register(PhotoCell.self, forCellWithReuseIdentifier: PhotoCell.reuseId)
        collectionView.prefetchDataSource = self
        collectionView.delegate = self
        view.addSubview(collectionView)
    }

    private func setupDataSource() {
        dataSource = UICollectionViewDiffableDataSource<Int, String>(
            collectionView: collectionView
        ) { [weak self] collectionView, indexPath, assetId in
            let cell = collectionView.dequeueReusableCell(
                withReuseIdentifier: PhotoCell.reuseId, for: indexPath
            ) as! PhotoCell
            guard let self else { return cell }
            guard indexPath.item < self.matchingAssets.count else { return cell }
            let asset = self.matchingAssets[indexPath.item]

            cell.representedAssetId = assetId
            cell.imageView.image = nil

            let options = PHImageRequestOptions()
            options.deliveryMode = .fastFormat
            options.isNetworkAccessAllowed = true
            options.resizeMode = .fast

            self.photoManager.cachingImageManager.requestImage(
                for: asset, targetSize: self.gridImageTargetSize,
                contentMode: .aspectFill, options: options
            ) { image, _ in
                if cell.representedAssetId == assetId {
                    cell.imageView.image = image
                }
            }

            return cell
        }
    }

    // MARK: - Data

    private func loadData() {
        guard let db = try? FaceDatabase() else { return }
        let faces = (try? db.faces(forPerson: personId)) ?? []
        assetIds = Set(faces.map(\.assetId))
        matchingAssets = photoManager.assets.filter { assetIds.contains($0.localIdentifier) }
        displayedAssetCount = min(Constants.pageSize, matchingAssets.count)
        personName = (try? db.name(forPerson: personId)) ?? ""
        title = personName.isEmpty ? "Person \(personId)" : personName

        applySnapshot()
    }

    private func applySnapshot() {
        var snapshot = NSDiffableDataSourceSnapshot<Int, String>()
        snapshot.appendSections([0])
        snapshot.appendItems(
            Array(matchingAssets.prefix(displayedAssetCount)).map(\.localIdentifier)
        )
        dataSource.apply(snapshot, animatingDifferences: false)
    }

    private func updateLayoutMetrics() {
        guard let layout = collectionView.collectionViewLayout as? UICollectionViewFlowLayout else { return }
        let updatedSize = CGSize(width: cellSize, height: cellSize)
        guard layout.itemSize != updatedSize else { return }
        layout.itemSize = updatedSize
        layout.invalidateLayout()
    }

    private func maybeLoadMoreAssets(near index: Int) {
        guard displayedAssetCount < matchingAssets.count else { return }
        guard index >= displayedAssetCount - Constants.loadMoreThreshold else { return }
        displayedAssetCount = min(matchingAssets.count, displayedAssetCount + Constants.pageSize)
        applySnapshot()
    }
}

// MARK: - UICollectionViewDelegate

extension PersonPhotosViewController: UICollectionViewDelegate {
    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        guard indexPath.item < matchingAssets.count else { return }
        let browser = PhotoBrowserViewController(
            assets: matchingAssets,
            initialIndex: indexPath.item,
            imageManager: photoManager.cachingImageManager
        )
        navigationController?.pushViewController(browser, animated: true)
    }

    func collectionView(_ collectionView: UICollectionView, willDisplay cell: UICollectionViewCell, forItemAt indexPath: IndexPath) {
        maybeLoadMoreAssets(near: indexPath.item)
    }
}

// MARK: - Prefetching

extension PersonPhotosViewController: UICollectionViewDataSourcePrefetching {
    func collectionView(_ collectionView: UICollectionView, prefetchItemsAt indexPaths: [IndexPath]) {
        let toCache = indexPaths.compactMap { $0.item < matchingAssets.count ? matchingAssets[$0.item] : nil }
        photoManager.cachingImageManager.startCachingImages(
            for: toCache, targetSize: gridImageTargetSize, contentMode: .aspectFill, options: nil
        )
    }

    func collectionView(_ collectionView: UICollectionView, cancelPrefetchingForItemsAt indexPaths: [IndexPath]) {
        let toCancel = indexPaths.compactMap { $0.item < matchingAssets.count ? matchingAssets[$0.item] : nil }
        photoManager.cachingImageManager.stopCachingImages(
            for: toCancel, targetSize: gridImageTargetSize, contentMode: .aspectFill, options: nil
        )
    }
}
