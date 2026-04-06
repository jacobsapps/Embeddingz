import UIKit
import Photos

final class PhotosViewController: UIViewController {
    private enum Constants {
        static let columns: CGFloat = 3
        static let spacing: CGFloat = 2
        static let pageSize = 180
        static let loadMoreThreshold = 45
    }

    private let photoManager: PhotoManager
    private let faceProcessor: FaceProcessor
    private let database: FaceDatabase

    private var collectionView: UICollectionView!
    private var dataSource: UICollectionViewDiffableDataSource<String, String>!
    private var observationTask: Task<Void, Never>?
    private var groupedByDate = false
    private var lastAssetIds: [String] = []
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

    init(photoManager: PhotoManager, faceProcessor: FaceProcessor, database: FaceDatabase) {
        self.photoManager = photoManager
        self.faceProcessor = faceProcessor
        self.database = database
        super.init(nibName: nil, bundle: nil)
        title = "Photos"
    }

    required init?(coder: NSCoder) { fatalError() }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground
        setupNavBar()
        setupCollectionView()
        setupDataSource()
        startObserving()
    }

    deinit {
        observationTask?.cancel()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        updateLayoutMetrics()
    }

    // MARK: - Nav Bar

    private func setupNavBar() {
        let currentLimit = photoManager.fetchLimit
        let limitButton = UIBarButtonItem(
            image: UIImage(systemName: "number"),
            menu: UIMenu(title: "Fetch Limit", children: [
                UIAction(title: "100", state: currentLimit == 100 ? .on : .off) { [weak self] _ in self?.setFetchLimit(100) },
                UIAction(title: "500", state: currentLimit == 500 ? .on : .off) { [weak self] _ in self?.setFetchLimit(500) },
                UIAction(title: "1,000", state: currentLimit == 1_000 ? .on : .off) { [weak self] _ in self?.setFetchLimit(1_000) },
                UIAction(title: "5,000", state: currentLimit == 5_000 ? .on : .off) { [weak self] _ in self?.setFetchLimit(5_000) },
                UIAction(title: "All Photos", state: currentLimit == 0 ? .on : .off) { [weak self] _ in self?.setFetchLimit(0) },
            ])
        )

        let calendarButton = UIBarButtonItem(
            image: UIImage(systemName: "calendar"),
            style: .plain, target: self, action: #selector(toggleGroupByDate)
        )

        let trashButton = UIBarButtonItem(
            image: UIImage(systemName: "trash"),
            style: .plain, target: self, action: #selector(confirmDeleteDatabase)
        )
        trashButton.tintColor = .systemRed

        navigationItem.rightBarButtonItems = [trashButton, calendarButton, limitButton]
    }

    private func setFetchLimit(_ limit: Int) {
        photoManager.fetchLimit = limit
        photoManager.refetchAssets()
        setupNavBar() // Rebuild menu to update checkmarks
    }

    @objc private func toggleGroupByDate() {
        groupedByDate.toggle()
        applySnapshot(assets: photoManager.assets)
    }

    @objc private func confirmDeleteDatabase() {
        let alert = UIAlertController(
            title: "Delete Database",
            message: "This will remove all detected faces and clusters. You'll need to re-scan.",
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
        alert.addAction(UIAlertAction(title: "Delete", style: .destructive) { [weak self] _ in
            self?.deleteDatabase()
        })
        present(alert, animated: true)
    }

    private func deleteDatabase() {
        let _ = try? database.deleteAll()
        faceProcessor.resetState()
    }

    // MARK: - Collection View

    private func setupCollectionView() {
        let layout = UICollectionViewFlowLayout()
        layout.itemSize = CGSize(width: cellSize, height: cellSize)
        layout.minimumInteritemSpacing = Constants.spacing
        layout.minimumLineSpacing = Constants.spacing
        layout.headerReferenceSize = CGSize(width: 0, height: 40)

        collectionView = UICollectionView(frame: view.bounds, collectionViewLayout: layout)
        collectionView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        collectionView.backgroundColor = .systemBackground
        collectionView.register(PhotoCell.self, forCellWithReuseIdentifier: PhotoCell.reuseId)
        collectionView.register(DateHeaderView.self, forSupplementaryViewOfKind: UICollectionView.elementKindSectionHeader, withReuseIdentifier: DateHeaderView.reuseId)
        collectionView.prefetchDataSource = self
        collectionView.delegate = self
        view.addSubview(collectionView)
    }

    private func setupDataSource() {
        dataSource = UICollectionViewDiffableDataSource<String, String>(
            collectionView: collectionView
        ) { [weak self] collectionView, indexPath, assetId in
            let cell = collectionView.dequeueReusableCell(
                withReuseIdentifier: PhotoCell.reuseId, for: indexPath
            ) as! PhotoCell
            guard let self else { return cell }
            guard let asset = self.photoManager.asset(for: assetId) else { return cell }

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

        dataSource.supplementaryViewProvider = { collectionView, kind, indexPath in
            let header = collectionView.dequeueReusableSupplementaryView(
                ofKind: kind, withReuseIdentifier: DateHeaderView.reuseId, for: indexPath
            ) as! DateHeaderView
            let snapshot = self.dataSource.snapshot()
            let section = snapshot.sectionIdentifiers[indexPath.section]
            header.label.text = section == "all" ? nil : section
            return header
        }
    }

    // MARK: - Observation

    private func startObserving() {
        observationTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let assets = withObservationTracking {
                    self.photoManager.assets
                } onChange: { }
                let assetIds = assets.map(\.localIdentifier)
                guard assetIds != self.lastAssetIds else {
                    try? await Task.sleep(for: .milliseconds(200))
                    continue
                }
                self.lastAssetIds = assetIds
                self.applySnapshot(assets: assets)
                try? await Task.sleep(for: .milliseconds(200))
            }
        }
    }

    @MainActor
    private func applySnapshot(assets: [PHAsset]) {
        if displayedAssetCount == 0 || displayedAssetCount > assets.count {
            displayedAssetCount = min(Constants.pageSize, assets.count)
        }

        let visibleAssets = Array(assets.prefix(displayedAssetCount))
        var snapshot = NSDiffableDataSourceSnapshot<String, String>()

        if groupedByDate {
            let formatter = DateFormatter()
            formatter.dateFormat = "MMMM yyyy"
            var groups: [(key: String, ids: [String])] = []
            var current: (key: String, ids: [String])?
            for asset in visibleAssets {
                let key = formatter.string(from: asset.creationDate ?? Date.distantPast)
                if current?.key == key {
                    current?.ids.append(asset.localIdentifier)
                } else {
                    if let c = current { groups.append(c) }
                    current = (key: key, ids: [asset.localIdentifier])
                }
            }
            if let c = current { groups.append(c) }

            for group in groups {
                snapshot.appendSections([group.key])
                snapshot.appendItems(group.ids, toSection: group.key)
            }
        } else {
            snapshot.appendSections(["all"])
            snapshot.appendItems(visibleAssets.map(\.localIdentifier), toSection: "all")
        }

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
        let total = photoManager.assets.count
        guard displayedAssetCount < total else { return }
        guard index >= displayedAssetCount - Constants.loadMoreThreshold else { return }
        displayedAssetCount = min(total, displayedAssetCount + Constants.pageSize)
        applySnapshot(assets: photoManager.assets)
    }
}

// MARK: - UICollectionViewDelegate

extension PhotosViewController: UICollectionViewDelegate {
    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        guard let assetId = dataSource.itemIdentifier(for: indexPath),
              let index = photoManager.index(of: assetId) else { return }
        let browser = PhotoBrowserViewController(
            assets: photoManager.assets,
            initialIndex: index,
            imageManager: photoManager.cachingImageManager
        )
        navigationController?.pushViewController(browser, animated: true)
    }

    func collectionView(_ collectionView: UICollectionView, willDisplay cell: UICollectionViewCell, forItemAt indexPath: IndexPath) {
        guard let assetId = dataSource.itemIdentifier(for: indexPath),
              let index = photoManager.index(of: assetId) else { return }
        maybeLoadMoreAssets(near: index)
    }
}

// MARK: - Prefetching

extension PhotosViewController: UICollectionViewDataSourcePrefetching {
    func collectionView(_ collectionView: UICollectionView, prefetchItemsAt indexPaths: [IndexPath]) {
        let toCache = indexPaths.compactMap { ip -> PHAsset? in
            guard let id = dataSource.itemIdentifier(for: ip) else { return nil }
            return photoManager.asset(for: id)
        }
        photoManager.cachingImageManager.startCachingImages(
            for: toCache, targetSize: gridImageTargetSize, contentMode: .aspectFill, options: nil
        )
    }

    func collectionView(_ collectionView: UICollectionView, cancelPrefetchingForItemsAt indexPaths: [IndexPath]) {
        let toCancel = indexPaths.compactMap { ip -> PHAsset? in
            guard let id = dataSource.itemIdentifier(for: ip) else { return nil }
            return photoManager.asset(for: id)
        }
        photoManager.cachingImageManager.stopCachingImages(
            for: toCancel, targetSize: gridImageTargetSize, contentMode: .aspectFill, options: nil
        )
    }
}

// MARK: - Photo Cell

final class PhotoCell: UICollectionViewCell {
    static let reuseId = "PhotoCell"
    var representedAssetId: String?
    let imageView = UIImageView()

    override init(frame: CGRect) {
        super.init(frame: frame)
        imageView.contentMode = .scaleAspectFill
        imageView.clipsToBounds = true
        imageView.frame = contentView.bounds
        imageView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        imageView.backgroundColor = .secondarySystemFill
        contentView.addSubview(imageView)
    }

    required init?(coder: NSCoder) { fatalError() }

    override func prepareForReuse() {
        super.prepareForReuse()
        imageView.image = nil
        representedAssetId = nil
    }
}

// MARK: - Date Header View

final class DateHeaderView: UICollectionReusableView {
    static let reuseId = "DateHeaderView"
    let label = UILabel()

    override init(frame: CGRect) {
        super.init(frame: frame)
        label.font = .preferredFont(forTextStyle: .headline)
        label.translatesAutoresizingMaskIntoConstraints = false
        addSubview(label)
        NSLayoutConstraint.activate([
            label.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 12),
            label.centerYAnchor.constraint(equalTo: centerYAnchor),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }
}
