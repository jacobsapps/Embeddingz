import UIKit
import Photos

// MARK: - Photo Browser

final class PhotoBrowserViewController: UIViewController {

    private let assets: [PHAsset]
    private let imageManager: PHCachingImageManager
    private var currentIndex: Int

    private var pageViewController: UIPageViewController!
    private var thumbnailCollectionView: UICollectionView!
    private var thumbnailHeightConstraint: NSLayoutConstraint!
    private var chromeVisible = true

    init(assets: [PHAsset], initialIndex: Int, imageManager: PHCachingImageManager) {
        self.assets = assets
        self.currentIndex = max(0, min(initialIndex, max(assets.count - 1, 0)))
        self.imageManager = imageManager
        super.init(nibName: nil, bundle: nil)
        modalPresentationStyle = .fullScreen
    }

    required init?(coder: NSCoder) { fatalError() }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        guard !assets.isEmpty else { return }
        setupPageViewController()
        setupThumbnailStrip()
        setupGestures()
        updateTitle()
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        navigationController?.navigationBar.prefersLargeTitles = false
        navigationController?.setNavigationBarHidden(false, animated: animated)
        let appearance = UINavigationBarAppearance()
        appearance.configureWithTransparentBackground()
        appearance.titleTextAttributes = [.foregroundColor: UIColor.white]
        navigationController?.navigationBar.standardAppearance = appearance
        navigationController?.navigationBar.scrollEdgeAppearance = appearance
        navigationController?.navigationBar.tintColor = .white
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        let appearance = UINavigationBarAppearance()
        appearance.configureWithDefaultBackground()
        navigationController?.navigationBar.standardAppearance = appearance
        navigationController?.navigationBar.scrollEdgeAppearance = appearance
        navigationController?.navigationBar.tintColor = nil
    }

    override var prefersStatusBarHidden: Bool { !chromeVisible }
    override var preferredStatusBarUpdateAnimation: UIStatusBarAnimation { .fade }

    // MARK: - Page View Controller

    private func setupPageViewController() {
        guard !assets.isEmpty else { return }
        pageViewController = UIPageViewController(
            transitionStyle: .scroll, navigationOrientation: .horizontal)
        pageViewController.dataSource = self
        pageViewController.delegate = self

        let initial = makePhotoPage(for: currentIndex)
        pageViewController.setViewControllers([initial], direction: .forward, animated: false)

        addChild(pageViewController)
        view.addSubview(pageViewController.view)
        pageViewController.view.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            pageViewController.view.topAnchor.constraint(equalTo: view.topAnchor),
            pageViewController.view.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            pageViewController.view.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            pageViewController.view.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])
        pageViewController.didMove(toParent: self)
    }

    private func makePhotoPage(for index: Int) -> ZoomablePhotoPage {
        ZoomablePhotoPage(asset: assets[index], index: index, imageManager: imageManager)
    }

    // MARK: - Thumbnail Strip

    private func setupThumbnailStrip() {
        let layout = UICollectionViewFlowLayout()
        layout.scrollDirection = .horizontal
        layout.itemSize = CGSize(width: 40, height: 40)
        layout.minimumInteritemSpacing = 2
        layout.minimumLineSpacing = 2
        layout.sectionInset = UIEdgeInsets(top: 0, left: 4, bottom: 0, right: 4)

        thumbnailCollectionView = UICollectionView(frame: .zero, collectionViewLayout: layout)
        thumbnailCollectionView.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        thumbnailCollectionView.showsHorizontalScrollIndicator = false
        thumbnailCollectionView.translatesAutoresizingMaskIntoConstraints = false
        thumbnailCollectionView.delegate = self
        thumbnailCollectionView.dataSource = self
        thumbnailCollectionView.register(ThumbnailCell.self, forCellWithReuseIdentifier: ThumbnailCell.reuseId)
        view.addSubview(thumbnailCollectionView)

        thumbnailHeightConstraint = thumbnailCollectionView.heightAnchor.constraint(equalToConstant: 50)
        NSLayoutConstraint.activate([
            thumbnailCollectionView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            thumbnailCollectionView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            thumbnailCollectionView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
            thumbnailHeightConstraint,
        ])

        // Add pan gesture for scrubbing through thumbnails
        let pan = UIPanGestureRecognizer(target: self, action: #selector(handleThumbnailPan(_:)))
        pan.delegate = self
        thumbnailCollectionView.addGestureRecognizer(pan)

        // Scroll to initial position
        DispatchQueue.main.async {
            self.updateThumbnailSelection(from: nil, to: self.currentIndex)
            self.scrollThumbnailToCurrentIndex(animated: false)
        }
    }

    @objc private func handleThumbnailPan(_ gesture: UIPanGestureRecognizer) {
        let location = gesture.location(in: thumbnailCollectionView)
        guard let indexPath = thumbnailCollectionView.indexPathForItem(at: location) else { return }
        let newIndex = indexPath.item
        guard newIndex != currentIndex, newIndex >= 0, newIndex < assets.count else { return }
        navigateToIndex(newIndex, animated: false)
    }

    private func scrollThumbnailToCurrentIndex(animated: Bool) {
        guard currentIndex < assets.count else { return }
        let indexPath = IndexPath(item: currentIndex, section: 0)
        thumbnailCollectionView.layoutIfNeeded()
        guard thumbnailCollectionView.numberOfItems(inSection: 0) > indexPath.item else { return }
        thumbnailCollectionView.scrollToItem(at: indexPath, at: .centeredHorizontally, animated: animated)
    }

    private func updateThumbnailSelection(from previousIndex: Int?, to newIndex: Int) {
        let indices = [previousIndex, newIndex]
            .compactMap { $0 }
            .filter { $0 >= 0 && $0 < assets.count }
        guard !indices.isEmpty else { return }

        let indexPaths = indices.map { IndexPath(item: $0, section: 0) }
        thumbnailCollectionView.reloadItems(at: indexPaths)
    }

    // MARK: - Gestures

    private func setupGestures() {
        let tap = UITapGestureRecognizer(target: self, action: #selector(handleSingleTap))
        tap.numberOfTapsRequired = 1
        // Don't interfere with double-tap zoom in ZoomablePhotoPage
        tap.require(toFail: doubleTapRecognizer())
        pageViewController.view.addGestureRecognizer(tap)
    }

    private func doubleTapRecognizer() -> UITapGestureRecognizer {
        // Find existing double-tap from the current page
        if let page = pageViewController.viewControllers?.first as? ZoomablePhotoPage,
           let doubleTap = page.view.gestureRecognizers?.first(where: {
               ($0 as? UITapGestureRecognizer)?.numberOfTapsRequired == 2
           }) as? UITapGestureRecognizer {
            return doubleTap
        }
        // Fallback: create a dummy recognizer
        let dummy = UITapGestureRecognizer()
        dummy.numberOfTapsRequired = 2
        return dummy
    }

    @objc private func handleSingleTap() {
        chromeVisible.toggle()
        UIView.animate(withDuration: 0.25) {
            self.navigationController?.setNavigationBarHidden(!self.chromeVisible, animated: false)
            self.thumbnailCollectionView.alpha = self.chromeVisible ? 1 : 0
            self.setNeedsStatusBarAppearanceUpdate()
        }
    }

    // MARK: - Navigation

    private func navigateToIndex(_ index: Int, animated: Bool) {
        guard index >= 0, index < assets.count, index != currentIndex else { return }
        let previousIndex = currentIndex
        let direction: UIPageViewController.NavigationDirection = index > currentIndex ? .forward : .reverse
        currentIndex = index
        let page = makePhotoPage(for: index)
        pageViewController.setViewControllers([page], direction: direction, animated: animated)
        updateTitle()
        updateThumbnailSelection(from: previousIndex, to: index)
        scrollThumbnailToCurrentIndex(animated: animated)
    }

    private func updateTitle() {
        title = "\(currentIndex + 1) of \(assets.count)"
    }
}

// MARK: - UIPageViewControllerDataSource

extension PhotoBrowserViewController: UIPageViewControllerDataSource {
    func pageViewController(
        _ pageViewController: UIPageViewController,
        viewControllerBefore viewController: UIViewController
    ) -> UIViewController? {
        guard let page = viewController as? ZoomablePhotoPage, page.index > 0 else { return nil }
        return makePhotoPage(for: page.index - 1)
    }

    func pageViewController(
        _ pageViewController: UIPageViewController,
        viewControllerAfter viewController: UIViewController
    ) -> UIViewController? {
        guard let page = viewController as? ZoomablePhotoPage, page.index < assets.count - 1 else { return nil }
        return makePhotoPage(for: page.index + 1)
    }
}

// MARK: - UIPageViewControllerDelegate

extension PhotoBrowserViewController: UIPageViewControllerDelegate {
    func pageViewController(
        _ pageViewController: UIPageViewController,
        didFinishAnimating finished: Bool,
        previousViewControllers: [UIViewController],
        transitionCompleted completed: Bool
    ) {
        guard completed,
              let page = pageViewController.viewControllers?.first as? ZoomablePhotoPage else { return }
        let previousIndex = currentIndex
        currentIndex = page.index
        updateTitle()
        updateThumbnailSelection(from: previousIndex, to: currentIndex)
        scrollThumbnailToCurrentIndex(animated: true)
    }
}

// MARK: - Thumbnail Collection View

extension PhotoBrowserViewController: UICollectionViewDataSource, UICollectionViewDelegate {
    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        assets.count
    }

    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        let cell = collectionView.dequeueReusableCell(
            withReuseIdentifier: ThumbnailCell.reuseId, for: indexPath
        ) as! ThumbnailCell

        let asset = assets[indexPath.item]
        cell.assetId = asset.localIdentifier
        cell.setSelected(indexPath.item == currentIndex)

        let options = PHImageRequestOptions()
        options.deliveryMode = .fastFormat
        options.resizeMode = .fast
        options.isNetworkAccessAllowed = true

        let size = CGSize(width: 80, height: 80)
        imageManager.requestImage(
            for: asset, targetSize: size,
            contentMode: .aspectFill, options: options
        ) { image, _ in
            if cell.assetId == asset.localIdentifier {
                cell.imageView.image = image
            }
        }

        return cell
    }

    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        navigateToIndex(indexPath.item, animated: false)
    }
}

// MARK: - UIGestureRecognizerDelegate

extension PhotoBrowserViewController: UIGestureRecognizerDelegate {
    func gestureRecognizer(
        _ gestureRecognizer: UIGestureRecognizer,
        shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer
    ) -> Bool {
        true
    }
}

// MARK: - Zoomable Photo Page

final class ZoomablePhotoPage: UIViewController, UIScrollViewDelegate, UIGestureRecognizerDelegate {

    let index: Int
    private let asset: PHAsset
    private let imageManager: PHCachingImageManager

    private var scrollView: UIScrollView!
    private var imageView: UIImageView!

    init(asset: PHAsset, index: Int, imageManager: PHCachingImageManager) {
        self.asset = asset
        self.index = index
        self.imageManager = imageManager
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder: NSCoder) { fatalError() }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        setupScrollView()
        setupDoubleTap()
        loadImage()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        centerImage()
    }

    // MARK: - Setup

    private func setupScrollView() {
        scrollView = UIScrollView(frame: view.bounds)
        scrollView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        scrollView.delegate = self
        scrollView.minimumZoomScale = 1.0
        scrollView.maximumZoomScale = 5.0
        scrollView.showsHorizontalScrollIndicator = false
        scrollView.showsVerticalScrollIndicator = false
        scrollView.bouncesZoom = true
        view.addSubview(scrollView)

        // Block rotation gestures on the scroll view's pinch recognizer
        for gesture in scrollView.gestureRecognizers ?? [] {
            if gesture is UIRotationGestureRecognizer {
                gesture.isEnabled = false
            }
            gesture.delegate = self
        }

        imageView = UIImageView()
        imageView.contentMode = .scaleAspectFit
        imageView.isUserInteractionEnabled = true
        scrollView.addSubview(imageView)
    }

    private func setupDoubleTap() {
        let doubleTap = UITapGestureRecognizer(target: self, action: #selector(handleDoubleTap(_:)))
        doubleTap.numberOfTapsRequired = 2
        view.addGestureRecognizer(doubleTap)
    }

    @objc private func handleDoubleTap(_ gesture: UITapGestureRecognizer) {
        if scrollView.zoomScale > scrollView.minimumZoomScale {
            scrollView.setZoomScale(scrollView.minimumZoomScale, animated: true)
        } else {
            let point = gesture.location(in: imageView)
            let zoomScale: CGFloat = 3.0
            let width = scrollView.bounds.width / zoomScale
            let height = scrollView.bounds.height / zoomScale
            let rect = CGRect(x: point.x - width / 2, y: point.y - height / 2, width: width, height: height)
            scrollView.zoom(to: rect, animated: true)
        }
    }

    // MARK: - Image Loading

    private func loadImage() {
        let targetSize = imageRequestTargetSize()
        let options = PHImageRequestOptions()
        options.deliveryMode = .highQualityFormat
        options.isNetworkAccessAllowed = true
        options.isSynchronous = false
        options.resizeMode = .exact

        imageManager.requestImage(
            for: asset, targetSize: targetSize,
            contentMode: .aspectFit, options: options
        ) { [weak self] image, _ in
            guard let self, let image else { return }
            self.imageView.image = image
            self.imageView.frame = CGRect(origin: .zero, size: image.size)
            self.scrollView.contentSize = image.size
            self.fitImageToView()
        }
    }

    private func imageRequestTargetSize() -> CGSize {
        let fallbackBounds = CGSize(width: 390, height: 844)
        let bounds = view.bounds.size == .zero ? fallbackBounds : view.bounds.size
        let screenScale = view.window?.windowScene?.screen.scale ?? traitCollection.displayScale
        let zoomBudget: CGFloat = 3

        let requestedWidth = bounds.width * screenScale * zoomBudget
        let requestedHeight = bounds.height * screenScale * zoomBudget

        return CGSize(
            width: min(CGFloat(asset.pixelWidth), requestedWidth),
            height: min(CGFloat(asset.pixelHeight), requestedHeight)
        )
    }

    private func fitImageToView() {
        guard let image = imageView.image else { return }
        let viewSize = scrollView.bounds.size
        guard viewSize.width > 0, viewSize.height > 0 else { return }

        let widthScale = viewSize.width / image.size.width
        let heightScale = viewSize.height / image.size.height
        let minScale = min(widthScale, heightScale)

        scrollView.minimumZoomScale = minScale
        scrollView.maximumZoomScale = max(minScale * 5, 1.0)
        scrollView.zoomScale = minScale
        centerImage()
    }

    private func centerImage() {
        let boundsSize = scrollView.bounds.size
        var frameToCenter = imageView.frame

        if frameToCenter.size.width < boundsSize.width {
            frameToCenter.origin.x = (boundsSize.width - frameToCenter.size.width) / 2
        } else {
            frameToCenter.origin.x = 0
        }

        if frameToCenter.size.height < boundsSize.height {
            frameToCenter.origin.y = (boundsSize.height - frameToCenter.size.height) / 2
        } else {
            frameToCenter.origin.y = 0
        }

        imageView.frame = frameToCenter
    }

    // MARK: - UIScrollViewDelegate

    func viewForZooming(in scrollView: UIScrollView) -> UIView? { imageView }

    func scrollViewDidZoom(_ scrollView: UIScrollView) {
        centerImage()
    }

    // MARK: - UIGestureRecognizerDelegate

    func gestureRecognizer(
        _ gestureRecognizer: UIGestureRecognizer,
        shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer
    ) -> Bool {
        // Block rotation gestures
        if gestureRecognizer is UIRotationGestureRecognizer || otherGestureRecognizer is UIRotationGestureRecognizer {
            return false
        }
        return true
    }
}

// MARK: - Thumbnail Cell

private final class ThumbnailCell: UICollectionViewCell {
    static let reuseId = "ThumbnailCell"
    var assetId: String?
    let imageView = UIImageView()
    private let borderView = UIView()

    override init(frame: CGRect) {
        super.init(frame: frame)
        imageView.contentMode = .scaleAspectFill
        imageView.clipsToBounds = true
        imageView.frame = contentView.bounds
        imageView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        contentView.addSubview(imageView)

        borderView.frame = contentView.bounds
        borderView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        borderView.layer.borderColor = UIColor.white.cgColor
        borderView.layer.borderWidth = 0
        borderView.isUserInteractionEnabled = false
        contentView.addSubview(borderView)
    }

    required init?(coder: NSCoder) { fatalError() }

    override func prepareForReuse() {
        super.prepareForReuse()
        imageView.image = nil
        assetId = nil
        setSelected(false)
    }

    func setSelected(_ selected: Bool) {
        borderView.layer.borderWidth = selected ? 2 : 0
    }
}
