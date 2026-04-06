import UIKit

class SceneDelegate: UIResponder, UIWindowSceneDelegate {

    var window: UIWindow?
    private var photoManager: PhotoManager!
    private var faceProcessor: FaceProcessor!
    private var database: FaceDatabase!

    func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options connectionOptions: UIScene.ConnectionOptions) {
        guard let windowScene = (scene as? UIWindowScene) else { return }

        photoManager = PhotoManager()
        guard let db = try? FaceDatabase() else { return }
        database = db
        faceProcessor = FaceProcessor(database: database, photoManager: photoManager)

        let photosVC = PhotosViewController(photoManager: photoManager, faceProcessor: faceProcessor, database: database)
        photosVC.tabBarItem = UITabBarItem(title: "Photos", image: UIImage(systemName: "photo.on.rectangle"), tag: 0)

        let facesVC = FacesViewController(faceProcessor: faceProcessor, photoManager: photoManager)
        facesVC.tabBarItem = UITabBarItem(title: "Faces", image: UIImage(systemName: "faceid"), tag: 1)

        let embeddingsVC = EmbeddingsViewController(faceProcessor: faceProcessor, photoManager: photoManager)
        embeddingsVC.tabBarItem = UITabBarItem(title: "Graphs", image: UIImage(systemName: "chart.dots.scatter"), tag: 2)

        let photosNav = UINavigationController(rootViewController: photosVC)
        photosNav.navigationBar.prefersLargeTitles = true

        let facesNav = UINavigationController(rootViewController: facesVC)
        facesNav.navigationBar.prefersLargeTitles = true

        let embeddingsNav = UINavigationController(rootViewController: embeddingsVC)
        embeddingsNav.navigationBar.prefersLargeTitles = true

        let tabBar = UITabBarController()
        tabBar.viewControllers = [photosNav, facesNav, embeddingsNav]

        let window = UIWindow(windowScene: windowScene)
        window.rootViewController = tabBar
        window.makeKeyAndVisible()
        self.window = window

        photoManager.checkAuthorization()
    }
}
