import UIKit

final class FaceClusterCell: UICollectionViewCell {
    static let reuseId = "FaceClusterCell"
    var clusterId: Int?

    private let faceImageView = UIImageView()
    private let gradientLayer = CAGradientLayer()
    private let nameLabel = UILabel()
    private let countLabel = UILabel()

    override init(frame: CGRect) {
        super.init(frame: frame)
        contentView.backgroundColor = .secondarySystemFill
        contentView.clipsToBounds = true

        faceImageView.contentMode = .scaleAspectFill
        faceImageView.clipsToBounds = true
        faceImageView.frame = contentView.bounds
        faceImageView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        contentView.addSubview(faceImageView)

        gradientLayer.colors = [UIColor.clear.cgColor, UIColor.black.withAlphaComponent(0.6).cgColor]
        gradientLayer.locations = [0.0, 1.0]
        contentView.layer.addSublayer(gradientLayer)

        nameLabel.font = .systemFont(ofSize: 13, weight: .semibold)
        nameLabel.textColor = .white
        nameLabel.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(nameLabel)

        countLabel.font = .systemFont(ofSize: 11)
        countLabel.textColor = UIColor.white.withAlphaComponent(0.8)
        countLabel.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(countLabel)

        NSLayoutConstraint.activate([
            countLabel.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -6),
            countLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 6),
            countLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -6),
            nameLabel.bottomAnchor.constraint(equalTo: countLabel.topAnchor, constant: -1),
            nameLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 6),
            nameLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -6),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }

    override func layoutSubviews() {
        super.layoutSubviews()
        let h = contentView.bounds.height
        gradientLayer.frame = CGRect(
            x: 0, y: h * 0.6,
            width: contentView.bounds.width, height: h * 0.4
        )
    }

    override func prepareForReuse() {
        super.prepareForReuse()
        faceImageView.image = nil
        clusterId = nil
        nameLabel.text = nil
        countLabel.text = nil
        gradientLayer.isHidden = false
    }

    func configureCluster(name: String, count: Int) {
        nameLabel.text = name
        countLabel.text = count > 0 ? "\(count) faces" : ""
        gradientLayer.isHidden = name.isEmpty && count <= 0
    }

    func configureDetectedFacePreview() {
        nameLabel.text = nil
        countLabel.text = nil
        gradientLayer.isHidden = true
        clusterId = nil
    }

    func setFaceImage(_ image: UIImage?) {
        guard faceImageView.image !== image else { return }
        UIView.transition(
            with: faceImageView,
            duration: 0.18,
            options: .transitionCrossDissolve
        ) {
            self.faceImageView.image = image
        }
    }
}
