import UIKit

enum ProcessingPipelineStep: Int, CaseIterable {
    case scan
    case graph
    case merge
    case save

    var title: String {
        switch self {
        case .scan:
            return "Scan Photos"
        case .graph:
            return "Build Graph"
        case .merge:
            return "Merge Faces"
        case .save:
            return "Save Results"
        }
    }

    var tintColor: UIColor {
        switch self {
        case .scan:
            return .systemBlue
        case .graph:
            return .systemPurple
        case .merge:
            return .systemGreen
        case .save:
            return .systemOrange
        }
    }
}

struct ProcessingPipelineStepState {
    let title: String
    let detail: String?
    let progress: Float
    let tintColor: UIColor
    let isDimmed: Bool
}

struct ProcessingPipelineHeaderPresentation {
    let isHidden: Bool
    let steps: [ProcessingPipelineStepState]
    let badgeText: String?
    let summaryText: String?
}

final class ProcessingPipelineHeaderView: UIView {
    private let stackView = UIStackView()
    private let badgeLabel = PipelineBadgeLabel()
    private let summaryLabel = UILabel()
    private let rowViews = ProcessingPipelineStep.allCases.map { _ in ProcessingPipelineRowView() }

    override init(frame: CGRect) {
        super.init(frame: frame)
        translatesAutoresizingMaskIntoConstraints = false

        stackView.axis = .vertical
        stackView.spacing = 8
        stackView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(stackView)

        for rowView in rowViews {
            stackView.addArrangedSubview(rowView)
        }

        badgeLabel.isHidden = true
        badgeLabel.translatesAutoresizingMaskIntoConstraints = false
        addSubview(badgeLabel)

        summaryLabel.font = .preferredFont(forTextStyle: .caption1)
        summaryLabel.textColor = .secondaryLabel
        summaryLabel.translatesAutoresizingMaskIntoConstraints = false
        summaryLabel.numberOfLines = 1
        addSubview(summaryLabel)

        NSLayoutConstraint.activate([
            stackView.topAnchor.constraint(equalTo: topAnchor),
            stackView.leadingAnchor.constraint(equalTo: leadingAnchor),
            stackView.trailingAnchor.constraint(equalTo: trailingAnchor),

            badgeLabel.topAnchor.constraint(equalTo: stackView.bottomAnchor, constant: 8),
            badgeLabel.leadingAnchor.constraint(equalTo: leadingAnchor),

            summaryLabel.centerYAnchor.constraint(equalTo: badgeLabel.centerYAnchor),
            summaryLabel.leadingAnchor.constraint(equalTo: badgeLabel.trailingAnchor, constant: 10),
            summaryLabel.trailingAnchor.constraint(equalTo: trailingAnchor),
            summaryLabel.bottomAnchor.constraint(equalTo: bottomAnchor),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }

    func apply(_ presentation: ProcessingPipelineHeaderPresentation) {
        isHidden = presentation.isHidden

        for (rowView, state) in zip(rowViews, presentation.steps) {
            rowView.apply(state)
        }

        if let badgeText = presentation.badgeText, !badgeText.isEmpty {
            badgeLabel.isHidden = false
            badgeLabel.text = badgeText
        } else {
            badgeLabel.isHidden = true
            badgeLabel.text = nil
        }

        summaryLabel.text = presentation.summaryText
        summaryLabel.isHidden = (presentation.summaryText?.isEmpty ?? true) && badgeLabel.isHidden
    }
}

private final class ProcessingPipelineRowView: UIView {
    private let titleLabel = UILabel()
    private let detailLabel = UILabel()
    private let progressView = UIProgressView(progressViewStyle: .default)

    override init(frame: CGRect) {
        super.init(frame: frame)
        translatesAutoresizingMaskIntoConstraints = false

        titleLabel.font = .preferredFont(forTextStyle: .caption1).withWeight(.semibold)
        titleLabel.textColor = .label
        titleLabel.translatesAutoresizingMaskIntoConstraints = false

        detailLabel.font = .preferredFont(forTextStyle: .caption2)
        detailLabel.textColor = .secondaryLabel
        detailLabel.textAlignment = .right
        detailLabel.translatesAutoresizingMaskIntoConstraints = false

        progressView.trackTintColor = .tertiarySystemFill
        progressView.translatesAutoresizingMaskIntoConstraints = false

        addSubview(titleLabel)
        addSubview(detailLabel)
        addSubview(progressView)

        NSLayoutConstraint.activate([
            titleLabel.topAnchor.constraint(equalTo: topAnchor),
            titleLabel.leadingAnchor.constraint(equalTo: leadingAnchor),

            detailLabel.firstBaselineAnchor.constraint(equalTo: titleLabel.firstBaselineAnchor),
            detailLabel.leadingAnchor.constraint(greaterThanOrEqualTo: titleLabel.trailingAnchor, constant: 12),
            detailLabel.trailingAnchor.constraint(equalTo: trailingAnchor),

            progressView.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 4),
            progressView.leadingAnchor.constraint(equalTo: leadingAnchor),
            progressView.trailingAnchor.constraint(equalTo: trailingAnchor),
            progressView.bottomAnchor.constraint(equalTo: bottomAnchor),
            progressView.heightAnchor.constraint(equalToConstant: 4),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }

    func apply(_ state: ProcessingPipelineStepState) {
        titleLabel.text = state.title
        detailLabel.text = state.detail
        progressView.progress = state.progress
        progressView.progressTintColor = state.tintColor

        let alpha: CGFloat = state.isDimmed ? 0.45 : 1
        titleLabel.alpha = alpha
        detailLabel.alpha = alpha
        progressView.alpha = max(alpha, 0.35)
    }
}

private final class PipelineBadgeLabel: UILabel {
    override init(frame: CGRect) {
        super.init(frame: frame)
        font = .preferredFont(forTextStyle: .caption2).withWeight(.semibold)
        textColor = .systemGreen
        backgroundColor = UIColor.systemGreen.withAlphaComponent(0.12)
        layer.cornerRadius = 10
        layer.masksToBounds = true
    }

    required init?(coder: NSCoder) { fatalError() }

    override func drawText(in rect: CGRect) {
        super.drawText(in: rect.insetBy(dx: 8, dy: 4))
    }

    override var intrinsicContentSize: CGSize {
        let size = super.intrinsicContentSize
        return CGSize(width: size.width + 16, height: size.height + 8)
    }
}

private extension UIFont {
    func withWeight(_ weight: UIFont.Weight) -> UIFont {
        let descriptor = fontDescriptor.addingAttributes([
            .traits: [UIFontDescriptor.TraitKey.weight: weight]
        ])
        return UIFont(descriptor: descriptor, size: pointSize)
    }
}
