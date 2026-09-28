import Foundation

/// Follows the interface language: the leading (non-system) choice from
/// `languageOverride`, else the system preference, through the localization
/// tables the packager places in the app bundle (en.lproj and zh-Hans.lproj).
/// When the executable runs without a bundle — `swift run`, or a missing
/// table — a key falls back to itself, which reads like the English default.
enum L10n {
    /// "zh-Hans", "en" or nil to follow the system.
    static var languageOverride: String?

    private static var bundle: Bundle {
        if let override = languageOverride,
           let path = Bundle.main.path(forResource: override, ofType: "lproj"),
           let bundle = Bundle(path: path) {
            return bundle
        }
        return Bundle.main
    }

    static func string(_ key: String) -> String {
        bundle.localizedString(forKey: key, value: key, table: "Localizable")
    }

    static func format(_ key: String, _ arguments: CVarArg...) -> String {
        String(format: string(key), arguments: arguments)
    }
}