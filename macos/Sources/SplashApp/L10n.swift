import Foundation

/// Follows the system interface language through the localization tables the
/// packager places in the app bundle (en.lproj and zh-Hans.lproj). When the
/// executable runs without a bundle — `swift run`, or a missing table — a key
/// falls back to itself, which reads like the English default.
enum L10n {
    static func string(_ key: String) -> String {
        Bundle.main.localizedString(forKey: key, value: key, table: "Localizable")
    }

    static func format(_ key: String, _ arguments: CVarArg...) -> String {
        String(format: string(key), arguments: arguments)
    }
}