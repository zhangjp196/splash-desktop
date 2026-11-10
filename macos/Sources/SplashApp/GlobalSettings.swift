import Foundation

/// The settings every configuration shares: the endpoint the server listens
/// on, the API's key and request limits, and what the cache does while the
/// server is idle. They live on the Settings page rather than in a model
/// configuration, because they describe the app's server, not the model it
/// happens to serve. How the engine runs — precision, admission and thinking
/// — belongs to the configuration instead.
///
/// `ServerSettings` still carries the whole command line, so the store, the
/// launch arguments and the selfcheck keep one value to pass around. This
/// value is the share that belongs to the app, and `applying(to:)` is how it
/// reaches that value at a start.
struct GlobalSettings: Codable, Equatable {
    static let `default` = GlobalSettings()

    var port = ServerSettings.First.port
    /// Listen on every IPv4 interface (`--host 0.0.0.0`) so other devices on
    /// the network can reach the server. Off keeps it on loopback.
    var lanAccess = false
    /// Extra HTTP Host names to accept (`--allowed-host`), comma-separated.
    var allowedHosts = ""
    /// Extra browser origins whose pages may call the API
    /// (`--allowed-origin`), comma-separated; `*` allows any.
    var allowedOrigins = ""

    /// Whether idle weights are released after `residencySeconds`.
    var unloadIdleWeights = ServerSettings.First.unloadIdleWeights
    var residencySeconds = ServerSettings.First.residencySeconds
    /// How long the buffers stay wired without work, apart from the weights'
    /// release: a duration, empty to follow the idle release, or `off`.
    var residencyKeepAlive = ServerSettings.First.residencyKeepAlive
    /// When the engine writes cached prefixes to disk and releases their
    /// memory (`--cache-idle-spill`): a duration, empty for 5 minutes, `off`
    /// to keep them.
    var cacheIdleSpill = ""
    /// Whether an idle pass with no writable disk tier evicts the cached
    /// prefixes.
    var cacheIdleEvict = ServerSettings.First.cacheIdleEvict
    var cacheDir = ""
    /// Whether the engine may use the SSD for cached KV pages and states.
    var diskCache = ServerSettings.First.diskCache
    var maxCacheDisk = ServerSettings.First.maxCacheDisk
    var maxCacheDiskWrites = ServerSettings.First.maxCacheDiskWrites
    var cacheEvictionPreset = ServerSettings.First.cacheEvictionPreset
    var cacheEvictionPolicy = ServerSettings.First.cacheEvictionPolicy
    var cacheEvictionProtect = ServerSettings.First.cacheEvictionProtect

    /// The API's access and request policy.
    var apiKey = ""
    var servedNames = ""
    var announceServedName = false
    var maxRequestSize = ""
    /// How long a queued or in-flight request may run (`--request-timeout`):
    /// a duration such as `30m`, empty for no limit.
    var requestTimeout = ""
    /// The largest resized image (`--max-image-pixels`), 65,536 to 4,194,304.
    var maxImagePixels = ""
    /// Let the Mac sleep while requests run (`--allow-idle-sleep`).
    var allowIdleSleep = false

    init() {}

    /// The app-owned share of a complete configuration, as the store and the
    /// model library keep it.
    init(_ settings: ServerSettings) {
        port = settings.port
        lanAccess = settings.lanAccess
        allowedHosts = settings.allowedHosts
        allowedOrigins = settings.allowedOrigins
        unloadIdleWeights = settings.unloadIdleWeights
        residencySeconds = settings.residencySeconds
        residencyKeepAlive = settings.residencyKeepAlive
        cacheIdleSpill = settings.cacheIdleSpill
        cacheIdleEvict = settings.cacheIdleEvict
        cacheDir = settings.cacheDir
        diskCache = settings.diskCache
        maxCacheDisk = settings.maxCacheDisk
        maxCacheDiskWrites = settings.maxCacheDiskWrites
        cacheEvictionPreset = settings.cacheEvictionPreset
        cacheEvictionPolicy = settings.cacheEvictionPolicy
        cacheEvictionProtect = settings.cacheEvictionProtect
        apiKey = settings.apiKey
        servedNames = settings.servedNames
        announceServedName = settings.announceServedName
        maxRequestSize = settings.maxRequestSize
        requestTimeout = settings.requestTimeout
        maxImagePixels = settings.maxImagePixels
        allowIdleSleep = settings.allowIdleSleep
    }

    /// Copy this share onto a complete configuration, leaving the model's own
    /// fields as they are.
    func applying(to settings: inout ServerSettings) {
        settings.port = port
        settings.lanAccess = lanAccess
        settings.allowedHosts = allowedHosts
        settings.allowedOrigins = allowedOrigins
        settings.unloadIdleWeights = unloadIdleWeights
        settings.residencySeconds = residencySeconds
        settings.residencyKeepAlive = residencyKeepAlive
        settings.cacheIdleSpill = cacheIdleSpill
        settings.cacheIdleEvict = cacheIdleEvict
        settings.cacheDir = cacheDir
        settings.diskCache = diskCache
        settings.maxCacheDisk = maxCacheDisk
        settings.maxCacheDiskWrites = maxCacheDiskWrites
        settings.cacheEvictionPreset = cacheEvictionPreset
        settings.cacheEvictionPolicy = cacheEvictionPolicy
        settings.cacheEvictionProtect = cacheEvictionProtect
        settings.apiKey = apiKey
        settings.servedNames = servedNames
        settings.announceServedName = announceServedName
        settings.maxRequestSize = maxRequestSize
        settings.requestTimeout = requestTimeout
        settings.maxImagePixels = maxImagePixels
        settings.allowIdleSleep = allowIdleSleep
    }
}
