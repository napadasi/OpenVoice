import Foundation

public struct OpenVoiceConfig: Decodable {
    public struct DataConfig: Decodable {
        public let filter_length: Int
        public let sampling_rate: Int
        public let hop_length: Int
        public let win_length: Int
        public let add_blank: Bool
        public let text_cleaners: [String]
        public let n_speakers: Int
    }

    public struct ModelConfig: Decodable {
        public let inter_channels: Int
        public let gin_channels: Int
    }

    public let data: DataConfig
    public let model: ModelConfig
    public let symbols: [String]?
    public let speakers: [String: Int]?

    public var specChannels: Int {
        return data.filter_length / 2 + 1
    }

    public static func load(from url: URL) throws -> OpenVoiceConfig {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(OpenVoiceConfig.self, from: data)
    }
}
