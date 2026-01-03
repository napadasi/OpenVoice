import Foundation

public final class OpenVoiceTTS {
    private let config: OpenVoiceConfig
    private let ttsRunner: ExecuTorchRunner

    private let languageMarks: [String: String] = [
        "english": "EN",
        "chinese": "ZH"
    ]

    public init(config: OpenVoiceConfig, ttsModelPath: String) throws {
        self.config = config
        self.ttsRunner = try ExecuTorchRunner(modelPath: ttsModelPath)
    }

    public func synthesize(
        text: String,
        speaker: String,
        language: String = "English",
        noiseScale: Float = 0.667,
        lengthScale: Float = 1.0,
        noiseScaleW: Float = 0.6,
        sdpRatio: Float = 0.2,
        speed: Float = 1.0
    ) throws -> [Float] {
        guard let mark = languageMarks[language.lowercased()],
              let speakerId = config.speakers?[speaker] else {
            return []
        }

        let texts = TextProcessing.splitSentences(text: text, languageTag: mark)
        var audioSegments: [[Float]] = []

        for sentence in texts {
            let normalized = sentence.replacingOccurrences(
                of: "([a-z])([A-Z])",
                with: "$1 $2",
                options: .regularExpression
            )
            let marked = "[\(mark)]\(normalized)[\(mark)]"
            let tokenIds = TextProcessing.getText(marked, config: config, isSymbol: false)

            let inputIds = ExecuTorchTensorFactory.int64Tensor(data: tokenIds, shape: [1, tokenIds.count])
            let lengths = ExecuTorchTensorFactory.int64Tensor(data: [Int64(tokenIds.count)], shape: [1])
            let sid = ExecuTorchTensorFactory.int64Tensor(data: [Int64(speakerId)], shape: [1])
            let noiseScaleTensor = ExecuTorchTensorFactory.floatTensor(data: [noiseScale], shape: [])
            let lengthScaleTensor = ExecuTorchTensorFactory.floatTensor(data: [1.0 / speed], shape: [])
            let noiseScaleWTensor = ExecuTorchTensorFactory.floatTensor(data: [noiseScaleW], shape: [])
            let sdpRatioTensor = ExecuTorchTensorFactory.floatTensor(data: [sdpRatio], shape: [])

            let outputs = try ttsRunner.run(inputs: [
                inputIds,
                lengths,
                sid,
                noiseScaleTensor,
                lengthScaleTensor,
                noiseScaleWTensor,
                sdpRatioTensor
            ])

            guard let audioTensor = outputs.first else {
                continue
            }
            audioSegments.append(audioTensor.toFloatArray())
        }

        return TextProcessing.audioConcat(
            segments: audioSegments,
            sampleRate: config.data.sampling_rate,
            speed: speed
        )
    }
}
