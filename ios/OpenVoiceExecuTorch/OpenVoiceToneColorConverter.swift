import Foundation

public final class OpenVoiceToneColorConverter {
    private let config: OpenVoiceConfig
    private let voiceConversionRunner: ExecuTorchRunner
    private let referenceEncoderRunner: ExecuTorchRunner?

    public init(config: OpenVoiceConfig, voiceConversionModelPath: String, referenceEncoderModelPath: String?) throws {
        self.config = config
        self.voiceConversionRunner = try ExecuTorchRunner(modelPath: voiceConversionModelPath)
        if let referenceEncoderModelPath {
            self.referenceEncoderRunner = try ExecuTorchRunner(modelPath: referenceEncoderModelPath)
        } else {
            self.referenceEncoderRunner = nil
        }
    }

    public func extractSpeakerEmbedding(audio: [Float], tau: Float = 0.3) throws -> [Float] {
        guard let referenceEncoderRunner else {
            return []
        }
        let spec = spectrogram(audio: audio)
        let freqBins = config.specChannels
        let timeFrames = spec.count / freqBins
        let specTimeMajor = transposeMatrix(spec, rows: freqBins, cols: timeFrames)
        let specTensor = ExecuTorchTensorFactory.floatTensor(
            data: specTimeMajor,
            shape: [1, timeFrames, freqBins]
        )
        let outputs = try referenceEncoderRunner.run(inputs: [specTensor])
        guard let embedding = outputs.first else {
            return []
        }
        return embedding.toFloatArray()
    }

    public func convert(
        audio: [Float],
        srcEmbedding: [Float],
        tgtEmbedding: [Float],
        tau: Float = 0.3
    ) throws -> [Float] {
        let spec = spectrogram(audio: audio)
        let freqBins = config.specChannels
        let timeFrames = spec.count / freqBins
        let specTensor = ExecuTorchTensorFactory.floatTensor(
            data: spec,
            shape: [1, freqBins, timeFrames]
        )
        let lengths = ExecuTorchTensorFactory.int64Tensor(data: [Int64(timeFrames)], shape: [1])
        let srcTensor = ExecuTorchTensorFactory.floatTensor(
            data: srcEmbedding,
            shape: [1, config.model.gin_channels, 1]
        )
        let tgtTensor = ExecuTorchTensorFactory.floatTensor(
            data: tgtEmbedding,
            shape: [1, config.model.gin_channels, 1]
        )
        let tauTensor = ExecuTorchTensorFactory.floatTensor(data: [tau], shape: [])

        let outputs = try voiceConversionRunner.run(inputs: [
            specTensor,
            lengths,
            srcTensor,
            tgtTensor,
            tauTensor
        ])
        guard let audioTensor = outputs.first else {
            return []
        }
        return audioTensor.toFloatArray()
    }

    private func spectrogram(audio: [Float]) -> [Float] {
        return Spectrogram.compute(
            audio: audio,
            nFFT: config.data.filter_length,
            hopLength: config.data.hop_length,
            winLength: config.data.win_length
        )
    }

    private func transposeMatrix(_ data: [Float], rows: Int, cols: Int) -> [Float] {
        var output = [Float](repeating: 0, count: data.count)
        for row in 0..<rows {
            for col in 0..<cols {
                output[col * rows + row] = data[row * cols + col]
            }
        }
        return output
    }
}
