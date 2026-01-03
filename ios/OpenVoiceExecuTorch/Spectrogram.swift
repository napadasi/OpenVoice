import Accelerate
import Foundation

public enum Spectrogram {
    public static func compute(
        audio: [Float],
        nFFT: Int,
        hopLength: Int,
        winLength: Int
    ) -> [Float] {
        let padAmount = (nFFT - hopLength) / 2
        let padded = reflectPad(audio, pad: padAmount)
        let frameCount = max(0, 1 + (padded.count - winLength) / hopLength)
        let freqBins = nFFT / 2 + 1

        var window = [Float](repeating: 0, count: winLength)
        vDSP_hann_window(&window, vDSP_Length(winLength), Int32(vDSP_HANN_NORM))

        var magnitudes = [Float](repeating: 0, count: frameCount * freqBins)
        var fftSetup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(nFFT), vDSP_DFT_Direction.FORWARD)
        guard let setup = fftSetup else {
            return magnitudes
        }
        defer {
            vDSP_DFT_DestroySetup(setup)
        }

        var real = [Float](repeating: 0, count: nFFT)
        var imag = [Float](repeating: 0, count: nFFT)

        for frame in 0..<frameCount {
            let start = frame * hopLength
            let frameSlice = padded[start..<(start + winLength)]
            var input = [Float](repeating: 0, count: nFFT)
            for i in 0..<winLength {
                input[i] = frameSlice[frameSlice.index(frameSlice.startIndex, offsetBy: i)] * window[i]
            }
            real.withUnsafeMutableBufferPointer { realPtr in
                imag.withUnsafeMutableBufferPointer { imagPtr in
                    input.withUnsafeBufferPointer { inputPtr in
                        vDSP_DFT_Execute(setup, inputPtr.baseAddress!, nil, realPtr.baseAddress!, imagPtr.baseAddress!)
                    }
                }
            }

            for bin in 0..<freqBins {
                let re = real[bin]
                let im = imag[bin]
                let mag = sqrt(re * re + im * im + 1e-6)
                magnitudes[bin * frameCount + frame] = mag
            }
        }
        return magnitudes
    }

    private static func reflectPad(_ input: [Float], pad: Int) -> [Float] {
        guard pad > 0 else { return input }
        let count = input.count
        var output = [Float](repeating: 0, count: count + pad * 2)
        for index in -pad..<(count + pad) {
            let sourceIndex = reflectIndex(index, count: count)
            output[index + pad] = input[sourceIndex]
        }
        return output
    }

    private static func reflectIndex(_ index: Int, count: Int) -> Int {
        if count == 1 { return 0 }
        if index < 0 {
            return -index
        }
        if index >= count {
            return 2 * count - 2 - index
        }
        return index
    }
}
