import Foundation

#if canImport(ExecuTorch)
import ExecuTorch

public extension ExecutorchTensor {
    func toFloatArray() -> [Float] {
        guard let data = self.data as? [Float] else {
            return []
        }
        return data
    }
}
#else
public extension ExecuTorchTensor {
    func toFloatArray() -> [Float] {
        return data
    }
}
#endif
