import Foundation

#if canImport(ExecuTorch)
import ExecuTorch
#endif

public enum ExecuTorchError: Error {
    case runtimeUnavailable
    case invalidOutput
}

public final class ExecuTorchRunner {
    #if canImport(ExecuTorch)
    private let module: ExecutorchModule
    #endif

    public init(modelPath: String) throws {
        #if canImport(ExecuTorch)
        module = try ExecutorchModule(filePath: modelPath)
        #else
        throw ExecuTorchError.runtimeUnavailable
        #endif
    }

    public func run(inputs: [ExecuTorchTensor]) throws -> [ExecuTorchTensor] {
        #if canImport(ExecuTorch)
        return try module.forward(inputs)
        #else
        throw ExecuTorchError.runtimeUnavailable
        #endif
    }
}

#if canImport(ExecuTorch)
public typealias ExecuTorchTensor = ExecutorchTensor
#else
public struct ExecuTorchTensor {
    public let shape: [Int]
    public let data: [Float]

    public init(shape: [Int], data: [Float]) {
        self.shape = shape
        self.data = data
    }
}
#endif

public enum ExecuTorchTensorFactory {
    public static func floatTensor(data: [Float], shape: [Int]) -> ExecuTorchTensor {
        #if canImport(ExecuTorch)
        return ExecutorchTensor(data: data, shape: shape)
        #else
        return ExecuTorchTensor(shape: shape, data: data)
        #endif
    }

    public static func int64Tensor(data: [Int64], shape: [Int]) -> ExecuTorchTensor {
        #if canImport(ExecuTorch)
        return ExecutorchTensor(data: data, shape: shape)
        #else
        return ExecuTorchTensor(shape: shape, data: data.map { Float($0) })
        #endif
    }
}
