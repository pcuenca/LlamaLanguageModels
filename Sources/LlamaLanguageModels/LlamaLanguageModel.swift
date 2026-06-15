import Foundation
import FoundationModels
import LlamaKit

/// A `FoundationModels.LanguageModel` backed by LlamaKit.
///
/// The lifetime of the llama.cpp model follows the instance, there's no shared cache.
public final class LlamaLanguageModel: LanguageModel {
    /// `owner/repo:QUANT` — e.g. `"Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q5_0"`.
    /// When `:QUANT` is omitted, defaults to `Q4_K_M`.
    public let modelIdentifier: String

    public let capabilities: LanguageModelCapabilities

    private let loader: Loader

    public init(
        modelIdentifier: String,
        capabilities: LanguageModelCapabilities = .init(capabilities: [])
    ) {
        self.modelIdentifier = modelIdentifier
        self.capabilities = capabilities
        self.loader = Loader(modelIdentifier: modelIdentifier)
    }

    public var executorConfiguration: Executor.Configuration {
        .init(modelIdentifier: modelIdentifier)
    }

    public func preload() async throws {
        _ = try await loader.load()
    }

    /// Internal accessor used by the executor to obtain the loaded model.
    func loadedModel() async throws -> LlamaKit.LlamaModel {
        try await loader.load()
    }

    /// Splits `owner/repo:QUANT` into (repo, glob). Defaults quant to `Q4_K_M`
    static func parse(_ identifier: String) -> (repo: String, filename: String) {
        if let colon = identifier.firstIndex(of: ":") {
            let repo = String(identifier[..<colon])
            let quant = String(identifier[identifier.index(after: colon)...])
            return (repo, "*\(quant).gguf")
        }
        return (identifier, "*Q4_K_M.gguf")
    }

    /// Serializes loading. The single in-flight `Task` deduplicates concurrent
    /// callers; once resolved the model is cached on this instance for the
    /// lifetime of the enclosing `LlamaLanguageModel`.
    private actor Loader {
        let modelIdentifier: String
        private var loaded: LlamaKit.LlamaModel?
        private var inflight: Task<LlamaKit.LlamaModel, Error>?

        init(modelIdentifier: String) {
            self.modelIdentifier = modelIdentifier
        }

        func load() async throws -> LlamaKit.LlamaModel {
            if let loaded { return loaded }
            if let inflight { return try await inflight.value }

            let id = modelIdentifier
            let task = Task<LlamaKit.LlamaModel, Error> {
                let (repo, filename) = LlamaLanguageModel.parse(id)
                return try await LlamaKit.LlamaModel.from(repo: repo, filename: filename)
            }
            inflight = task
            do {
                let model = try await task.value
                loaded = model
                inflight = nil
                return model
            } catch {
                inflight = nil
                throw error
            }
        }
    }

    public struct Executor: LanguageModelExecutor {
        public struct Configuration: Hashable, Sendable {
            public let modelIdentifier: String
        }

        let modelIdentifier: String

        public init(configuration: Configuration) throws {
            self.modelIdentifier = configuration.modelIdentifier
        }

        public func prewarm(model: LlamaLanguageModel, transcript: Transcript) {
            Task { try? await model.preload() }
        }

        public func respond(
            to request: LanguageModelExecutorGenerationRequest,
            model: LlamaLanguageModel,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            // Load (cold path) or reuse (warm path) the underlying model.
            _ = try await model.loadedModel()

            // TODO: render request.transcript → prompt via LlamaKit's tokenizer
            //       chat template, drive LlamaSession.generate(prompt:), and
            //       publish .response(.appendText(...)) events into `channel`.
        }
    }
}
