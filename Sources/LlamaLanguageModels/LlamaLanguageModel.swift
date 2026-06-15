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

    // backendLogsEnabled is set on model init but it's a global llama.cpp engine flag: last call wins
    public init(
        modelIdentifier: String,
        capabilities: LanguageModelCapabilities = .init(capabilities: []),
        backendLogsEnabled: Bool = false
    ) {
        LlamaBackend.loggingEnabled = backendLogsEnabled
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
    func loadedModel() async throws -> LlamaModel {
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
        private var loaded: LlamaModel?
        private var inflight: Task<LlamaKit.LlamaModel, Error>?

        init(modelIdentifier: String) {
            self.modelIdentifier = modelIdentifier
        }

        func load() async throws -> LlamaModel {
            if let loaded { return loaded }
            if let inflight { return try await inflight.value }

            let id = modelIdentifier
            let task = Task<LlamaModel, Error> {
                let (repo, filename) = LlamaLanguageModel.parse(id)
                return try await LlamaModel.from(repo: repo, filename: filename)
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
            // Error on unimplemented features
            if request.schema != nil {
                throw LanguageModelError.unsupportedCapability(
                    .init(
                        capability: .guidedGeneration,
                        debugDescription:
                            "LlamaLanguageModel does not yet support schema-guided generation."
                    )
                )
            }
            if !request.enabledToolDefinitions.isEmpty {
                throw LanguageModelError.unsupportedCapability(
                    .init(
                        capability: .toolCalling,
                        debugDescription:
                            "LlamaLanguageModel does not yet support tool calling."
                    )
                )
            }
            if request.contextOptions.reasoningLevel != nil {
                throw LanguageModelError.unsupportedCapability(
                    .init(
                        capability: .reasoning,
                        debugDescription:
                            "LlamaLanguageModel does not yet support reasoning."
                    )
                )
            }
            if Self.transcriptContainsAttachments(request.transcript) {
                throw LanguageModelError.unsupportedCapability(
                    .init(
                        capability: .vision,
                        debugDescription:
                            "LlamaLanguageModel does not yet support vision or attachment inputs."
                    )
                )
            }

            let llamaModel = try await model.loadedModel()
            let session = try LlamaSession(model: llamaModel)

            let messages = Self.chatMessages(from: request.transcript)
            let prompt = try llamaModel.tokenizer.applyChatTemplate(messages)

            let sampler = Self.makeSampler(from: request.generationOptions)
            let maxTokens = request.generationOptions.maximumResponseTokens ?? 512

            // Each respond call publishes under a fresh entryID
            let entryID = UUID().uuidString

            for try await chunk in session.generate(
                prompt: prompt,
                sampler: sampler,
                maxTokens: maxTokens
            ) {
                await channel.send(
                    .response(
                        entryID: entryID,
                        action: .appendText(chunk, tokenCount: 1)
                    )
                )
            }
        }

        // MARK: - Transcript → LlamaKit.ChatMessage

        private static func chatMessages(from transcript: Transcript) -> [ChatMessage] {
            var messages: [ChatMessage] = []
            for entry in transcript {
                switch entry {
                case .instructions(let instructions):
                    let text = textContent(from: instructions.segments)
                    if !text.isEmpty { messages.append(.system(text)) }
                case .prompt(let prompt):
                    let text = textContent(from: prompt.segments)
                    if !text.isEmpty { messages.append(.user(text)) }
                case .response(let response):
                    let text = textContent(from: response.segments)
                    if !text.isEmpty { messages.append(.assistant(text)) }
                default:
                    break
                }
            }
            return messages
        }

        private static func transcriptContainsAttachments(_ transcript: Transcript) -> Bool {
            for entry in transcript {
                let segments: [Transcript.Segment]
                switch entry {
                case .instructions(let i): segments = i.segments
                case .prompt(let p): segments = p.segments
                case .response(let r): segments = r.segments
                default: continue
                }
                for segment in segments {
                    if case .attachment = segment {
                        return true
                    }
                }
            }
            return false
        }

        private static func textContent(from segments: [Transcript.Segment]) -> String {
            segments.compactMap { segment -> String? in
                if case .text(let t) = segment { return t.content }
                return nil
            }.joined()
        }

        // MARK: - GenerationOptions → LlamaKit.Sampler

        /// Maps FoundationModels' `GenerationOptions` onto a LlamaKit `Sampler`
        /// Defaults temperature to 0.8 if unspecified
        private static func makeSampler(from options: GenerationOptions) -> Sampler {
            let temperature = options.temperature.map(Float.init) ?? 0.8
            if temperature == 0 {
                return .greedy
            }

            if let kind = options.samplingMode?.kind {
                switch kind {
                case .greedy:
                    return .greedy
                case .top(let k, let seed):
                    let resolvedSeed = seed.map { UInt32(truncatingIfNeeded: $0) }
                        ?? .random(in: .min ... .max)
                    return Sampler(stages: [
                        .topK(Int32(k)),
                        .temperature(temperature),
                        .distribution(seed: resolvedSeed),
                    ])
                case .nucleus(let threshold, let seed):
                    let resolvedSeed = seed.map { UInt32(truncatingIfNeeded: $0) }
                        ?? .random(in: .min ... .max)
                    return Sampler(stages: [
                        .topP(Float(threshold), minKeep: 1),
                        .temperature(temperature),
                        .distribution(seed: resolvedSeed),
                    ])
                @unknown default:
                    break
                }
            }

            // TODO: samplingMode nil should probably use defaults from the model
            return .temperature(temperature)
        }
    }
}
