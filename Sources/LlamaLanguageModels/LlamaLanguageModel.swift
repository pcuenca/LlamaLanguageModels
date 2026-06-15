import Foundation
import FoundationModels
import LlamaKit

public struct LlamaLanguageModel: LanguageModel {
    public let modelIdentifier: String

    public let capabilities: LanguageModelCapabilities = .init(capabilities: [])
    
    public var executorConfiguration: Executor.Configuration {
        Executor.Configuration(modelIdentifier: modelIdentifier)
    }

    public init(modelIdentifier: String) {
        self.modelIdentifier = modelIdentifier
    }
    
    public struct Executor: LanguageModelExecutor {
        let modelIdentifier: String

        public struct Configuration: Hashable, Sendable {
            public let modelIdentifier: String
        }

        /// Creates a executor from a configuration.
        public init(configuration: Configuration) throws {
            self.modelIdentifier = configuration.modelIdentifier
        }
        
        public func respond(
            to request: LanguageModelExecutorGenerationRequest,
            model: LlamaLanguageModel,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
        }
    }
}
