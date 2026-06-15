import Foundation
import FoundationModels
import LlamaLanguageModels

@main
struct LlamaChat {
    static func main() async throws {
        let args = Array(CommandLine.arguments.dropFirst())
        guard args.count >= 2 else {
            FileHandle.standardError.write(
                Data("usage: llamachat <owner/repo[:QUANT]> <prompt>\n".utf8))
            exit(EXIT_FAILURE)
        }

        let modelIdentifier = args[0]
        let prompt = args.dropFirst().joined(separator: " ")

        let model = LlamaLanguageModel(modelIdentifier: modelIdentifier)
        let session = LanguageModelSession(model: model)
        let response = try await session.respond(to: prompt)
        print(response.content)
    }
}
