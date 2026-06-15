import Foundation
import FoundationModels

@main
struct FMChat {
    static func main() async throws {
        let prompt = CommandLine.arguments.dropFirst().joined(separator: " ")
        guard !prompt.isEmpty else {
            FileHandle.standardError.write(Data("usage: fmchat <prompt>\n".utf8))
            exit(EXIT_FAILURE)
        }

        let model = SystemLanguageModel()
        let session = LanguageModelSession(model: model)
        let response = try await session.respond(to: prompt)
        print(response.content)
    }
}
