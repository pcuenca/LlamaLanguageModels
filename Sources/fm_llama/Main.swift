import Foundation
import FoundationModels
import LlamaLanguageModels

@main
struct FMLlama {
    static func main() async throws {
        let args = Array(CommandLine.arguments.dropFirst())
        guard let modelIdentifier = args.first else {
            FileHandle.standardError.write(
                Data("usage: fm_llama <owner/repo[:QUANT]> [instructions...]\n".utf8))
            exit(EXIT_FAILURE)
        }

        let instructions = args.dropFirst().joined(separator: " ")
        let model = LlamaLanguageModel(modelIdentifier: modelIdentifier)
        let session = LanguageModelSession(
            model: model,
            instructions: instructions.isEmpty ? nil : instructions
        )

        printToStderr("Connected.\n")

        while true {
            printToStderr("> ")
            guard let line = readLine() else { break }
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { continue }
            do {
                let response = try await session.respond(to: trimmed)
                print(response.content)
            } catch {
                printToStderr("error: \(error)\n")
            }
        }
    }

    private static func printToStderr(_ s: String) {
        FileHandle.standardError.write(Data(s.utf8))
    }
}
