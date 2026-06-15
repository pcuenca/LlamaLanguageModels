# LlamaLanguageModels

`FoundationModels.LanguageModel` support for llama.cpp. Leverages the `LanguageModel` and `LanguageModelExecutor` protocols introduced in WWDC 2026 to provide an unified API to llama.cpp GGUF files downloaded from Hugging Face.

Built on [LlamaKit (experimental)](https://github.com/pcuenca/llamakit).

## Usage

```swift
import FoundationModels
import LlamaLanguageModels

let model = LlamaLanguageModel(
    modelIdentifier: "Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M"
)
let session = LanguageModelSession(model: model)
let response = try await session.respond(to: "Who are you?")
print(response.content)
```

## Example CLI: `fm_llama`

[`Sources/fm_llama/`](https://github.com/pcuenca/LlamaLanguageModels/blob/main/Sources/fm_llama/Main.swift) is a minimal REPL with streaming.

```bash
swift run fm_llama ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M "Respond in verse"
```

## Requirements

- Swift 6.4+
- macOS 27+ / iOS 27+ / Xcode 27.0 beta

## To Do

- [ ] Default model generation parameters
- [ ] Faithful token counts
- [ ] Tool calling
- [ ] Reasoning
- [ ] Constrained generation

