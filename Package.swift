// swift-tools-version: 6.4
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "LlamaLanguageModels",
    platforms: [
        .macOS(.v27),
        .iOS(.v27),
    ],
    products: [
        .library(name: "LlamaLanguageModels", targets: ["LlamaLanguageModels"]),
    ],
    dependencies: [
        .package(path: "../LlamaKit"),
    ],
    targets: [
        .target(name: "LlamaLanguageModels", dependencies: ["LlamaKit"]),
        .executableTarget(
            name: "fm_llama",
            dependencies: ["LlamaLanguageModels"],
            path: "Sources/fm_llama"
        ),
        .testTarget(name: "LlamaLanguageModelsTests", dependencies: ["LlamaLanguageModels"]),
    ],
    swiftLanguageModes: [.v6]
)
