// swift-tools-version:5.7
import PackageDescription

let package = Package(
    name: "DebugBias",
    platforms: [.macOS(.v13)],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "DebugBias",
            dependencies: [
                .product(name: "FlashAttention", package: "metal-flash-attention")
            ],
            path: "."
        )
    ]
)
