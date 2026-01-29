// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MFABridge",
    platforms: [.macOS(.v14)],
    products: [
        .library(
            name: "MFABridge",
            type: .dynamic,
            targets: ["MFABridge"]),
    ],
    dependencies: [
        .package(path: "../metal-flash-attention"),
    ],
    targets: [
        .target(
            name: "MFABridge",
            dependencies: [
                .product(name: "FlashAttention", package: "metal-flash-attention"),
            ],
            swiftSettings: [
                .unsafeFlags(["-Ounchecked"])
            ]
        ),
    ]
)
