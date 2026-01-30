/**
 * MPS Flash Attention - PyTorch C++ Extension
 *
 * Bridges PyTorch MPS tensors to the MFA Swift library.
 */

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <dlfcn.h>
#include <string>
#include <vector>

// ============================================================================
// MFA Bridge Function Types
// ============================================================================

typedef bool (*mfa_init_fn)();
typedef void* (*mfa_create_kernel_fn)(int32_t, int32_t, int32_t, bool, bool, bool, bool);
typedef void* (*mfa_create_kernel_v2_fn)(int32_t, int32_t, int32_t, bool, bool, bool, bool, bool);
typedef void* (*mfa_create_kernel_v3_fn)(int32_t, int32_t, int32_t, bool, bool, bool, bool, bool, uint32_t);
typedef void* (*mfa_create_kernel_v4_fn)(int32_t, int32_t, int32_t, bool, bool, bool, bool, bool, uint32_t, uint16_t);
typedef void* (*mfa_create_kernel_v5_fn)(int32_t, int32_t, int32_t, bool, bool, bool, bool, bool, uint32_t, uint16_t, bool);
// New zero-sync encode functions that take PyTorch's command encoder
// Added mask_ptr and mask_offset parameters
typedef bool (*mfa_forward_encode_fn)(void*, void*, void*, void*, void*, void*, void*, void*,
                                       int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                       int32_t, int32_t);
typedef bool (*mfa_forward_encode_quantized_fn)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*,
                                                 int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                                 int32_t, int32_t);
typedef bool (*mfa_backward_encode_fn)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*,
                                        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                        int32_t, int32_t);
// Legacy sync functions (fallback)
typedef bool (*mfa_forward_fn)(void*, void*, void*, void*, void*, void*, void*,
                                int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                int32_t, int32_t);
typedef bool (*mfa_backward_fn)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*,
                                 int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                 int32_t, int32_t);
typedef void (*mfa_release_kernel_fn)(void*);

// Global function pointers
static mfa_init_fn g_mfa_init = nullptr;
static mfa_create_kernel_fn g_mfa_create_kernel = nullptr;
static mfa_create_kernel_v2_fn g_mfa_create_kernel_v2 = nullptr;
static mfa_create_kernel_v3_fn g_mfa_create_kernel_v3 = nullptr;
static mfa_create_kernel_v4_fn g_mfa_create_kernel_v4 = nullptr;
static mfa_create_kernel_v5_fn g_mfa_create_kernel_v5 = nullptr;
static mfa_forward_encode_fn g_mfa_forward_encode = nullptr;
static mfa_forward_encode_quantized_fn g_mfa_forward_encode_quantized = nullptr;
static mfa_backward_encode_fn g_mfa_backward_encode = nullptr;
static mfa_forward_fn g_mfa_forward = nullptr;
static mfa_backward_fn g_mfa_backward = nullptr;
static mfa_release_kernel_fn g_mfa_release_kernel = nullptr;
static void* g_dylib_handle = nullptr;
static bool g_initialized = false;

// ============================================================================
// Load MFA Bridge Library
// ============================================================================

// Get the directory containing this shared library
static std::string get_module_dir() {
    Dl_info info;
    if (dladdr((void*)get_module_dir, &info) && info.dli_fname) {
        std::string path(info.dli_fname);
        size_t last_slash = path.rfind('/');
        if (last_slash != std::string::npos) {
            return path.substr(0, last_slash);
        }
    }
    return ".";
}

static bool load_mfa_bridge() {
    if (g_dylib_handle) return true;

    // First check environment variable (highest priority)
    const char* mfa_path = getenv("MFA_BRIDGE_PATH");
    if (mfa_path) {
        g_dylib_handle = dlopen(mfa_path, RTLD_NOW);
        if (g_dylib_handle) return true;
    }

    // Get the directory containing this extension module
    std::string module_dir = get_module_dir();

    // Try paths relative to the module directory
    std::vector<std::string> paths = {
        module_dir + "/lib/libMFABridge.dylib",  // Bundled in wheel
        module_dir + "/../swift-bridge/.build/release/libMFABridge.dylib",  // Dev build
        "libMFABridge.dylib",  // Current directory fallback
    };

    for (const auto& path : paths) {
        g_dylib_handle = dlopen(path.c_str(), RTLD_NOW);
        if (g_dylib_handle) break;
    }

    if (!g_dylib_handle) {
        throw std::runtime_error(
            "Failed to load libMFABridge.dylib. Set MFA_BRIDGE_PATH environment variable.");
    }

    // Load function pointers
    g_mfa_init = (mfa_init_fn)dlsym(g_dylib_handle, "mfa_init");
    g_mfa_create_kernel = (mfa_create_kernel_fn)dlsym(g_dylib_handle, "mfa_create_kernel");
    g_mfa_create_kernel_v2 = (mfa_create_kernel_v2_fn)dlsym(g_dylib_handle, "mfa_create_kernel_v2");
    g_mfa_create_kernel_v3 = (mfa_create_kernel_v3_fn)dlsym(g_dylib_handle, "mfa_create_kernel_v3");
    g_mfa_create_kernel_v4 = (mfa_create_kernel_v4_fn)dlsym(g_dylib_handle, "mfa_create_kernel_v4");
    g_mfa_create_kernel_v5 = (mfa_create_kernel_v5_fn)dlsym(g_dylib_handle, "mfa_create_kernel_v5");
    g_mfa_forward_encode_quantized = (mfa_forward_encode_quantized_fn)dlsym(g_dylib_handle, "mfa_forward_encode_quantized");
    g_mfa_forward_encode = (mfa_forward_encode_fn)dlsym(g_dylib_handle, "mfa_forward_encode");
    g_mfa_backward_encode = (mfa_backward_encode_fn)dlsym(g_dylib_handle, "mfa_backward_encode");
    g_mfa_forward = (mfa_forward_fn)dlsym(g_dylib_handle, "mfa_forward");
    g_mfa_backward = (mfa_backward_fn)dlsym(g_dylib_handle, "mfa_backward");
    g_mfa_release_kernel = (mfa_release_kernel_fn)dlsym(g_dylib_handle, "mfa_release_kernel");

    // Require at least init, create_kernel, forward_encode (for zero-sync path)
    if (!g_mfa_init || !g_mfa_create_kernel || !g_mfa_forward_encode) {
        throw std::runtime_error("Failed to load MFA bridge functions");
    }

    return true;
}

// ============================================================================
// Get MTLBuffer from PyTorch MPS Tensor
// ============================================================================

struct BufferInfo {
    id<MTLBuffer> buffer;
    int64_t byte_offset;
};

static BufferInfo getBufferInfo(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.device().is_mps(), "Tensor must be on MPS device");
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");

    // Get the underlying Metal buffer (covers entire storage)
    id<MTLBuffer> buffer = at::native::mps::getMTLBufferStorage(tensor);

    // Calculate byte offset: storage_offset() is in elements, multiply by element size
    int64_t element_size = tensor.element_size();
    int64_t byte_offset = tensor.storage_offset() * element_size;

    return {buffer, byte_offset};
}

// ============================================================================
// Kernel Cache
// ============================================================================

struct KernelCacheKey {
    int64_t seq_len_q;
    int64_t seq_len_kv;
    int64_t head_dim;
    bool low_precision;
    bool low_precision_outputs;
    bool causal;
    bool has_mask;
    bool use_bf16;
    uint32_t window_size;
    uint16_t quantized_kv;  // 0 = none, 3 = FP8_E4M3, 4 = FP8_E5M2, 5 = INT8, 6 = NF4
    bool bf16_backward;     // true = use BF16 for backward intermediates (faster)

    bool operator==(const KernelCacheKey& other) const {
        return seq_len_q == other.seq_len_q &&
               seq_len_kv == other.seq_len_kv &&
               head_dim == other.head_dim &&
               low_precision == other.low_precision &&
               low_precision_outputs == other.low_precision_outputs &&
               causal == other.causal &&
               has_mask == other.has_mask &&
               use_bf16 == other.use_bf16 &&
               window_size == other.window_size &&
               quantized_kv == other.quantized_kv &&
               bf16_backward == other.bf16_backward;
    }
};

struct KernelCacheKeyHash {
    size_t operator()(const KernelCacheKey& k) const {
        return std::hash<int64_t>()(k.seq_len_q) ^
               (std::hash<int64_t>()(k.seq_len_kv) << 1) ^
               (std::hash<int64_t>()(k.head_dim) << 2) ^
               (std::hash<bool>()(k.low_precision) << 3) ^
               (std::hash<bool>()(k.low_precision_outputs) << 4) ^
               (std::hash<bool>()(k.causal) << 5) ^
               (std::hash<bool>()(k.has_mask) << 6) ^
               (std::hash<bool>()(k.use_bf16) << 7) ^
               (std::hash<uint32_t>()(k.window_size) << 8) ^
               (std::hash<uint16_t>()(k.quantized_kv) << 9) ^
               (std::hash<bool>()(k.bf16_backward) << 10);
    }
};

static std::unordered_map<KernelCacheKey, void*, KernelCacheKeyHash> g_kernel_cache;

static void* get_or_create_kernel(int64_t seq_q, int64_t seq_kv, int64_t head_dim, bool low_prec, bool low_prec_outputs, bool causal, bool has_mask, bool use_bf16 = false, uint32_t window_size = 0, uint16_t quantized_kv = 0, bool bf16_backward = false) {
    KernelCacheKey key{seq_q, seq_kv, head_dim, low_prec, low_prec_outputs, causal, has_mask, use_bf16, window_size, quantized_kv, bf16_backward};

    auto it = g_kernel_cache.find(key);
    if (it != g_kernel_cache.end()) {
        return it->second;
    }

    void* kernel = nullptr;
    if (g_mfa_create_kernel_v5) {
        // Use v5 API with all features including mixed-precision backward
        kernel = g_mfa_create_kernel_v5(
            static_cast<int32_t>(seq_q),
            static_cast<int32_t>(seq_kv),
            static_cast<int32_t>(head_dim),
            low_prec,
            low_prec_outputs,
            causal,
            has_mask,
            use_bf16,
            window_size,
            quantized_kv,
            bf16_backward
        );
    } else if (quantized_kv > 0 && g_mfa_create_kernel_v4) {
        // Use v4 API with quantized K/V support
        kernel = g_mfa_create_kernel_v4(
            static_cast<int32_t>(seq_q),
            static_cast<int32_t>(seq_kv),
            static_cast<int32_t>(head_dim),
            low_prec,
            low_prec_outputs,
            causal,
            has_mask,
            use_bf16,
            window_size,
            quantized_kv
        );
    } else if (g_mfa_create_kernel_v3) {
        // Use v3 API with sliding window support
        kernel = g_mfa_create_kernel_v3(
            static_cast<int32_t>(seq_q),
            static_cast<int32_t>(seq_kv),
            static_cast<int32_t>(head_dim),
            low_prec,
            low_prec_outputs,
            causal,
            has_mask,
            use_bf16,
            window_size
        );
    } else if (use_bf16 && g_mfa_create_kernel_v2) {
        // Use v2 API with BF16 support (no sliding window)
        kernel = g_mfa_create_kernel_v2(
            static_cast<int32_t>(seq_q),
            static_cast<int32_t>(seq_kv),
            static_cast<int32_t>(head_dim),
            low_prec,
            low_prec_outputs,
            causal,
            has_mask,
            use_bf16
        );
    } else {
        // Legacy API (no sliding window, no BF16)
        kernel = g_mfa_create_kernel(
            static_cast<int32_t>(seq_q),
            static_cast<int32_t>(seq_kv),
            static_cast<int32_t>(head_dim),
            low_prec,
            low_prec_outputs,
            causal,
            has_mask
        );
    }

    if (!kernel) {
        throw std::runtime_error("Failed to create MFA kernel");
    }

    g_kernel_cache[key] = kernel;
    return kernel;
}

// ============================================================================
// Flash Attention Forward
// ============================================================================

std::tuple<at::Tensor, at::Tensor> mps_flash_attention_forward_with_lse(
    const at::Tensor& query,   // (B, H, N, D)
    const at::Tensor& key,     // (B, H, N, D)
    const at::Tensor& value,   // (B, H, N, D)
    bool is_causal,
    const c10::optional<at::Tensor>& attn_mask,  // Optional (B, 1, N_q, N_kv) or (B, H, N_q, N_kv)
    int64_t window_size  // 0 = full attention, >0 = sliding window
) {
    // Initialize MFA on first call
    if (!g_initialized) {
        load_mfa_bridge();
        if (!g_mfa_init()) {
            throw std::runtime_error("Failed to initialize MFA");
        }
        g_initialized = true;
    }

    // Validate inputs
    TORCH_CHECK(query.dim() == 4, "Query must be 4D (B, H, N, D)");
    TORCH_CHECK(key.dim() == 4, "Key must be 4D (B, H, N, D)");
    TORCH_CHECK(value.dim() == 4, "Value must be 4D (B, H, N, D)");
    TORCH_CHECK(query.device().is_mps(), "Query must be on MPS device");
    TORCH_CHECK(key.device().is_mps(), "Key must be on MPS device");
    TORCH_CHECK(value.device().is_mps(), "Value must be on MPS device");

    const int64_t batch_size = query.size(0);
    const int64_t num_heads_q = query.size(1);
    const int64_t num_heads_kv = key.size(1);
    const int64_t seq_len_q = query.size(2);
    const int64_t head_dim = query.size(3);
    const int64_t seq_len_kv = key.size(2);

    TORCH_CHECK(key.size(0) == batch_size && value.size(0) == batch_size,
                "Batch size mismatch");
    TORCH_CHECK(key.size(3) == head_dim && value.size(3) == head_dim,
                "Head dimension mismatch");
    TORCH_CHECK(key.size(1) == value.size(1),
                "K and V must have same number of heads");

    // Handle GQA (Grouped Query Attention): expand K/V if fewer heads than Q
    const int64_t num_heads = num_heads_q;
    at::Tensor k_expanded, v_expanded;

    if (num_heads_kv != num_heads_q) {
        // GQA: num_heads_q must be divisible by num_heads_kv
        TORCH_CHECK(num_heads_q % num_heads_kv == 0,
                    "num_heads_q (", num_heads_q, ") must be divisible by num_heads_kv (", num_heads_kv, ")");
        int64_t repeat_factor = num_heads_q / num_heads_kv;

        // Expand K and V to match Q's head count: (B, H_kv, S, D) -> (B, H_q, S, D)
        // Use repeat_interleave for proper GQA expansion
        k_expanded = key.repeat_interleave(repeat_factor, /*dim=*/1);
        v_expanded = value.repeat_interleave(repeat_factor, /*dim=*/1);
    } else {
        k_expanded = key;
        v_expanded = value;
    }

    // Determine precision - MFA kernel supports FP16, BF16, and FP32
    bool is_bfloat16 = (query.scalar_type() == at::kBFloat16);
    bool is_fp16 = (query.scalar_type() == at::kHalf);

    // Use native BF16 kernel if available, otherwise fall back to FP32
    bool use_bf16_kernel = is_bfloat16 && g_mfa_create_kernel_v2;
    bool low_precision = is_fp16;  // FP16 path
    bool low_precision_outputs = is_fp16 || use_bf16_kernel;

    // Make inputs contiguous
    auto q = query.contiguous();
    auto k = k_expanded.contiguous();
    auto v = v_expanded.contiguous();

    // For BF16 without native kernel support, convert to FP32 (avoids FP16 overflow)
    if (is_bfloat16 && !use_bf16_kernel) {
        q = q.to(at::kFloat);
        k = k.to(at::kFloat);
        v = v.to(at::kFloat);
    }

    // Handle attention mask
    bool has_mask = attn_mask.has_value();
    at::Tensor mask;
    if (has_mask) {
        mask = attn_mask.value();
        TORCH_CHECK(mask.dim() == 4, "Attention mask must be 4D (B, H or 1, N_q, N_kv)");
        TORCH_CHECK(mask.device().is_mps(), "Attention mask must be on MPS device");
        // Convert to bool/uint8 if needed - kernel expects uchar (0 = attend, non-0 = mask out)
        if (mask.scalar_type() == at::kBool) {
            // Convert bool to uint8 for Metal compatibility
            mask = mask.to(at::kByte);
        }
        TORCH_CHECK(mask.scalar_type() == at::kByte,
                    "Attention mask must be bool or uint8");
        // Expand mask dimensions if needed -> (B, H, N_q, N_kv)
        // Handle (B, 1, N_q, N_kv) -> expand heads
        // Handle (B, H, 1, N_kv) -> expand query dim (1D key mask)
        // Handle (B, 1, 1, N_kv) -> expand both
        if (mask.size(1) == 1 || mask.size(2) == 1) {
            mask = mask.expand({batch_size, num_heads, seq_len_q, seq_len_kv});
        }
        mask = mask.contiguous();
    }

    // Allocate output in the appropriate precision
    at::Tensor output;
    if (use_bf16_kernel) {
        // Native BF16 kernel outputs BF16
        output = at::empty({batch_size, num_heads, seq_len_q, head_dim},
                           query.options().dtype(at::kBFloat16));
    } else if (low_precision_outputs) {
        // FP16 kernel outputs FP16
        output = at::empty({batch_size, num_heads, seq_len_q, head_dim},
                           query.options().dtype(at::kHalf));
    } else {
        // FP32 kernel outputs FP32
        output = at::empty({batch_size, num_heads, seq_len_q, head_dim},
                           query.options().dtype(at::kFloat));
    }

    // Allocate logsumexp (for backward pass, always fp32)
    auto logsumexp = at::empty({batch_size, num_heads, seq_len_q},
                                query.options().dtype(at::kFloat));

    // Get or create kernel with matching precision and causal mode
    void* kernel = get_or_create_kernel(seq_len_q, seq_len_kv, head_dim, low_precision, low_precision_outputs, is_causal, has_mask, use_bf16_kernel, static_cast<uint32_t>(window_size > 0 ? window_size : 0));

    // Get Metal buffers with byte offsets
    auto q_info = getBufferInfo(q);
    auto k_info = getBufferInfo(k);
    auto v_info = getBufferInfo(v);
    auto o_info = getBufferInfo(output);
    auto l_info = getBufferInfo(logsumexp);

    // Mask buffer info (may be nullptr if no mask)
    BufferInfo mask_info = {nil, 0};
    if (has_mask) {
        mask_info = getBufferInfo(mask);
    }

    // Use PyTorch's MPS stream command encoder for zero-sync integration
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();

        // Get PyTorch's shared command encoder - this is the key for zero-sync!
        // All our dispatches go onto the same encoder that PyTorch uses.
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        // Execute MFA using the shared encoder (no sync needed!)
        bool success = g_mfa_forward_encode(
            kernel,
            (__bridge void*)encoder,  // PyTorch's shared command encoder
            (__bridge void*)q_info.buffer,
            (__bridge void*)k_info.buffer,
            (__bridge void*)v_info.buffer,
            (__bridge void*)o_info.buffer,
            (__bridge void*)l_info.buffer,
            has_mask ? (__bridge void*)mask_info.buffer : nullptr,
            q_info.byte_offset,
            k_info.byte_offset,
            v_info.byte_offset,
            o_info.byte_offset,
            l_info.byte_offset,
            mask_info.byte_offset,
            static_cast<int32_t>(batch_size),
            static_cast<int32_t>(num_heads)
        );

        if (!success) {
            throw std::runtime_error("MFA forward pass failed");
        }

        // No commit needed - PyTorch will commit when it needs the results
        // The encoder stays open for coalescing more kernels
    }

    // Convert output back to BF16 if input was BF16 and we used FP32 fallback
    // (native BF16 kernel already outputs BF16, no conversion needed)
    if (is_bfloat16 && !use_bf16_kernel) {
        output = output.to(at::kBFloat16);
    }

    // Return both output and logsumexp (needed for backward pass)
    return std::make_tuple(output, logsumexp);
}

// Simple forward that only returns output (for inference)
at::Tensor mps_flash_attention_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    bool is_causal,
    const c10::optional<at::Tensor>& attn_mask,
    int64_t window_size
) {
    auto [output, logsumexp] = mps_flash_attention_forward_with_lse(query, key, value, is_causal, attn_mask, window_size);
    return output;
}

// ============================================================================
// Quantized Flash Attention Forward (FP8, INT8, NF4)
// ============================================================================

// Quantization type enum matching GEMMOperandPrecision
enum class QuantizationType : uint16_t {
    NONE = 0,
    FP8_E4M3 = 3,
    FP8_E5M2 = 4,
    INT8 = 5,
    NF4 = 6
};

at::Tensor mps_flash_attention_forward_quantized(
    const at::Tensor& query,       // (B, H, N, D) - FP16/BF16/FP32
    const at::Tensor& key,         // (B, H, N, D) - Quantized (uint8 for FP8/INT8, packed for NF4)
    const at::Tensor& value,       // (B, H, N, D) - Quantized (same as K)
    const at::Tensor& k_scale,     // (B, H) or (H,) - Per-head scale factors for K
    const at::Tensor& v_scale,     // (B, H) or (H,) - Per-head scale factors for V
    int64_t quant_type,            // 3=FP8_E4M3, 4=FP8_E5M2, 5=INT8, 6=NF4
    bool is_causal,
    const c10::optional<at::Tensor>& attn_mask,
    int64_t window_size
) {
    // Initialize MFA on first call
    if (!g_initialized) {
        load_mfa_bridge();
        if (!g_mfa_init()) {
            throw std::runtime_error("Failed to initialize MFA");
        }
        g_initialized = true;
    }

    // Check that v4 API is available
    TORCH_CHECK(g_mfa_create_kernel_v4, "Quantized attention requires MFA v4 API (update libMFABridge.dylib)");
    TORCH_CHECK(g_mfa_forward_encode_quantized, "Quantized attention requires mfa_forward_encode_quantized");

    // Validate quantization type
    TORCH_CHECK(quant_type >= 3 && quant_type <= 6,
                "Invalid quantization type. Use 3=FP8_E4M3, 4=FP8_E5M2, 5=INT8, 6=NF4");

    // Validate inputs
    TORCH_CHECK(query.dim() == 4, "Query must be 4D (B, H, N, D)");
    TORCH_CHECK(key.dim() == 4, "Key must be 4D (B, H, N, D)");
    TORCH_CHECK(value.dim() == 4, "Value must be 4D (B, H, N, D)");
    TORCH_CHECK(query.device().is_mps(), "Query must be on MPS device");
    TORCH_CHECK(key.device().is_mps(), "Key must be on MPS device");
    TORCH_CHECK(value.device().is_mps(), "Value must be on MPS device");
    TORCH_CHECK(k_scale.device().is_mps(), "K scale must be on MPS device");
    TORCH_CHECK(v_scale.device().is_mps(), "V scale must be on MPS device");

    // Quantized K/V should be uint8 (for FP8/INT8) or special packed format (NF4)
    TORCH_CHECK(key.scalar_type() == at::kByte,
                "Quantized Key must be uint8 tensor");
    TORCH_CHECK(value.scalar_type() == at::kByte,
                "Quantized Value must be uint8 tensor");
    TORCH_CHECK(k_scale.scalar_type() == at::kFloat,
                "K scale must be float32");
    TORCH_CHECK(v_scale.scalar_type() == at::kFloat,
                "V scale must be float32");

    const int64_t batch_size = query.size(0);
    const int64_t num_heads_q = query.size(1);
    const int64_t num_heads_kv = key.size(1);
    const int64_t seq_len_q = query.size(2);
    const int64_t head_dim = query.size(3);
    const int64_t seq_len_kv = key.size(2);

    TORCH_CHECK(key.size(0) == batch_size && value.size(0) == batch_size,
                "Batch size mismatch");
    TORCH_CHECK(key.size(3) == head_dim && value.size(3) == head_dim,
                "Head dimension mismatch");
    TORCH_CHECK(key.size(1) == value.size(1),
                "K and V must have same number of heads");

    // Handle GQA (Grouped Query Attention): expand K/V if fewer heads than Q
    const int64_t num_heads = num_heads_q;
    at::Tensor k_expanded, v_expanded, k_scale_expanded, v_scale_expanded;

    if (num_heads_kv != num_heads_q) {
        TORCH_CHECK(num_heads_q % num_heads_kv == 0,
                    "num_heads_q must be divisible by num_heads_kv for GQA");
        int64_t repeat_factor = num_heads_q / num_heads_kv;
        k_expanded = key.repeat_interleave(repeat_factor, /*dim=*/1);
        v_expanded = value.repeat_interleave(repeat_factor, /*dim=*/1);
        k_scale_expanded = k_scale.repeat_interleave(repeat_factor, /*dim=*/-1);
        v_scale_expanded = v_scale.repeat_interleave(repeat_factor, /*dim=*/-1);
    } else {
        k_expanded = key;
        v_expanded = value;
        k_scale_expanded = k_scale;
        v_scale_expanded = v_scale;
    }

    // Determine precision for Q (non-quantized input)
    bool is_bfloat16 = (query.scalar_type() == at::kBFloat16);
    bool is_fp16 = (query.scalar_type() == at::kHalf);
    bool use_bf16_kernel = is_bfloat16 && g_mfa_create_kernel_v2;
    bool low_precision = is_fp16;
    bool low_precision_outputs = is_fp16 || use_bf16_kernel;

    // Make inputs contiguous
    auto q = query.contiguous();
    auto k = k_expanded.contiguous();
    auto v = v_expanded.contiguous();
    auto ks = k_scale_expanded.contiguous();
    auto vs = v_scale_expanded.contiguous();

    // Handle attention mask
    bool has_mask = attn_mask.has_value();
    at::Tensor mask;
    if (has_mask) {
        mask = attn_mask.value();
        TORCH_CHECK(mask.dim() == 4, "Attention mask must be 4D");
        TORCH_CHECK(mask.device().is_mps(), "Attention mask must be on MPS device");
        if (mask.scalar_type() == at::kBool) {
            mask = mask.to(at::kByte);
        }
        TORCH_CHECK(mask.scalar_type() == at::kByte, "Attention mask must be bool or uint8");
        if (mask.size(1) == 1 || mask.size(2) == 1) {
            mask = mask.expand({batch_size, num_heads, seq_len_q, seq_len_kv});
        }
        mask = mask.contiguous();
    }

    // Allocate output
    at::Tensor output;
    if (use_bf16_kernel) {
        output = at::empty({batch_size, num_heads, seq_len_q, head_dim},
                           query.options().dtype(at::kBFloat16));
    } else if (low_precision_outputs) {
        output = at::empty({batch_size, num_heads, seq_len_q, head_dim},
                           query.options().dtype(at::kHalf));
    } else {
        output = at::empty({batch_size, num_heads, seq_len_q, head_dim},
                           query.options().dtype(at::kFloat));
    }

    // Allocate logsumexp (always fp32)
    auto logsumexp = at::empty({batch_size, num_heads, seq_len_q},
                                query.options().dtype(at::kFloat));

    // Get or create quantized kernel
    void* kernel = get_or_create_kernel(
        seq_len_q, seq_len_kv, head_dim,
        low_precision, low_precision_outputs, is_causal, has_mask,
        use_bf16_kernel,
        static_cast<uint32_t>(window_size > 0 ? window_size : 0),
        static_cast<uint16_t>(quant_type)
    );

    // Get Metal buffers with byte offsets
    auto q_info = getBufferInfo(q);
    auto k_info = getBufferInfo(k);
    auto v_info = getBufferInfo(v);
    auto o_info = getBufferInfo(output);
    auto l_info = getBufferInfo(logsumexp);
    auto ks_info = getBufferInfo(ks);
    auto vs_info = getBufferInfo(vs);

    BufferInfo mask_info = {nil, 0};
    if (has_mask) {
        mask_info = getBufferInfo(mask);
    }

    // Execute using quantized forward encode
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        bool success = g_mfa_forward_encode_quantized(
            kernel,
            (__bridge void*)encoder,
            (__bridge void*)q_info.buffer,
            (__bridge void*)k_info.buffer,
            (__bridge void*)v_info.buffer,
            (__bridge void*)o_info.buffer,
            (__bridge void*)l_info.buffer,
            (__bridge void*)ks_info.buffer,
            (__bridge void*)vs_info.buffer,
            has_mask ? (__bridge void*)mask_info.buffer : nullptr,
            q_info.byte_offset,
            k_info.byte_offset,
            v_info.byte_offset,
            o_info.byte_offset,
            l_info.byte_offset,
            ks_info.byte_offset,
            vs_info.byte_offset,
            mask_info.byte_offset,
            static_cast<int32_t>(batch_size),
            static_cast<int32_t>(num_heads)
        );

        if (!success) {
            throw std::runtime_error("MFA quantized forward pass failed");
        }
    }

    return output;
}

// Helper function to quantize K/V to FP8
std::tuple<at::Tensor, at::Tensor> quantize_to_fp8(
    const at::Tensor& tensor,  // (B, H, N, D) FP16/BF16/FP32
    bool use_e5m2              // true=FP8_E5M2, false=FP8_E4M3
) {
    // Convert to FP32 for quantization
    auto t = tensor.to(at::kFloat);

    // Compute per-head max for scale calculation
    // Shape: (B, H, N, D) -> (B, H)
    auto abs_max = t.abs().amax({2, 3});  // max over N and D dimensions

    // FP8 E4M3 max representable: 448.0, FP8 E5M2 max: 57344.0
    float fp8_max = use_e5m2 ? 57344.0f : 448.0f;

    // Scale to fit in FP8 range
    auto scale = abs_max / fp8_max;
    scale = scale.clamp_min(1e-12);  // Avoid division by zero

    // Quantize: divide by scale, clamp, convert to uint8
    // Scale has shape (B, H), need to broadcast to (B, H, N, D)
    auto scale_expanded = scale.unsqueeze(-1).unsqueeze(-1);  // (B, H, 1, 1)
    auto scaled = t / scale_expanded;

    // Clamp to FP8 representable range and convert to uint8
    // For simplicity, we're doing linear quantization here
    // A proper FP8 implementation would use the actual FP8 bit representation
    scaled = scaled.clamp(-fp8_max, fp8_max);

    // Convert to uint8 representation
    // This is a simplified version - proper FP8 uses different bit layout
    auto quantized = ((scaled / fp8_max) * 127.0f + 128.0f).to(at::kByte);

    return std::make_tuple(quantized, scale);
}

// Helper function to quantize K/V to INT8
std::tuple<at::Tensor, at::Tensor> quantize_to_int8(
    const at::Tensor& tensor  // (B, H, N, D) FP16/BF16/FP32
) {
    auto t = tensor.to(at::kFloat);

    // Compute per-head max for scale
    auto abs_max = t.abs().amax({2, 3});  // (B, H)
    auto scale = abs_max / 127.0f;
    scale = scale.clamp_min(1e-12);

    // Quantize
    auto scale_expanded = scale.unsqueeze(-1).unsqueeze(-1);
    auto scaled = t / scale_expanded;
    scaled = scaled.clamp(-127.0f, 127.0f);

    // Convert to uint8 (centered at 128)
    auto quantized = (scaled + 128.0f).to(at::kByte);

    return std::make_tuple(quantized, scale);
}

// ============================================================================
// Flash Attention Backward
// ============================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor> mps_flash_attention_backward(
    const at::Tensor& grad_output,  // (B, H, N, D)
    const at::Tensor& query,        // (B, H, N, D)
    const at::Tensor& key,          // (B, H, N, D)
    const at::Tensor& value,        // (B, H, N, D)
    const at::Tensor& output,       // (B, H, N, D)
    const at::Tensor& logsumexp,    // (B, H, N)
    bool is_causal,
    const c10::optional<at::Tensor>& attn_mask,  // Optional (B, 1, N_q, N_kv) or (B, H, N_q, N_kv)
    int64_t window_size,  // 0 = full attention, >0 = sliding window
    bool bf16_backward    // true = use BF16 intermediates for ~2x faster backward
) {
    // Initialize MFA on first call
    if (!g_initialized) {
        load_mfa_bridge();
        if (!g_mfa_init()) {
            throw std::runtime_error("Failed to initialize MFA");
        }
        g_initialized = true;
    }

    // Validate inputs
    TORCH_CHECK(grad_output.dim() == 4, "grad_output must be 4D (B, H, N, D)");
    TORCH_CHECK(query.dim() == 4, "Query must be 4D (B, H, N, D)");
    TORCH_CHECK(key.dim() == 4, "Key must be 4D (B, H, N, D)");
    TORCH_CHECK(value.dim() == 4, "Value must be 4D (B, H, N, D)");
    TORCH_CHECK(output.dim() == 4, "Output must be 4D (B, H, N, D)");
    TORCH_CHECK(logsumexp.dim() == 3, "Logsumexp must be 3D (B, H, N)");

    const int64_t batch_size = query.size(0);
    const int64_t num_heads = query.size(1);
    const int64_t seq_len_q = query.size(2);
    const int64_t head_dim = query.size(3);
    const int64_t seq_len_kv = key.size(2);

    // Determine precision
    bool low_precision = (query.scalar_type() == at::kHalf ||
                          query.scalar_type() == at::kBFloat16);
    bool low_precision_outputs = low_precision;

    // Handle attention mask
    bool has_mask = attn_mask.has_value();
    at::Tensor mask;
    if (has_mask) {
        mask = attn_mask.value();
        TORCH_CHECK(mask.dim() == 4, "Attention mask must be 4D (B, H or 1, N_q, N_kv)");
        TORCH_CHECK(mask.device().is_mps(), "Attention mask must be on MPS device");
        if (mask.scalar_type() == at::kBool) {
            mask = mask.to(at::kByte);
        }
        TORCH_CHECK(mask.scalar_type() == at::kByte,
                    "Attention mask must be bool or uint8");
        if (mask.size(1) == 1 && num_heads > 1) {
            mask = mask.expand({batch_size, num_heads, seq_len_q, seq_len_kv});
        }
        mask = mask.contiguous();
    }

    // Make inputs contiguous
    // When bf16_backward=true, use mixed-precision types for the backward kernel:
    // Q/K/V: FP16 (lowPrecisionInputs memory precision)
    // O: FP32 (lowPrecisionOutputs=false)
    // dO: BF16 (lowPrecisionInputs memory precision for dO)
    // Otherwise, upcast everything to FP32 for numerical stability
    at::Tensor q, k, v, o, dO;
    if (bf16_backward && low_precision) {
        // Mixed-precision backward
        q = query.contiguous().to(at::kHalf);
        k = key.contiguous().to(at::kHalf);
        v = value.contiguous().to(at::kHalf);
        o = output.contiguous().to(at::kFloat);  // O is FP32 when lowPrecisionOutputs=false
        dO = grad_output.contiguous().to(at::kBFloat16);  // dO must be BF16 when lowPrecisionInputs=true
    } else {
        // Standard backward: upcast to FP32 for numerical stability
        q = query.contiguous().to(at::kFloat);
        k = key.contiguous().to(at::kFloat);
        v = value.contiguous().to(at::kFloat);
        o = output.contiguous().to(at::kFloat);
        dO = grad_output.contiguous().to(at::kFloat);
    }

    // L (logsumexp) and D have specific precision requirements:
    // When lowPrecisionIntermediates=true: L is FP16, D is BF16
    // Otherwise: both FP32
    at::Tensor lse_tensor;
    at::Tensor D;
    if (bf16_backward && low_precision) {
        lse_tensor = logsumexp.contiguous().to(at::kHalf);  // L is FP16
        D = at::empty({batch_size, num_heads, seq_len_q},
                      query.options().dtype(at::kBFloat16));  // D is BF16
    } else {
        lse_tensor = logsumexp.contiguous();  // Already FP32
        D = at::empty({batch_size, num_heads, seq_len_q},
                      query.options().dtype(at::kFloat));
    }

    // Get or create kernel
    // When bf16_backward=true with low_precision inputs, use mixed-precision backward
    bool use_mixed_backward = bf16_backward && low_precision;
    void* kernel = get_or_create_kernel(seq_len_q, seq_len_kv, head_dim, use_mixed_backward, false, is_causal, has_mask, false, static_cast<uint32_t>(window_size > 0 ? window_size : 0), 0, bf16_backward);

    // Allocate gradients (always fp32 for numerical stability)
    auto dQ = at::zeros({batch_size, num_heads, seq_len_q, head_dim},
                         query.options().dtype(at::kFloat));
    auto dK = at::zeros({batch_size, num_heads, seq_len_kv, head_dim},
                         query.options().dtype(at::kFloat));
    auto dV = at::zeros({batch_size, num_heads, seq_len_kv, head_dim},
                         query.options().dtype(at::kFloat));

    // Get Metal buffers with byte offsets
    auto q_info = getBufferInfo(q);
    auto k_info = getBufferInfo(k);
    auto v_info = getBufferInfo(v);
    auto o_info = getBufferInfo(o);
    auto do_info = getBufferInfo(dO);
    auto l_info = getBufferInfo(lse_tensor);
    auto d_info = getBufferInfo(D);
    auto dq_info = getBufferInfo(dQ);
    auto dk_info = getBufferInfo(dK);
    auto dv_info = getBufferInfo(dV);

    // Mask buffer info (may be nullptr if no mask)
    BufferInfo mask_info = {nil, 0};
    if (has_mask) {
        mask_info = getBufferInfo(mask);
    }

    // Use PyTorch's MPS stream command encoder for zero-sync integration
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();

        // Get PyTorch's shared command encoder
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        bool success = g_mfa_backward_encode(
            kernel,
            (__bridge void*)encoder,  // PyTorch's shared command encoder
            (__bridge void*)q_info.buffer,
            (__bridge void*)k_info.buffer,
            (__bridge void*)v_info.buffer,
            (__bridge void*)o_info.buffer,
            (__bridge void*)do_info.buffer,
            (__bridge void*)l_info.buffer,
            (__bridge void*)d_info.buffer,
            (__bridge void*)dq_info.buffer,
            (__bridge void*)dk_info.buffer,
            (__bridge void*)dv_info.buffer,
            has_mask ? (__bridge void*)mask_info.buffer : nullptr,
            q_info.byte_offset,
            k_info.byte_offset,
            v_info.byte_offset,
            o_info.byte_offset,
            do_info.byte_offset,
            l_info.byte_offset,
            d_info.byte_offset,
            dq_info.byte_offset,
            dk_info.byte_offset,
            dv_info.byte_offset,
            mask_info.byte_offset,
            static_cast<int32_t>(batch_size),
            static_cast<int32_t>(num_heads)
        );

        if (!success) {
            throw std::runtime_error("MFA backward pass failed");
        }

        // No commit needed - PyTorch will commit when it needs the results
    }

    // Convert gradients back to input dtype if needed
    if (low_precision) {
        dQ = dQ.to(query.scalar_type());
        dK = dK.to(query.scalar_type());
        dV = dV.to(query.scalar_type());
    }

    return std::make_tuple(dQ, dK, dV);
}

// ============================================================================
// Python Bindings
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "MPS Flash Attention - Metal accelerated attention for Apple Silicon";

    m.def("forward", &mps_flash_attention_forward,
          "Flash Attention forward pass (returns output only)",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("is_causal") = false,
          py::arg("attn_mask") = py::none(),
          py::arg("window_size") = 0);

    m.def("forward_with_lse", &mps_flash_attention_forward_with_lse,
          "Flash Attention forward pass (returns output and logsumexp for backward)",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("is_causal") = false,
          py::arg("attn_mask") = py::none(),
          py::arg("window_size") = 0);

    m.def("backward", &mps_flash_attention_backward,
          "Flash Attention backward pass",
          py::arg("grad_output"),
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("output"),
          py::arg("logsumexp"),
          py::arg("is_causal") = false,
          py::arg("attn_mask") = py::none(),
          py::arg("window_size") = 0,
          py::arg("bf16_backward") = false);

    // Quantized attention (forward only - no gradients through quantized weights)
    m.def("forward_quantized", &mps_flash_attention_forward_quantized,
          "Quantized Flash Attention forward (FP8/INT8/NF4 K/V)",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("k_scale"),
          py::arg("v_scale"),
          py::arg("quant_type"),
          py::arg("is_causal") = false,
          py::arg("attn_mask") = py::none(),
          py::arg("window_size") = 0);

    // Quantization helper functions
    m.def("quantize_to_fp8", &quantize_to_fp8,
          "Quantize tensor to FP8 format, returns (quantized, scale)",
          py::arg("tensor"),
          py::arg("use_e5m2") = false);

    m.def("quantize_to_int8", &quantize_to_int8,
          "Quantize tensor to INT8 format, returns (quantized, scale)",
          py::arg("tensor"));

    // Quantization type constants
    m.attr("QUANT_FP8_E4M3") = static_cast<int64_t>(QuantizationType::FP8_E4M3);
    m.attr("QUANT_FP8_E5M2") = static_cast<int64_t>(QuantizationType::FP8_E5M2);
    m.attr("QUANT_INT8") = static_cast<int64_t>(QuantizationType::INT8);
    m.attr("QUANT_NF4") = static_cast<int64_t>(QuantizationType::NF4);
}
