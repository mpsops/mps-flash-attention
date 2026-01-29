/**
 * MPS Flash Attention - PyTorch C++ Extension
 *
 * This file provides the bridge between PyTorch MPS tensors and the
 * Metal Flash Attention kernels from metal-flash-attention.
 */

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

// Forward declarations for the Swift wrapper
// These will be implemented in attention_wrapper.swift and exposed via a bridging header
extern "C" {
    void* mfa_create_attention_kernel(
        int seq_len_q,
        int seq_len_kv,
        int head_dim,
        bool is_causal,
        bool is_backward_query,
        bool is_backward_kv
    );

    void mfa_execute_attention(
        void* kernel,
        id<MTLCommandBuffer> command_buffer,
        id<MTLBuffer> q_buffer,
        id<MTLBuffer> k_buffer,
        id<MTLBuffer> v_buffer,
        id<MTLBuffer> o_buffer,
        id<MTLBuffer> l_buffer,  // logsumexp
        float scale
    );

    void mfa_execute_attention_backward(
        void* kernel_dq,
        void* kernel_dkv,
        id<MTLCommandBuffer> command_buffer,
        id<MTLBuffer> q_buffer,
        id<MTLBuffer> k_buffer,
        id<MTLBuffer> v_buffer,
        id<MTLBuffer> o_buffer,
        id<MTLBuffer> l_buffer,
        id<MTLBuffer> do_buffer,
        id<MTLBuffer> dq_buffer,
        id<MTLBuffer> dk_buffer,
        id<MTLBuffer> dv_buffer,
        float scale
    );

    void mfa_destroy_kernel(void* kernel);
}


namespace mps_flash_attention {

// Get the Metal buffer from a PyTorch MPS tensor
id<MTLBuffer> getMTLBuffer(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.device().is_mps(), "Tensor must be on MPS device");
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");

    // Get the MPS allocator's buffer
    // This is the internal PyTorch API to access the underlying Metal buffer
    return __bridge id<MTLBuffer>(tensor.storage().data());
}


std::tuple<at::Tensor, at::Tensor> flash_attention_forward(
    const at::Tensor& query,   // (B, H, N, D)
    const at::Tensor& key,     // (B, H, N, D)
    const at::Tensor& value,   // (B, H, N, D)
    float scale,
    bool is_causal
) {
    // Validate inputs
    TORCH_CHECK(query.dim() == 4, "Query must be 4D (B, H, N, D)");
    TORCH_CHECK(key.dim() == 4, "Key must be 4D (B, H, N, D)");
    TORCH_CHECK(value.dim() == 4, "Value must be 4D (B, H, N, D)");

    const int batch_size = query.size(0);
    const int num_heads = query.size(1);
    const int seq_len_q = query.size(2);
    const int head_dim = query.size(3);
    const int seq_len_kv = key.size(2);

    TORCH_CHECK(key.size(0) == batch_size && value.size(0) == batch_size,
                "Batch size mismatch");
    TORCH_CHECK(key.size(1) == num_heads && value.size(1) == num_heads,
                "Number of heads mismatch");
    TORCH_CHECK(key.size(3) == head_dim && value.size(3) == head_dim,
                "Head dimension mismatch");

    // Allocate output tensors
    auto output = at::empty_like(query);
    auto logsumexp = at::empty({batch_size, num_heads, seq_len_q},
                                query.options().dtype(at::kFloat));

    // Get Metal buffers
    id<MTLBuffer> q_buffer = getMTLBuffer(query);
    id<MTLBuffer> k_buffer = getMTLBuffer(key);
    id<MTLBuffer> v_buffer = getMTLBuffer(value);
    id<MTLBuffer> o_buffer = getMTLBuffer(output);
    id<MTLBuffer> l_buffer = getMTLBuffer(logsumexp);

    // Get the MPS stream and command buffer
    @autoreleasepool {
        MPSStream* mps_stream = at::mps::getCurrentMPSStream();
        id<MTLCommandBuffer> command_buffer = mps_stream->commandBuffer();

        // Create the attention kernel
        // Note: In production, we'd cache these kernels
        void* kernel = mfa_create_attention_kernel(
            seq_len_q, seq_len_kv, head_dim,
            is_causal, false, false
        );

        // Execute for each batch and head
        // TODO: Batch this properly in the kernel
        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < num_heads; h++) {
                // Calculate buffer offsets
                size_t qkv_offset = (b * num_heads + h) * seq_len_q * head_dim * sizeof(float);
                size_t l_offset = (b * num_heads + h) * seq_len_q * sizeof(float);

                // Execute attention
                mfa_execute_attention(
                    kernel,
                    command_buffer,
                    q_buffer,  // with offset
                    k_buffer,
                    v_buffer,
                    o_buffer,
                    l_buffer,
                    scale
                );
            }
        }

        // Cleanup
        mfa_destroy_kernel(kernel);

        // Synchronize
        mps_stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
    }

    return std::make_tuple(output, logsumexp);
}


std::tuple<at::Tensor, at::Tensor, at::Tensor> flash_attention_backward(
    const at::Tensor& grad_output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& output,
    const at::Tensor& logsumexp,
    float scale,
    bool is_causal
) {
    // Validate inputs
    TORCH_CHECK(grad_output.dim() == 4, "grad_output must be 4D");

    const int batch_size = query.size(0);
    const int num_heads = query.size(1);
    const int seq_len_q = query.size(2);
    const int head_dim = query.size(3);
    const int seq_len_kv = key.size(2);

    // Allocate gradient tensors
    auto grad_q = at::empty_like(query);
    auto grad_k = at::empty_like(key);
    auto grad_v = at::empty_like(value);

    // Get Metal buffers
    id<MTLBuffer> q_buffer = getMTLBuffer(query);
    id<MTLBuffer> k_buffer = getMTLBuffer(key);
    id<MTLBuffer> v_buffer = getMTLBuffer(value);
    id<MTLBuffer> o_buffer = getMTLBuffer(output);
    id<MTLBuffer> l_buffer = getMTLBuffer(logsumexp);
    id<MTLBuffer> do_buffer = getMTLBuffer(grad_output);
    id<MTLBuffer> dq_buffer = getMTLBuffer(grad_q);
    id<MTLBuffer> dk_buffer = getMTLBuffer(grad_k);
    id<MTLBuffer> dv_buffer = getMTLBuffer(grad_v);

    @autoreleasepool {
        MPSStream* mps_stream = at::mps::getCurrentMPSStream();
        id<MTLCommandBuffer> command_buffer = mps_stream->commandBuffer();

        // Create backward kernels (dQ and dK/dV are separate in MFA)
        void* kernel_dq = mfa_create_attention_kernel(
            seq_len_q, seq_len_kv, head_dim,
            is_causal, true, false  // backward_query
        );
        void* kernel_dkv = mfa_create_attention_kernel(
            seq_len_q, seq_len_kv, head_dim,
            is_causal, false, true  // backward_kv
        );

        // Execute backward passes
        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < num_heads; h++) {
                mfa_execute_attention_backward(
                    kernel_dq, kernel_dkv,
                    command_buffer,
                    q_buffer, k_buffer, v_buffer,
                    o_buffer, l_buffer, do_buffer,
                    dq_buffer, dk_buffer, dv_buffer,
                    scale
                );
            }
        }

        mfa_destroy_kernel(kernel_dq);
        mfa_destroy_kernel(kernel_dkv);

        mps_stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
    }

    return std::make_tuple(grad_q, grad_k, grad_v);
}

}  // namespace mps_flash_attention


// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "MPS Flash Attention - Metal backend for scaled dot-product attention";

    m.def("flash_attention_forward",
          &mps_flash_attention::flash_attention_forward,
          "Flash Attention forward pass (MPS)",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("scale"),
          py::arg("is_causal"));

    m.def("flash_attention_backward",
          &mps_flash_attention::flash_attention_backward,
          "Flash Attention backward pass (MPS)",
          py::arg("grad_output"),
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("output"),
          py::arg("logsumexp"),
          py::arg("scale"),
          py::arg("is_causal"));
}
