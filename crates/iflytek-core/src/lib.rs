mod attention;
mod beam;
mod frontend;
mod numerics;
mod ops;
mod postprocess;
mod postprocess_english;
mod postprocess_models;
mod postprocess_numeric;
mod postprocess_tables;
mod postprocess_types;
mod tensor;
mod vad_endpoint;

pub use attention::{
    MemoryAttentionConfig, MemoryAttentionInput, MemoryAttentionResult, MemoryTryAttention,
};
pub use beam::{
    BeamCandidate, BeamSearchConfig, BeamSearchResult, OriginalBeamSearch, preprocess_scores,
    select_top_k,
};
pub use frontend::{
    FEATURE_SIZE, FFT_LENGTH, FRAME_LENGTH, FRAME_SHIFT, OriginalFeatureExtractor, SAMPLE_RATE,
    VAD_FEATURE_SIZE,
};
pub use ops::{
    Conv2dParams, DecoderGemmBlock, DecoderGemmParams, decoder_active_rows, decoder_cos,
    decoder_gemm_column_block, decoder_gemm_element, decoder_gemm_into, decoder_gemm_params,
    decoder_layer_norm, decoder_log_softmax, decoder_matmul, decoder_reduce_sum,
    decoder_sigmoid, decoder_sin,
    depthwise_conv_element, depthwise_conv_into, depthwise_conv_output_len, gelu_f16,
    gemm_f16_element, gemm_f16_into, gemm_f16_output_len, gemm_f16_uses_accelerate,
    gemm_f16_uses_packed_neon, layer_norm_f16, matmul_f16, optimized_kernel_backend,
    original_add, original_multiply, sigmoid_f16,
    pack_decoder_gemm_left, pack_decoder_gemm_left_into, punctuation_context, punctuation_qk,
    punctuation_quantized_linear, punctuation_softmax, set_decoder_active_rows, softmax_f16,
    standard_conv_element, standard_conv_into,
    standard_conv_output_len, standard_conv_uses_accelerate,
};
pub use postprocess::{
    EdgeEsrPostprocessor, PostprocessOptions, PostprocessResult, Postprocessor,
};
pub use tensor::Tensor;
pub use vad_endpoint::{
    OriginalVadEndpoint, VadEndpointConfig, VadEndpointState, VadEvidenceFrame, VadStatus,
};

pub const CUSTOM_OP_DOMAIN: &str = "com.azazo1.xlite";
pub const CUSTOM_OP_VERSION: i32 = 1;
