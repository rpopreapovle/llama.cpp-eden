#include "common.cuh"

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d (branchless)
    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = (2*bit_0 - 1) * d;
    v.y = (2*bit_1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

// 4-bit EDEN: Lloyd-Max codebook + optimal scale
// qr=1: iqs is the linear index within block, increments by 2
static __device__ __forceinline__ void dequantize_eden4(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_eden4 * x = (const block_eden4 *) vx;

    const float d = x[ib].d;
    const uint8_t byte = x[ib].qs[iqs / 2];

    // Precomputed Lloyd-Max centroids for N(0,1), 16 levels
    const float codebook[16] = {
        -2.2227f, -1.7930f, -1.4570f, -1.1602f,
        -0.8828f, -0.6191f, -0.3652f, -0.1172f,
         0.1172f,  0.3652f,  0.6191f,  0.8828f,
         1.1602f,  1.4570f,  1.7930f,  2.2227f
    };

    v.x = d * codebook[byte & 0xF];
    v.y = d * codebook[byte >> 4];
}

// 3-bit EDEN: Lloyd-Max codebook + optimal scale
// qr=1: iqs is the linear index within block, increments by 2
static __device__ __forceinline__ void dequantize_eden3(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_eden3 * x = (const block_eden3 *) vx;

    const float d = x[ib].d;

    // Precomputed Lloyd-Max centroids for N(0,1), 8 levels
    const float codebook[8] = {
        -2.0829f, -1.2597f, -0.7247f, -0.2332f,
         0.2332f,  0.7247f,  1.2597f,  2.0829f
    };

    // Extract 3-bit values at positions iqs and iqs+1
    auto extract = [&](int pos) -> uint8_t {
        const int byte_idx = (pos * 3) / 8;
        const int bit_off  = (pos * 3) % 8;
        uint16_t word;
        memcpy(&word, x[ib].qs + byte_idx, sizeof(uint16_t));
        return (word >> bit_off) & 7;
    };

    v.x = d * codebook[extract(iqs)];
    v.y = d * codebook[extract(iqs + 1)];
}
