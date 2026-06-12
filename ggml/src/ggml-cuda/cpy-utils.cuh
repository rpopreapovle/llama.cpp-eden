#pragma once

#include "ggml-common.h"
#include "convert.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_eden4_block(const float * __restrict__ x, block_eden4 * __restrict__ y) {
    float sum1 = 0.0f;
    for (int j = 0; j < QK_EDEN; ++j) {
        sum1 += x[j] * x[j];
    }
    const float rms = sqrtf(sum1 / QK_EDEN);
    const float irms = rms ? 1.0f/rms : 0.0f;

    // Precomputed Lloyd-Max centroids for N(0,1), 16 levels
    const float codebook[16] = {
        -2.2227f, -1.7930f, -1.4570f, -1.1602f,
        -0.8828f, -0.6191f, -0.3652f, -0.1172f,
         0.1172f,  0.3652f,  0.6191f,  0.8828f,
         1.1602f,  1.4570f,  1.7930f,  2.2227f
    };

    uint8_t idx[QK_EDEN];
    float sum_q = 0.0f, sum2 = 0.0f;
    for (int j = 0; j < QK_EDEN; ++j) {
        const float z = x[j] * irms;
        int best = 0;
        float best_d = fabsf(z - codebook[0]);
        for (int l = 1; l < 16; ++l) {
            const float d = fabsf(z - codebook[l]);
            if (d < best_d) { best_d = d; best = l; }
        }
        idx[j] = best;
        const float q = codebook[best];
        sum_q += z * q;
        sum2  += q * q;
    }

    const float S = sum2 > 0.0f ? sum_q / sum2 : 1.0f;
    y->d = rms * S;

    for (int j = 0; j < QK_EDEN/2; ++j) {
        y->qs[j] = idx[2*j] | (idx[2*j+1] << 4);
    }
}

static __device__ void quantize_f32_eden3_block(const float * __restrict__ x, block_eden3 * __restrict__ y) {
    float sum1 = 0.0f;
    for (int j = 0; j < QK_EDEN; ++j) {
        sum1 += x[j] * x[j];
    }
    const float rms = sqrtf(sum1 / QK_EDEN);
    const float irms = rms ? 1.0f/rms : 0.0f;

    // Precomputed Lloyd-Max centroids for N(0,1), 8 levels
    const float codebook[8] = {
        -2.0829f, -1.2597f, -0.7247f, -0.2332f,
         0.2332f,  0.7247f,  1.2597f,  2.0829f
    };

    uint8_t idx[QK_EDEN];
    float sum_q = 0.0f, sum2 = 0.0f;
    for (int j = 0; j < QK_EDEN; ++j) {
        const float z = x[j] * irms;
        int best = 0;
        float best_d = fabsf(z - codebook[0]);
        for (int l = 1; l < 8; ++l) {
            const float d = fabsf(z - codebook[l]);
            if (d < best_d) { best_d = d; best = l; }
        }
        idx[j] = best;
        const float q = codebook[best];
        sum_q += z * q;
        sum2  += q * q;
    }

    const float S = sum2 > 0.0f ? sum_q / sum2 : 1.0f;
    y->d = rms * S;

    memset(y->qs, 0, sizeof(y->qs));
    for (int j = 0; j < QK_EDEN; ++j) {
        const int byte_idx = (j * 3) / 8;
        const int bit_off  = (j * 3) % 8;
        y->qs[byte_idx] |= (idx[j] & 7) << bit_off;
        if (bit_off > 5) {
            y->qs[byte_idx + 1] |= (idx[j] & 7) >> (8 - bit_off);
        }
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

static __device__ void cpy_blck_f32_eden4(const char * cxi, char * cdsti) {
    quantize_f32_eden4_block((const float *)cxi, (block_eden4 *)cdsti);
}

static __device__ void cpy_blck_f32_eden3(const char * cxi, char * cdsti) {
    quantize_f32_eden3_block((const float *)cxi, (block_eden3 *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
