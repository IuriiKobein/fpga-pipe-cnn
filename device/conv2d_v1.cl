#pragma OPENCL EXTENSION cl_intel_channels : enable

#ifndef DEPTH
#define DEPTH 4
#endif

#ifndef VEC_SIZE
#define VEC_SIZE 8
#endif

#ifndef BLOCK_X
#define BLOCK_X 512
#endif

#ifndef CU_NUM
#define CU_NUM 1
#endif

#define FILTER_W 3

#define SR_CONV_BASE_SIZE (2 * BLOCK_X + FILTER_W)
#define SR_CONV_VEC_SIZE (SR_CONV_BASE_SIZE + VEC_SIZE)

typedef struct {
    int data[VEC_SIZE];
} vec_feature_t;

typedef struct {
    int data[FILTER_W * FILTER_W];
} vec_weight_t;

channel vec_feature_t ch_conv2d_feature __attribute__((depth(DEPTH)));
channel vec_weight_t ch_conv2d_weight __attribute__((depth(DEPTH)));
channel vec_feature_t ch_conv2d_out __attribute__((depth(DEPTH)));
channel int4 ch_conv2d_ctrl[CU_NUM] __attribute__((depth(0)));

__attribute__((max_global_work_dim(0))) __kernel void read(
    __global const int* restrict in, __global const int* restrict weights,
    const int exit_cond, int times) {
    // int4 conv2d_params = (int4)(exit_cond, 0, 0, 0);
    // write_channel_intel(ch_conv2d_ctrl[0], conv2d_params);
    // mem_fence(CLK_CHANNEL_MEM_FENCE);

    int count = 0;
    vec_weight_t vec_weight;

#pragma unroll
    for (int i = 0; i < (FILTER_W * FILTER_W); ++i) {
        vec_weight.data[i] = weights[i];
    }

    // send filter firstly
    write_channel_intel(ch_conv2d_weight, vec_weight);

    while (count != times) {
        ++count;
        int g_index = 0;
        int x = 0;
        int y = 0;
        while (g_index != exit_cond) {
            ++g_index;
            vec_feature_t vec_feature;

            int read_offset = x + y * BLOCK_X;

#pragma unroll
            for (int i = 0; i < VEC_SIZE; i++) {
                vec_feature.data[i] = in[read_offset + i];
            }

            write_channel_intel(ch_conv2d_feature, vec_feature);

            x = (x + VEC_SIZE) & (BLOCK_X - 1);

            if (x == 0) {
                ++y;
            }
        }
    }
}

__attribute__((max_global_work_dim(0)))
//__attribute__((autorun))
//__attribute__((num_compute_units(1, 1, 1)))
__kernel void
conv2d(int exit_cond, int times) {
    // const int id = get_compute_id(0);
    // const int4 conv_param = read_channel_intel(ch_conv2d_ctrl[id]);
    // mem_fence(CLK_CHANNEL_MEM_FENCE);
    // const int exit_cond = conv_param.s0;
    //
    int count = 0;
    // Filter coefficients
    const vec_weight_t k0 = read_channel_intel(ch_conv2d_weight);

    // shift register of 2 rows plus filter width(w=h) and size of input
    // vector
    int sr_conv[SR_CONV_VEC_SIZE];

    // initialize
#pragma unroll
    for (int i = 0; i < SR_CONV_VEC_SIZE; ++i) {
        sr_conv[i] = 0;
    }

    while (count != times) {
        ++count;
        int g_index = 0;
        while (g_index != exit_cond) {
            ++g_index;

            // vectors for input features and conv features
            vec_feature_t vec_in;
            vec_feature_t vec_out;

            // shift register by VEC_SIZE
#pragma unroll
            for (int i = 0; i < SR_CONV_BASE_SIZE; ++i) {
                sr_conv[i] = sr_conv[i + VEC_SIZE];
            }

            vec_in = read_channel_intel(ch_conv2d_feature);

#pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k) {
                int k0_acc = 0;

                // cache new input feature in sr for future reuse
                sr_conv[SR_CONV_BASE_SIZE + k] = vec_in.data[k];

                // perform convolutions for Gx and Gy
#pragma unroll
                for (int i = 0; i < FILTER_W; ++i) {
#pragma unroll
                    for (int j = 0; j < FILTER_W; ++j) {
                        int pixel = sr_conv[i * BLOCK_X + j + k];
                        int weight = k0.data[i * FILTER_W + j];
                        k0_acc += pixel * weight;
                    }
                }

                // store results of convolution
                vec_out.data[k] = k0_acc;
            }

            // populate results to write pipe
            write_channel_intel(ch_conv2d_out, vec_out);
        }
    }
}

__attribute__((max_global_work_dim(0))) __kernel void write(
    __global int* restrict out, const int exit_cond, int times) {
    int count = 0;

    while (count != times) {
        ++count;

        int g_index = 0;
        int x = 0;
        int y = 0;
        while (g_index != exit_cond) {
            ++g_index;

            vec_feature_t vec = read_channel_intel(ch_conv2d_out);

            int write_offset = x + y * BLOCK_X;

#pragma unroll
            for (int i = 0; i < VEC_SIZE; i++) {
                out[write_offset + i] = vec.data[i];
            }

            x = (x + VEC_SIZE) & (BLOCK_X - 1);

            if (x == 0) {
                ++y;
            }
        }
    }
}
