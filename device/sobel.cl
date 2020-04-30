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
} ch_item_t;

channel ch_item_t ch_conv2d_data __attribute__((depth(DEPTH)));
channel ch_item_t ch_conv2d_out __attribute__((depth(DEPTH)));
channel int4 ch_conv2d_ctrl[CU_NUM] __attribute__((depth(0)));

__attribute__((max_global_work_dim(0))) __kernel void read(
        __global const int* restrict in, const int dim_x, const int exit_cond) {
    //int4 conv2d_params = (int4)(exit_cond, 0, 0, 0);
    //write_channel_intel(ch_conv2d_ctrl[0], conv2d_params);
    //mem_fence(CLK_CHANNEL_MEM_FENCE);

    int g_index = 0;
    int x = 0;
    int y = 0;
    int read_offset = 0;
    ch_item_t vec;

    while (g_index != exit_cond) {
        ++g_index;

        read_offset = x + y * BLOCK_X;

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            vec.data[i] = in[read_offset + i];
        }

        write_channel_intel(ch_conv2d_data, vec);

        x = (x + VEC_SIZE) & (BLOCK_X - 1);

        if (x == 0) {
            ++y;
        }
    }
}

__attribute__((max_global_work_dim(0)))
//__attribute__((autorun))
//__attribute__((num_compute_units(1, 1, 1)))
__kernel void conv2d(int exit_cond) {
        //const int id = get_compute_id(0);
        //const int4 conv_param = read_channel_intel(ch_conv2d_ctrl[id]);
        //mem_fence(CLK_CHANNEL_MEM_FENCE);
        //const int exit_cond = conv_param.s0;
        //
        int g_index = 0;

        // Filter coefficients
        int k1[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
        int k2[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int k3[3][3] = {{0, -1, 0}, {-1, 4, -1}, {0, -1, 0}};
        int k4[3][3] = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};

        // shift register of 2 rows plus filter width(w=h) and size of input vector
        int sr_conv[SR_CO k1_acc + k2_acc + k3_acc + k4_acc
        // initialize
        #pragma unroll
        for (int i = 0; i < SR_CONV_VEC_SIZE; ++i) {
            sr_conv[i] = 0;
        }

        while (g_index != exit_cond) {
            ++g_index;
    
            // vectors for input features and conv features
            ch_item_t vec_in;
            ch_item_t vec_out;

            // shift register by VEC_SIZE
            #pragma unroll
            for (int i = 0; i < SR_CONV_BASE_SIZE; ++i) {
                sr_conv[i] = sr_conv[i + VEC_SIZE];
            }

            vec_in = read_channel_intel(ch_conv2d_data);

            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k) {
                int k1_acc = 0;
                int k2_acc = 0;
                int k3_acc = 0;
                int k4_acc = 0;

                // cache new input feature in sr for future reuse   
                sr_conv[SR_CONV_BASE_SIZE + k] = vec_in.data[k];

                // perform convolutions for Gx and Gy
                #pragma unroll
                for (int i = 0; i < FILTER_W; ++i) {
                    #pragma unroll
                    for (int j = 0; j < FILTER_W; ++j) {
                        int pixel = sr_conv[i * BLOCK_X + j + k];
                        k1_acc += pixel * k1[i][j];
                        k2_acc += pixel * k2[i][j];
                        k3_acc += pixel * k3[i][j];
                        k4_acc += pixel * k4[i][j];
                    }
                }

                // store results of convolution
                vec_out.data[k] = k1_acc + k2_acc + k3_acc + k4_acc;
            }

            // populate results to write pipe
            write_channel_intel(ch_conv2d_out, vec_out);
        }
    }

__attribute__((max_global_work_dim(0))) __kernel void write(
        __global int* restrict out, const int dim_x, const int exit_cond) {
    int cond = 0;
    int x = 0;
    int y = 0;
    int write_offset = 0;
    ch_item_t vec;

    while (cond != exit_cond) {
        cond++;

        vec = read_channel_intel(ch_conv2d_out);

        write_offset = x + y * BLOCK_X;

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
