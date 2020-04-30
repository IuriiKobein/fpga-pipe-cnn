#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <assert.h>
#include <stdarg.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

// Error functions
void printError(cl_int error);
void _checkError(int line, const char* file, cl_int error, const char* msg,
                 ...);  // does not return
#define EXIT_ON_ERROR(status, ...) \
    _checkError(__LINE__, __FILE__, status, __VA_ARGS__)

// Print the error associciated with an error code
void printError(cl_int error) {
    // Print error message
    switch (error) {
        case -1:
            printf("CL_DEVICE_NOT_FOUND ");
            break;
        case -2:
            printf("CL_DEVICE_NOT_AVAILABLE ");
            break;
        case -3:
            printf("CL_COMPILER_NOT_AVAILABLE ");
            break;
        case -4:
            printf("CL_MEM_OBJECT_ALLOCATION_FAILURE ");
            break;
        case -5:
            printf("CL_OUT_OF_RESOURCES ");
            break;
        case -6:
            printf("CL_OUT_OF_HOST_MEMORY ");
            break;
        case -7:
            printf("CL_PROFILING_INFO_NOT_AVAILABLE ");
            break;
        case -8:
            printf("CL_MEM_COPY_OVERLAP ");
            break;
        case -9:
            printf("CL_IMAGE_FORMAT_MISMATCH ");
            break;
        case -10:
            printf("CL_IMAGE_FORMAT_NOT_SUPPORTED ");
            break;
        case -11:
            printf("CL_BUILD_PROGRAM_FAILURE ");
            break;
        case -12:
            printf("CL_MAP_FAILURE ");
            break;
        case -13:
            printf("CL_MISALIGNED_SUB_BUFFER_OFFSET ");
            break;
        case -14:
            printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ");
            break;

        case -30:
            printf("CL_INVALID_VALUE ");
            break;
        case -31:
            printf("CL_INVALID_DEVICE_TYPE ");
            break;
        case -32:
            printf("CL_INVALID_PLATFORM ");
            break;
        case -33:
            printf("CL_INVALID_DEVICE ");
            break;
        case -34:
            printf("CL_INVALID_CONTEXT ");
            break;
        case -35:
            printf("CL_INVALID_QUEUE_PROPERTIES ");
            break;
        case -36:
            printf("CL_INVALID_COMMAND_QUEUE ");
            break;
        case -37:
            printf("CL_INVALID_HOST_PTR ");
            break;
        case -38:
            printf("CL_INVALID_MEM_OBJECT ");
            break;
        case -39:
            printf("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR ");
            break;
        case -40:
            printf("CL_INVALID_IMAGE_SIZE ");
            break;
        case -41:
            printf("CL_INVALID_SAMPLER ");
            break;
        case -42:
            printf("CL_INVALID_BINARY ");
            break;
        case -43:
            printf("CL_INVALID_BUILD_OPTIONS ");
            break;
        case -44:
            printf("CL_INVALID_PROGRAM ");
            break;
        case -45:
            printf("CL_INVALID_PROGRAM_EXECUTABLE ");
            break;
        case -46:
            printf("CL_INVALID_KERNEL_NAME ");
            break;
        case -47:
            printf("CL_INVALID_KERNEL_DEFINITION ");
            break;
        case -48:
            printf("CL_INVALID_KERNEL ");
            break;
        case -49:
            printf("CL_INVALID_ARG_INDEX ");
            break;
        case -50:
            printf("CL_INVALID_ARG_VALUE ");
            break;
        case -51:
            printf("CL_INVALID_ARG_SIZE ");
            break;
        case -52:
            printf("CL_INVALID_KERNEL_ARGS ");
            break;
        case -53:
            printf("CL_INVALID_WORK_DIMENSION ");
            break;
        case -54:
            printf("CL_INVALID_WORK_GROUP_SIZE ");
            break;
        case -55:
            printf("CL_INVALID_WORK_ITEM_SIZE ");
            break;
        case -56:
            printf("CL_INVALID_GLOBAL_OFFSET ");
            break;
        case -57:
            printf("CL_INVALID_EVENT_WAIT_LIST ");
            break;
        case -58:
            printf("CL_INVALID_EVENT ");
            break;
        case -59:
            printf("CL_INVALID_OPERATION ");
            break;
        case -60:
            printf("CL_INVALID_GL_OBJECT ");
            break;
        case -61:
            printf("CL_INVALID_BUFFER_SIZE ");
            break;
        case -62:
            printf("CL_INVALID_MIP_LEVEL ");
            break;
        case -63:
            printf("CL_INVALID_GLOBAL_WORK_SIZE ");
            break;
        default:
            printf("UNRECOGNIZED ERROR CODE (%d)", error);
    }
}

// Print line, file name, and error code if there is an error. Exits the
// application upon error.
void _checkError(int line, const char* file, cl_int error, const char* msg,
                 ...) {
    // If not successful
    if (error != CL_SUCCESS) {
        // Print line and file
        printf("ERROR: ");
        printError(error);
        printf("\nLocation: %s:%d\n", file, line);

        // Print custom message.
        va_list vl;
        va_start(vl, msg);
        vprintf(msg, vl);
        printf("\n");
        va_end(vl);

        // Cleanup and bail.
        // cleanup();
        exit(error);
    }
}

const char* kern_r1w1_read = "r1w1_read";
const char* kern_r1w1_write = "r1w1_write";
const char* kern_sobel = "sobel";
const char* kern_read = "read";
const char* kern_write = "write";
const char* kern_conv2d = "conv2d";

using Data_t = std::float_t;

template <typename TimeT = std::chrono::milliseconds>
struct measure {
    template <typename F, typename... Args>
    static typename TimeT::rep execution(F&& func, Args&&... args) {
        auto start = std::chrono::steady_clock::now();
        std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
        auto duration = std::chrono::duration_cast<TimeT>(
            std::chrono::steady_clock::now() - start);
        return duration.count();
    }
};

std::vector<std::uint8_t> program_buf_load(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::in | std::ios::ate);

    uint32_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::uint8_t> buf(size);

    file.read((std::basic_istream<char>::char_type*)buf.data(), buf.size());
    file.close();

    return buf;
}

cl::Program program_make(cl::Context& ctx, std::vector<cl::Device>& devices,
                         const std::string& path) {
    auto buf = program_buf_load(path);

    cl_int status;
    cl::Program::Binaries binaries;
    std::vector<int> bin_status(1);

    binaries.emplace_back(buf.data(), buf.size());

    auto program = cl::Program(ctx, devices, binaries, &bin_status, &status);
    if (bin_status[0] != CL_SUCCESS || status != CL_SUCCESS) {
        std::stringstream ss;
        ss << "Failed to create OpenCL program from binary file \"" << path
           << "\":";
        if (bin_status[0] != CL_SUCCESS) {
            ss << " binary status: " << bin_status[0] << ".";
        }
        if (status != CL_SUCCESS) {
            ss << " error code: " << status << ".";
        }

        return cl::Program();
    }

    status = program.build(devices, nullptr, nullptr, nullptr);
    if (status != CL_SUCCESS) {
        std::stringstream ss;
        ss << "Failed to build OpenCL program from binary file \"" << path
           << "\".";
        return cl::Program();
    }

    return program;
}

template <typename T>
cl::Buffer shared_buffer_make(cl::Context& ctx, cl::CommandQueue& queue,
                              std::size_t num, T** host_ptr) {
    cl_int status;
    cl_mem_flags flags = 0;

    flags |= CL_MEM_READ_WRITE;
    flags |= CL_MEM_ALLOC_HOST_PTR;
    auto size = sizeof(T) * num;

    auto device_ptr = cl::Buffer(ctx, flags, size, nullptr, &status);
    EXIT_ON_ERROR(status, "Failed to create buffer");
    assert(host_ptr != NULL);

    *host_ptr = (T*)queue.enqueueMapBuffer(device_ptr, CL_TRUE,
                                           CL_MAP_WRITE | CL_MAP_READ, 0, size);
    assert(*host_ptr != NULL);

    return device_ptr;
}

template <typename T>
cl_int shared_buffer_release(cl::CommandQueue& queue, cl::Buffer& sh_buf,
                             T** host_ptr) {
    cl_int status;

    status = queue.enqueueUnmapMemObject(sh_buf, *host_ptr);
    EXIT_ON_ERROR(status, "Failed to unmap shared buffer");

    return status;
}

template <typename T>
void kernel_args_set(cl::Kernel& kernel, size_t index, T&& arg) {
    auto status = kernel.setArg(index, arg);
    EXIT_ON_ERROR(status, "failed to set kernel args");
}

void kernel_args_set(size_t) {}

void kernel_args_set() {}

template <typename T, typename... Ts>
void kernel_args_set(cl::Kernel& kernel, size_t index, T&& arg, Ts&&... args) {
    kernel_args_set(kernel, index, std::forward<T>(arg));
    kernel_args_set(kernel, index + 1, std::forward<Ts>(args)...);
}

void profiling_time_print(cl::Event& event, const char* title) {
    cl_int status;
    cl_ulong s_time, e_time;

    status = event.getProfilingInfo(CL_PROFILING_COMMAND_START, &s_time);
    EXIT_ON_ERROR(status, "failed to create queue");

    status = event.getProfilingInfo(CL_PROFILING_COMMAND_END, &e_time);
    EXIT_ON_ERROR(status, "failed to create queue");

    auto time_ms = (e_time - s_time) * 1e-6f;
    printf("%s: took:(%.5f ms)\n", title, time_ms);
}

void mem_throuput_test(cl::Device& dev, cl::Context& ctx, cl::Program& prog,
                       const char* arg) {
    cl_int status;
    std::int32_t test_size = 1024 * 1024 * 4;

    cl::Kernel in_kernel(prog, kern_r1w1_read, &status);
    if (status != CL_SUCCESS) {
        printf("fail: %d to load kernel: %s", status, kern_r1w1_read);
    }

    cl::Kernel out_kernel(prog, kern_r1w1_write, &status);
    if (status != CL_SUCCESS) {
        printf("fail: %d to load kernel: %s", status, kern_r1w1_write);
    }

    cl::CommandQueue in_queue(ctx, dev, 0, &status);
    cl::CommandQueue out_queue(ctx, dev, 0, &status);
    EXIT_ON_ERROR(status, "failed to create queue");

    std::uint32_t* host_input;
    auto dev_in_buf = shared_buffer_make(ctx, in_queue, test_size, &host_input);

    std::uint32_t* host_output;
    auto dev_out_buf =
        shared_buffer_make(ctx, out_queue, test_size, &host_output);

    kernel_args_set(in_kernel, 0, dev_in_buf, std::int32_t(0),
                    std::int64_t(1024), std::int64_t(test_size),
                    std::int32_t(0));
    kernel_args_set(out_kernel, 0, dev_out_buf, std::int32_t(0),
                    std::int64_t(1024), std::int64_t(test_size),
                    std::int32_t(0));

    cl::Event event;

    in_queue.enqueueTask(in_kernel);
    out_queue.enqueueTask(out_kernel, nullptr, &event);

    event.wait();
    profiling_time_print(event, "mem test");

    shared_buffer_release(in_queue, dev_in_buf, &host_input);
    shared_buffer_release(out_queue, dev_out_buf, &host_output);

    for (auto i = 0; i < test_size; ++i) {
        if (host_input[i] != host_output[i]) {
            printf("mismatch pos:%d, in_val:%d, out_val:%d\n", i, host_input[i],
                   host_output[i]);
            break;
        }
    }

    printf("mem test ok\n");
}

void sobel_filter_test(cl::Device& dev, cl::Context& ctx, cl::Program& prog,
                       const char* img_path, std::uint32_t times) {
    cl_int status;
    std::int32_t h, w;

    auto tmp = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    cv::Mat in_img;
    tmp.convertTo(in_img, CV_32S, 0.5);

    printf("dt:%d ds:%d\n", in_img.type(), in_img.elemSize());

    size_t size_b = in_img.total() * in_img.elemSize();

    cl::Kernel kread(prog, kern_read, &status);
    EXIT_ON_ERROR(status, "fail: %d to load kernel: %s", kern_read);

    cl::CommandQueue rqueue(ctx, dev, 0, &status);
    EXIT_ON_ERROR(status, "failed to create rqueue");

    cl::Kernel kconv2d(prog, kern_conv2d, &status);
    EXIT_ON_ERROR(status, "fail: %d to load kernel: %s", kern_conv2d);

    cl::CommandQueue conv_queue(ctx, dev, 0, &status);
    EXIT_ON_ERROR(status, "failed to create wqueue");

    cl::Kernel kwrite(prog, kern_write, &status);
    EXIT_ON_ERROR(status, "fail: %d to load kernel: %s", kern_write);

    cl::CommandQueue wqueue(ctx, dev, 0, &status);
    EXIT_ON_ERROR(status, "failed to create wqueue");

    std::int32_t* host_in;
    std::int32_t* host_w;
    std::int32_t* host_out;

    auto dev_in = shared_buffer_make(ctx, rqueue, in_img.total(), &host_in);
    std::memcpy(host_in, in_img.data, size_b);

    auto dev_w = shared_buffer_make(ctx, rqueue, 9, &host_w);
    std::int32_t lapl[] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    for (auto i = 0; i < 9; ++i) {
        host_w[i] = lapl[i];
    }

    auto dev_out = shared_buffer_make(ctx, wqueue, in_img.total(), &host_out);

    kernel_args_set(kread, 0, dev_in, dev_w,
                    std::int32_t(in_img.total() / VEC_SIZE),
                    std::int32_t(times));

    kernel_args_set(kconv2d, 0, std::int32_t(in_img.total() / VEC_SIZE),
                    std::int32_t(times));

    kernel_args_set(kwrite, 0, dev_out,
                    std::int32_t(in_img.total() / VEC_SIZE),
                    std::int32_t(times));

    cl::Event revent;
    cl::Event wevent;
    cl::Event conv_event;

    auto start = std::chrono::high_resolution_clock::now();

    rqueue.enqueueTask(kread, nullptr, &revent);
    conv_queue.enqueueTask(kconv2d, nullptr, &conv_event);
    wqueue.enqueueTask(kwrite, nullptr, &wevent);

    wevent.wait();

    /* measured work */
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "total time: " << elapsed.count() * 1000 << "ms\n";

    profiling_time_print(revent, "read from global memory");
    profiling_time_print(conv_event, "covn2d computation");
    profiling_time_print(wevent, "write to global memory");

    cv::Mat png(in_img.size(), in_img.type(), host_out);
    cv::imwrite("sobel_filter_output.png", png);

    shared_buffer_release(rqueue, dev_in, &host_in);
    shared_buffer_release(wqueue, dev_out, &host_out);

    printf("sobel filter test ok\n");
}

}  // namespace

int main(int argc, char** argv) {
    cl_int err = CL_SUCCESS;
    try {
        cl_int status;
        std::vector<cl::Platform> platforms;

        cl::Platform::get(&platforms);
        if (platforms.size() == 0) {
            std::cout << "No OpenCl capab platforms\n";
            return -1;
        }

        cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};

        cl::Context context(CL_DEVICE_TYPE_ACCELERATOR, properties);

        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        for (const auto& dev : devices) {
            cl_int status = 0;
            auto caps = dev.getInfo(CL_DEVICE_SVM_CAPABILITIES, &status);
            if (status != CL_SUCCESS) {
                printf("err [%d] to get caps from dev", status);
                continue;
            }
            std::cout << caps << '\n';
        }

        auto program = program_make(context, devices, argv[1]);
        if (!strcmp(argv[2], "mem_test")) {
            mem_throuput_test(devices[0], context, program, nullptr);
        } else if (!strcmp(argv[2], "sobel_test")) {
            sobel_filter_test(devices[0], context, program, argv[3],
                              std::atoi(argv[4]));
        }

    } catch (const cl::Error& err) {
        std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }

    return 0;
}
