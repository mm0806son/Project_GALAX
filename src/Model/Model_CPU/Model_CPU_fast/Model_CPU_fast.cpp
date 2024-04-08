/*
 * @Date: 2024-04-07 14:29:10
 * @Author: Zijie Ning zijie.ning@kuleuven.be
 * @LastEditors: Zijie Ning zijie.ning@kuleuven.be
 * @LastEditTime: 2024-04-08 12:43:18
 * @FilePath: /Project_GALAX/src/Model/Model_CPU/Model_CPU_fast/Model_CPU_fast.cpp
 */
#define GALAX_MODEL_CPU_FAST 1
#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

// #include <typeinfo>
#include <iostream>
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>


Model_CPU_fast ::Model_CPU_fast(const Initstate &initstate, Particles &particles)
    : Model_CPU(initstate, particles)
{
}

// ! choose version
#define VERSION 1

#if VERSION == 1
void forward(int n_particles, const Initstate &initstate, Particles &particles, std::vector<float> &velocitiesx, std::vector<float> &velocitiesy, std::vector<float> &velocitiesz, std::vector<float> &accelerationsx, std::vector<float> &accelerationsy, std::vector<float> &accelerationsz)
{

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem buffer_x, buffer_y, buffer_z, buffer_masses, buffer_accx, buffer_accy, buffer_accz;
    cl_int err;

    // Define the buffers
    // float *buffer_x = (float*)malloc(sizeof(float) * n_particles);
    // float *buffer_y = (float*)malloc(sizeof(float) * n_particles);
    // float *buffer_z = (float*)malloc(sizeof(float) * n_particles);
    // float *buffer_masses = (float*)malloc(sizeof(float) * n_particles);
    // float *buffer_accx = (float*)malloc(sizeof(float) * n_particles);
    // float *buffer_accy = (float*)malloc(sizeof(float) * n_particles);
    // float *buffer_accz = (float*)malloc(sizeof(float) * n_particles);

    // Step 1: Get the first available platform
    err = clGetPlatformIDs(1, &platform, NULL);
    // Step 2: Get the first available device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    // Step 3: Create the context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    // Step 4: Create the command queue
    const cl_command_queue_properties properties[] = {CL_QUEUE_PROPERTIES, 0, 0};
    queue = clCreateCommandQueueWithProperties(context, device, properties, &err);

    // Step 5: Load the kernel source code
    /*/ OPTION 1: From string
    char* kernelSource = "__kernel void vector_add("
                                "__global const int* A,"
                                "__global const int* B,"
                                "__global int* C) {"

                                "int id = get_global_id(0);"
                                "C[id] = A[id] + B[id];"
                                "}";

    //*/

    //*/ OPTION 2: From a file
    FILE* file = fopen("kernel.cl", "r");
    if (!file) {
        fprintf(stderr, "Failed to load the kernel.\n");
    }
    fseek(file, 0, SEEK_END);
    size_t source_size = ftell(file);
    rewind(file);

    char* kernelSource = (char*)malloc(source_size + 1);
    kernelSource[source_size] = '\0';
    err = fread(kernelSource, sizeof(char), source_size, file);
    fclose(file);
    //*/

    // Step 6: Create the program
    program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);
    // Step 7: Build the program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    if (err != CL_SUCCESS) {
        // If there's a build error, retrieve and print log
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char* buildLog = (char*)malloc(logSize);
        if (buildLog) {
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog, NULL);
            buildLog[logSize-1] = '\0';
            printf("Error in kernel build:\n%s\n", buildLog);
            free(buildLog);
        }
    }
    
    // Step 8: Create the kernel
    kernel = clCreateKernel(program, "compute_accelerations", &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create kernel. Error: %d\n", err);
    }

    // Step 9: Create the buffers
    buffer_x = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * n_particles, NULL, &err);
    buffer_y = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * n_particles, NULL, &err);
    buffer_z = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * n_particles, NULL, &err);
    buffer_masses = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * n_particles, NULL, &err);
    buffer_accx = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n_particles, NULL, &err);
    buffer_accy = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n_particles, NULL, &err);
    buffer_accz = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n_particles, NULL, &err);

    // Step 10: Set the kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_x);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_y);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_z);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &buffer_masses);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &buffer_accx);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &buffer_accy);
    err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &buffer_accz);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &n_particles); 

    // Copy data from host to device
    err = clEnqueueWriteBuffer(queue, buffer_x, CL_TRUE, 0, sizeof(float) * n_particles, particles.x.data(), 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, buffer_y, CL_TRUE, 0, sizeof(float) * n_particles, particles.y.data(), 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, buffer_z, CL_TRUE, 0, sizeof(float) * n_particles, particles.z.data(), 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, buffer_masses, CL_TRUE, 0, sizeof(float) * n_particles, initstate.masses.data(), 0, NULL, NULL);

    // Step 11: Execute the kernel
    size_t global_work_size[] = {size_t(n_particles)};
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
     if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to enqueue the kernel. Error: %d\n", err);
    }
    
    // Wait for the commands in the queue to finish:
    clFinish(queue);

    // Step 12: Read the results
    err = clEnqueueReadBuffer(queue, buffer_accx, CL_TRUE, 0, sizeof(float) * n_particles, accelerationsx.data(), 0, NULL, NULL);
    err |= clEnqueueReadBuffer(queue, buffer_accy, CL_TRUE, 0, sizeof(float) * n_particles, accelerationsy.data(), 0, NULL, NULL);
    err |= clEnqueueReadBuffer(queue, buffer_accz, CL_TRUE, 0, sizeof(float) * n_particles, accelerationsz.data(), 0, NULL, NULL);


    // Check the error on reading buffers
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to read the buffer. Error: %d\n", err);
    }
    
    // Cleanup
    clReleaseMemObject(buffer_x);
    clReleaseMemObject(buffer_y);
    clReleaseMemObject(buffer_z);
    clReleaseMemObject(buffer_masses);
    clReleaseMemObject(buffer_accx);
    clReleaseMemObject(buffer_accy);
    clReleaseMemObject(buffer_accz);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    if (kernelSource != NULL) {
    free(kernelSource);
    }

    for (int i = 0; i < n_particles; i++)
    {
        velocitiesx[i] += accelerationsx[i] * 2.0f;
        velocitiesy[i] += accelerationsy[i] * 2.0f;
        velocitiesz[i] += accelerationsz[i] * 2.0f;
        particles.x[i] += velocitiesx[i] * 0.1f;
        particles.y[i] += velocitiesy[i] * 0.1f;
        particles.z[i] += velocitiesz[i] * 0.1f;
    }
}

#else
    std::cerr << "Not implemented!" << std::endl;
#endif

void Model_CPU_fast ::step()
{
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

    forward(n_particles, initstate, particles, velocitiesx, velocitiesy, velocitiesz, accelerationsx, accelerationsy, accelerationsz);
}
#endif // GALAX_MODEL_CPU_FAST
