/*Copyright (c) 2018 Miguel Monteiro, Andrew Adams, Jongmin Baek, Abe Davis

Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

//This class is responsible for managing GPU memory for GOOGLE_CUDA (tensorflow) or simply CUDA in C++

#ifndef PERMUTOHEDRAL_LATTICE_DEVICEMEMORYALLOCATOR_H
#define PERMUTOHEDRAL_LATTICE_DEVICEMEMORYALLOCATOR_H


#include <cuda_runtime.h>
// function for debugging cuda calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if(code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

class DeviceMemoryAllocator {

    //allocator has a capacity to store 10 pointers
    void* ptr_addresses[10];
    int filled;

public:

    DeviceMemoryAllocator(): filled(0){
        for(int i=0; i < 10; i++) {
            ptr_addresses[i] = 0;
        }
    }

    ~DeviceMemoryAllocator(){
        for(int i=0; i < filled; i++)
            gpuErrchk(cudaFree(ptr_addresses[i]));
    }

    template<typename t>
    t* allocate_device_memory(int num_elements){
        gpuErrchk(cudaMalloc(&ptr_addresses[filled], num_elements*sizeof(t)));
        filled++;
        return (t*) ptr_addresses[filled-1];
    }

    template<typename t> void memset(void * ptr, t value, int num_elements){
        gpuErrchk(cudaMemset(ptr, value, num_elements * sizeof(t)));
    }

    template<typename t> void memcpy(void * device_ptr, void * host_ptr, int num_elements){
        gpuErrchk(cudaMemcpy(device_ptr, host_ptr, num_elements * sizeof(t), cudaMemcpyHostToDevice));
    }

};

#endif //PERMUTOHEDRAL_LATTICE_DEVICEMEMORYALLOCATOR_H

