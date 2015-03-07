#include <fstream>
#include <string>
#include <iostream>
#include <map>
#include <cstdlib>

#include <stdio.h> 
#include <stdlib.h> 
#include <stdarg.h> 
#include <string.h> 
#include <ctype.h> 
#include <math.h> 
#include <unistd.h> 
 
#include <time.h>
#include <assert.h>

#include <cuda.h>


// number of amino acids plus gap symbol
static int Q = 21;


// device selection (copied from previous assignment)
static void selectGpu(int *gpu_num, int *num_devs)
{
    // gpu_num: (I/O): I: Default choice,
    //                 O: best device, changed only if more than one device
    // num_devs: (O)   Number of found devices.
    int best = *gpu_num;

    cudaGetDeviceCount(num_devs);
    if ( *num_devs > 1 )
    {
        int dev_num;
        int max_cores = 0;

        for (dev_num = 0; dev_num < *num_devs; dev_num++)
        {
            cudaDeviceProp dev_properties;

            cudaGetDeviceProperties(&dev_properties, dev_num);
            if (max_cores < dev_properties.multiProcessorCount)
            {
                max_cores = dev_properties.multiProcessorCount;
                best = dev_num;
            }
        }
        *gpu_num = best;
    }
}


// device test (copied from previous assignment)
static void testDevice(int devID)
{
    // Check if we can run. Maybe do something more...
    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, devID);
    if (deviceProp.major == 9999 && deviceProp.minor == 9999)
    {   /* Simulated device. */
        printf("There is no device supporting CUDA.\n");
        cudaThreadExit();
    }
    else
        printf("Using GPU device number %d.\n", devID);
}


void getAlignmentDim(char* file_name, size_t &B, size_t &N){

    std::string line;
    std::ifstream infile(file_name);

    // get length of sequences: N
    getline(infile, line);
    N = line.length();
    
    infile.close();
    infile.clear();
    
    // get number of sequences: B
    B = 0;
    infile.open(file_name);
    if (infile){
        while (getline(infile, line)){
            B++;
        }
    }
}


void readAlignment(char* file_name, int* aln, size_t B, size_t N){

    // amino acid to number dictionary
    std::map<char,int> aa_dict;
    aa_dict['R'] = 1;
    aa_dict['H'] = 2;
    aa_dict['K'] = 3;
    aa_dict['D'] = 4;
    aa_dict['E'] = 5;
    aa_dict['S'] = 6;
    aa_dict['T'] = 7;
    aa_dict['N'] = 8;
    aa_dict['Q'] = 9;
    aa_dict['C'] = 10;
    aa_dict['G'] = 11;
    aa_dict['P'] = 12;
    aa_dict['A'] = 13;
    aa_dict['I'] = 14;
    aa_dict['L'] = 15;
    aa_dict['M'] = 16;
    aa_dict['F'] = 17;
    aa_dict['W'] = 18;
    aa_dict['Y'] = 19;
    aa_dict['V'] = 20;
    aa_dict['X'] = 21;
    aa_dict['-'] = 21;


    // fill aln matrix: BxN    
    std::string line;
    std::ifstream infile(file_name);
    if (infile){
        int b = 0;
        while (getline(infile, line)){
            for (int i = 0; i < line.length(); i++){
                aln[b*N + i] = aa_dict[line[i]];
            }
            b++;
        }
    }
}


int delta(int a, int b){
    return a == b;
}


void getFreqSingle(int* aln, float* f_single, size_t B, size_t N){

    for (int i = 0; i < N; i++){
        for (int k = 0; k < Q; k++)
            f_single[k*N + i] = 0.0;
        
        for (int b = 0; b < B; b++){
            int k = aln[b*N + i] - 1;
            f_single[k*N + i] += 1.0; 
        }

        for (int k = 0; k < Q; k++)
            f_single[k*N + i] /= (float)B;
    }
}

__global__ void getFreqSingleOnDevice(int* aln, float* f_single, size_t B, size_t N, int Q){

    // Grid dimensions: N x Q
    int i = threadIdx.x;

    if(i < N){
    for (int k = 0; k < Q; k++)
        f_single[k*N + i] = 0.0;
    
    for (int b = 0; b < B; b++){
        int k = aln[b*N + i] - 1;
        f_single[k*N + i] += 1.0; 
    }

    for (int k = 0; k < Q; k++)
        f_single[k*N + i] /= (float)B;

    }
}


void getFreqPair(int* aln, float* f_pair, size_t B, size_t N){

    for (int i = 0; i < N; i++){

    for (int j = 0; j < N; j++){
        for (int k = 0; k < Q; k++)
        for (int l = 0; l < Q; l++)
            f_pair[(Q*i + k) * Q*N + (Q*j + l)] = 0.0;

        for (int b = 0; b < B; b++){
            int k = aln[b*N + i] - 1;
            int l = aln[b*N + j] - 1;
            f_pair[(Q*i + k) * Q*N + (Q*j + l)] += 1.0;
        }

        for (int k = 0; k < Q; k++){
        for (int l = 0; l < Q; l++){
            f_pair[(Q*i + k) * Q*N + (Q*j + l)] /= (float)B;
        }
        }
    }
    }

}

__global__ void getFreqPairOnDevice(int* aln, float* f_pair, size_t B, size_t N, int Q){

    // Grid dimensions: N x N
    int i = threadIdx.x;
    int j = threadIdx.y;

    if(i<N && j<N){
    for (int k = 0; k < Q; k++)
    for (int l = 0; l < Q; l++)
        f_pair[(Q*i + k) * Q*N + (Q*j + l)] = 0.0;

    for (int b = 0; b < B; b++){
        int k = aln[b*N + i] - 1;
        int l = aln[b*N + j] - 1;

        f_pair[(Q*i + k) * Q*N + (Q*j + l)] += 1.0;
    }

    for (int k = 0; k < Q; k++){
    for (int l = 0; l < Q; l++){
        f_pair[(Q*i + k) * Q*N + (Q*j + l)] /= (float)B;
        //f_pair[(Q*j + l) * Q*N + (Q*i + k)] = f_pair[(Q*i + k) * Q*N + (Q*j + l)];
    }
    }
    }
}
    

void getCovMat(float* f_single, float* f_pair, float* cov_mat, size_t B, size_t N){

    for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
        for (int k = 0; k < Q; k++){
        for (int l = 0; l < Q; l++){
            cov_mat[(Q*i + k) * Q*N + (Q*j + l)] = f_pair[(Q*i + k) * Q*N + (Q*j + l)] - (f_single[k*N + i] * f_single[l*N + j]);
        }
        }
    }
    }
}

__global__ void getCovMatOnDevice(float* f_single, float* f_pair, float* cov_mat, size_t B, size_t N, int Q){

    // Grid dimensions: N x N
    int i = threadIdx.x;
    int j = threadIdx.y;

    for (int k = 0; k < Q; k++){
    for (int l = 0; l < Q; l++){
        cov_mat[(Q*i + k) * Q*N + (Q*j + l)] = f_pair[(Q*i + k) * Q*N + (Q*j + l)] - (f_single[k*N + i] * f_single[l*N + j]);
    }
    }
}



int main(int argc, char** argv){
 
    // Check available device.
    int devID = 0, num_devs = 1;
    selectGpu(&devID, &num_devs);
    testDevice(devID);    
   
    // number of sequences in aln
    size_t B;
    // length of each sequence
    size_t N;

    // Observe B and N from input file
    getAlignmentDim(argv[1], B, N);

    // Read aln from input file
    // each amino acid/gap is represented by an integer
    
    std::cout << "Read alignment.." << std::endl;
    
    int* aln = (int*) std::malloc(B*N * sizeof(int));
    readAlignment(argv[1], aln, B, N);

    // Host calculations:
   
    time_t start_h = clock();

    // calculate column-wise amino acid frequencies
    float* f_single = (float*) std::malloc(Q*N * sizeof(float));
    getFreqSingle(aln, f_single, B, N);

    // calculate column-wise amino acid frequencies
    // for each possible pair of amino acids and columns
    float* f_pair = (float*) std::malloc(Q*N * Q*N * sizeof(float));
    getFreqPair(aln, f_pair, B, N);

    // calculate covariance matrix from frequencies
    float* cov_mat = (float*) std::malloc(Q*N * Q*N * sizeof(float));
    getCovMat(f_single, f_pair, cov_mat, B, N);

    time_t end_h = clock();


    // Device calculations:
    
    time_t start_d = clock();
    
    // calculate column-wise amino acid frequencies
    int* aln_d;
    float* f_single_d;
    assert(cudaSuccess == cudaMalloc((void**) &aln_d, B*N * sizeof(int)));
    assert(cudaSuccess == cudaMalloc((void**) &f_single_d, Q*N * sizeof(float)));
    cudaMemcpy(aln_d, aln, B*N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(f_single_d, f_single, Q*N * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "Calculate single frequencies.." << std::endl;
    getFreqSingleOnDevice <<< N, 1 >>> (aln_d, f_single_d, B, N, Q);

    // calculate column-wise amino acid frequencies
    // for each possible pair of amino acids and columns
    float* f_pair_d;
    assert(cudaSuccess == cudaMalloc((void**) &f_pair_d,
                Q*N * Q*N * sizeof(float)));
    cudaMemcpy(f_pair_d, f_pair, Q*N * Q*N * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "Calculate pair frequencies.." << std::endl;
    getFreqPairOnDevice <<< N, N >>> (aln_d, f_pair_d, B, N, Q);

    // calculate covariance matrix from frequencies
    float* cov_mat_d;
    assert(cudaSuccess == cudaMalloc((void**) &cov_mat_d,
                Q*N * Q*N * sizeof(float)));
    cudaMemcpy(cov_mat_d, cov_mat, Q*N * Q*N * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "Calculate covariance matrix.." << std::endl;
    getCovMatOnDevice <<< N, N >>> (f_single_d, f_pair_d, cov_mat_d, B, N, Q);

    // copy covariance matrix back from device
    float* cov_mat_from_d = (float*) malloc(Q*N * Q*N * sizeof(float));
    cudaMemcpy(cov_mat_from_d, cov_mat_d, Q*N *
            Q*N * sizeof(float), cudaMemcpyDeviceToHost);
    
    time_t end_d = clock();

    float t_full = ((float)end_d - (float)start_h) / CLOCKS_PER_SEC;
    float t_host = ((float)end_h - (float)start_h) / CLOCKS_PER_SEC;
    float t_dev = ((float)end_d - (float)start_d) / CLOCKS_PER_SEC;
    printf("\nTiming:\nFull: %f\nHost: %f\nDevice: %f\n\n", t_full, t_host, t_dev);


    std::cout << B << ' ' << N << ' ' << Q << std::endl;
    //for (int i = 0; i < B; i++){
    //    for (int j = 0; j < N; j++){
    //        std::cout << aln[i*N +j] << ' ';
    //    }
    //    std::cout << std::endl;
    //}
    //for (int i = 0; i < Q; i++){
    //    for (int j = 0; j < N; j++){
    //        std::cout << f_single[i*N + j] << ' ';
    //    }
    //    std::cout << std::endl;
    //}
    float err = 0.0;
    for (int i = 0; i < N*Q; i++){
        int j;
        //std::cout << std::endl << i << ' ';
        for (j = 0; j < N*Q; j++){
            err += (cov_mat_from_d[i*N*Q + j] - cov_mat[i*N*Q + j]) / N*Q;
            //std::cout << ' ' << j <<  '/' << cov_mat_from_d[i*N*Q + j] << '/' << cov_mat[i*N*Q + j];
        }
    }
    std::cout << std::endl << err << std::endl;

    return 0;
}
