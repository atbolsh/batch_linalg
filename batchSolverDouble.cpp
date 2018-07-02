#include <ATen/ATen.h>

#include  <cublas.h>
#include  <time.h>
#include  <stdio.h>
#include  <stdlib.h>
#include  <string.h>
#include  <cuda_runtime.h>
#include  <cublas_v2.h>
#include  <cusolverDn.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cusolverSp.h>
#include <cuda_runtime_api.h>

// Given a bxnxn batch of matrices (row-major, CPU memory), and
// bxn batch of solutions, computes the xs, and stores them in bx.
// 
// cusolve backend
//
// b = size of batch
// n = size of matrix
int batch_solver_double(double* bA, double* bb, double* bx, int b, int n)
{
    //Copied declarations
    cusolverSpHandle_t cusolverH = NULL;
    csrqrInfo_t info = NULL;
    cusparseMatDescr_t descrA = NULL;

    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;

    int *d_csrRowPtrA = NULL;
    int *d_csrColIndA = NULL;
    double *d_csrValA = NULL;
    double *d_b = NULL; // batchSize * m
    double *d_x = NULL; // batchSize * m

    size_t size_qr = 0;
    size_t size_internal = 0;
    void *buffer_qr = NULL; // working space for numerical factorization



    int nnzA = n*n;
    
    //Remember -- only need template of one matrix.
    int *csrRowPtr = (int*)malloc(sizeof(int)*(n + 1));
    int *csrColPtr = (int*)malloc(sizeof(int)*(nnzA));
   
    //Replace with a kernel call
    for(int i = 0; i < n + 1; i++)
        csrRowPtr[i] = i*n;
    
    //Replace with a kernel call
    for(int i = 0; i < nnzA; i++)
        csrColPtr[i] = ( i%n );

    
    //Create all necessary objects.
    cusolver_status = cusolverSpCreate(&cusolverH);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cusparse_status = cusparseCreateMatDescr(&descrA); 
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO); // base-0

    cusolver_status = cusolverSpCreateCsrqrInfo(&info); 
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS); 


 /////////////////////////////////

// step 3: copy Aj and bj to device
    cudaStat1 = cudaMalloc ((void**)&d_csrValA   , sizeof(double) * nnzA * b);
    cudaStat2 = cudaMalloc ((void**)&d_csrColIndA, sizeof(int) * nnzA);
    cudaStat3 = cudaMalloc ((void**)&d_csrRowPtrA, sizeof(int) * (n+1));
    cudaStat4 = cudaMalloc ((void**)&d_b         , sizeof(double) * n * b);
    cudaStat5 = cudaMalloc ((void**)&d_x         , sizeof(double) * n * b);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);
    assert(cudaStat5 == cudaSuccess);

    cudaStat1 = cudaMemcpy(d_csrValA   , bA,         sizeof(double) * nnzA * b, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_csrColIndA, csrColPtr,  sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(d_csrRowPtrA, csrRowPtr,  sizeof(int) * (n+1), cudaMemcpyHostToDevice);
    cudaStat4 = cudaMemcpy(d_b,          bb, sizeof(double) * n * b, cudaMemcpyHostToDevice);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);


// step 4: symbolic analysis
    cusolver_status = cusolverSpXcsrqrAnalysisBatched(
        cusolverH, n, n, nnzA,
        descrA, d_csrRowPtrA, d_csrColIndA,
        info);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

// step 5: prepare working space
    cusolver_status = cusolverSpDcsrqrBufferInfoBatched(
         cusolverH, n, n, nnzA,
         descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
         b,
         info,
         &size_internal,
         &size_qr);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

//    printf("numerical factorization needs internal data %lld bytes\n", (long long)size_internal);      
//    printf("numerical factorization needs working space %lld bytes\n", (long long)size_qr);      

    cudaStat1 = cudaMalloc((void**)&buffer_qr, size_qr);
    assert(cudaStat1 == cudaSuccess);

///////////////


// step 6: numerical factorization
// assume device memory is big enough to compute all matrices.
    cusolver_status = cusolverSpDcsrqrsvBatched(
        cusolverH, n, n, nnzA,
        descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
        d_b, d_x,
        b,
        info,
        buffer_qr);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

// step 7: Copy
// xBatch = [x0, x1, x2, ...]
    cudaStat1 = cudaMemcpy(bx, d_x, sizeof(double)*n*b, cudaMemcpyDeviceToHost);
    assert(cudaStat1 == cudaSuccess);

// Free memory

    cudaFree(d_csrRowPtrA);
    cudaFree(d_csrColIndA);
    cudaFree(d_csrValA);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(buffer_qr);
    
    delete[] csrRowPtr;
    delete[] csrColPtr;
 
    return 0;
 
}

// Given a bxnxn batch of matrices (row-major, GPU memory), and
// bxn batch of solutions, computes the xs, and stores them in bx.
// 
// cusolve backend
//
// Assumes all the arrays are in GPU memory.
//
// b = size of batch
// n = size of matrix
int batch_solver_double_GPU(double* bA, double* bb, double* bx, int b, int n)
{
    //Copied declarations
    cusolverSpHandle_t cusolverH = NULL;
    csrqrInfo_t info = NULL;
    cusparseMatDescr_t descrA = NULL;

    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;

    int *d_csrRowPtrA = NULL;
    int *d_csrColIndA = NULL;
    size_t size_qr = 0;
    size_t size_internal = 0;
    void *buffer_qr = NULL; // working space for numerical factorization



    int nnzA = n*n;
    
    //Replace with GPU-only versions.
    //Remember -- only need template of one matrix.
    int *csrRowPtr = (int*)malloc(sizeof(int)*(n + 1));
    int *csrColPtr = (int*)malloc(sizeof(int)*(nnzA));
   
    //Replace with a kernel call
    for(int i = 0; i < n + 1; i++)
        csrRowPtr[i] = i*n;
    
    //Replace with a kernel call
    for(int i = 0; i < nnzA; i++)
        csrColPtr[i] = ( i%n );

    
    //Create all necessary objects.
    cusolver_status = cusolverSpCreate(&cusolverH);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cusparse_status = cusparseCreateMatDescr(&descrA); 
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO); // base-0

    cusolver_status = cusolverSpCreateCsrqrInfo(&info); 
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS); 


 /////////////////////////////////

// step 3: copy Aj and bj to device
    cudaStat2 = cudaMalloc ((void**)&d_csrColIndA, sizeof(int) * nnzA);
    cudaStat3 = cudaMalloc ((void**)&d_csrRowPtrA, sizeof(int) * (n+1));

    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);
    assert(cudaStat5 == cudaSuccess);

    cudaStat2 = cudaMemcpy(d_csrColIndA, csrColPtr,  sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(d_csrRowPtrA, csrRowPtr,  sizeof(int) * (n+1), cudaMemcpyHostToDevice);

    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);


// step 4: symbolic analysis
    cusolver_status = cusolverSpXcsrqrAnalysisBatched(
        cusolverH, n, n, nnzA,
        descrA, d_csrRowPtrA, d_csrColIndA,
        info);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

// step 5: prepare working space
    cusolver_status = cusolverSpDcsrqrBufferInfoBatched(
         cusolverH, n, n, nnzA,
         descrA, bA, d_csrRowPtrA, d_csrColIndA,
         b,
         info,
         &size_internal,
         &size_qr);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

//    printf("numerical factorization needs internal data %lld bytes\n", (long long)size_internal);      
//    printf("numerical factorization needs working space %lld bytes\n", (long long)size_qr);      

    cudaStat1 = cudaMalloc((void**)&buffer_qr, size_qr);
    assert(cudaStat1 == cudaSuccess);

///////////////


// step 6: numerical factorization
// assume device memory is big enough to compute all matrices.
    cusolver_status = cusolverSpDcsrqrsvBatched(
        cusolverH, n, n, nnzA,
        descrA, bA, d_csrRowPtrA, d_csrColIndA,
        bb, bx,
        b,
        info,
        buffer_qr);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

// Free memory

    cudaFree(d_csrRowPtrA);
    cudaFree(d_csrColIndA);
    cudaFree(buffer_qr);
    
    delete[] csrRowPtr;
    delete[] csrColPtr;
   
    return 0;
 
}


//Useful in debugging.
int print_em_double(double* d_bA, double* d_bb, double* d_bx, int b, int n)
{
    double* c_bA = (double*)malloc(sizeof(double)*b*n*n);
    double* c_bb = (double*)malloc(sizeof(double)*b*n);
    double* c_bx = (double*)malloc(sizeof(double)*b*n);

    cudaMemcpy(c_bx, d_bx, sizeof(double)*n*b,   cudaMemcpyDeviceToHost);
    cudaMemcpy(c_bb, d_bb, sizeof(double)*n*b,   cudaMemcpyDeviceToHost);
    cudaMemcpy(c_bA, d_bA, sizeof(double)*n*n*b, cudaMemcpyDeviceToHost);

    printf("bA[0]: %E, %E, %E, %E\n", c_bA[0], c_bA[1], c_bA[2], c_bA[3]);
    printf("bA[1]: %E, %E, %E, %E\n", c_bA[4], c_bA[5], c_bA[6], c_bA[7]);
    printf("bx: %E, %E, %E< %E\n",    c_bx[0], c_bx[1], c_bx[2], c_bx[3]);
    printf("bb: %E, %E, %E< %E\n\n",    c_bb[0], c_bb[1], c_bb[2], c_bb[3]);

    delete[] c_bA;
    delete[] c_bb;
    delete[] c_bx;
}
