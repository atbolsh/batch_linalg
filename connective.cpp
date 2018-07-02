#include <torch/torch.h>
#include <vector>
#include <string>

// CUDA forward declarations

int demo_backend();

int batch_solver_double(double* bA, double* bb, double* bx, int b, int n);
int batch_solver_double_GPU(double* bA, double* bb, double* bx, int b, int n);

int batch_solver_single(float* bA, float* bb, float* bx, int b, int n);
int batch_solver_single_GPU(float* bA, float* bb, float* bx, int b, int n);

int print_em_double(double* bA, double* bb, double* bx, int b, int n);
int print_em_single(float* bA, float* bb, float* bx, int b, int n);

//at::Tensor binv_cuda(
//    at::Tensor input, int n);


// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
//#define CHECK_CUDA_DOUBLE(x) AT_ASSERT(x.type().ID() == at::TypeID::CUDADouble, #x " must be a CUDADouble")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x) //; CHECK_CUDA_DOUBLE(x)

int demo() 
{
//    int r = demo_backend();

// Dramatis personae
    const int n = 4 ;
    const int nnzA = 16;

    const double A[nnzA] = { 1.0, 0.0, 0.0, 0.0,
                                   0.0, 2.0, 0.0, 0.0,
                                   0.0, 0.0, 3.0, 0.0, 
                                   0.1, 0.1, 0.1, 4.0};
    const double b[n] = {1.0, 1.0, 1.0, 1.0};
    const int B = 17;


// Main objects
    double *bA = (double*)malloc(sizeof(double)*nnzA*B);
    double *bBatch       = (double*)malloc(sizeof(double)*n*B);
    double *xBatch       = (double*)malloc(sizeof(double)*n*B);

// Data generation

    for(int colidx = 0 ; colidx < nnzA ; colidx++){
        double Areg = A[colidx];
        for (int batchId = 0 ; batchId < B ; batchId++){
            double eps = ((double)((rand() % 100) + 1)) * 1.e-4;
            bA[batchId*nnzA + colidx] = Areg + eps;
        }  
    }

    for(int j = 0 ; j < n ; j++){
        double breg = b[j];
        for (int batchId = 0 ; batchId < B ; batchId++){
            double eps = ((double)((rand() % 100) + 1)) * 1.e-4;
            bBatch[batchId*n + j] = breg + eps;
        }  
    }

// And now call the function to be used later.

   int r = batch_solver_double(bA, bBatch, xBatch, B, n);
   
   printf("Results: %E, %E, %E, %E, %E \n\n", xBatch[0], xBatch[1], xBatch[2], xBatch[3], xBatch[4]);

   delete[] bA;
   delete[] bBatch;
   delete[] xBatch;

   return r;
}


// Right now, only for GPU Double tensors.
at::Tensor batchSolveDouble(
    at::Tensor bA,
    at::Tensor bb
)
{
    CHECK_INPUT(bA);
    CHECK_INPUT(bb);
//    CHECK_INPUT(bx);

    at::Tensor bx = at::zeros_like(bb);

    double* d_bA = bA.data<double>();
    double* d_bb = bb.data<double>();
    double* d_bx = bx.data<double>(); 
   
    int b = bA.size(0);
    int n = bA.size(1);

//    print_em_double(d_bA, d_bb, d_bx, b, n);    
  
    AT_ASSERT(n == bA.size(2), "BA must be a square");
    
// And now call the function

    int r = batch_solver_double_GPU(d_bA, d_bb, d_bx, b, n);

//    print_em_double(d_bA, d_bb, d_bx, b, n);    

    return bx;
}

at::Tensor batchSolveSingle(
    at::Tensor bA,
    at::Tensor bb
)
{
    CHECK_INPUT(bA);
    CHECK_INPUT(bb);
//    CHECK_INPUT(bx);

    at::Tensor bx = at::zeros_like(bb);

    float* d_bA = bA.data<float>();
    float* d_bb = bb.data<float>();
    float* d_bx = bx.data<float>(); 
   
    int b = bA.size(0);
    int n = bA.size(1);

//    print_em_single(d_bA, d_bb, d_bx, b, n);    
  
    AT_ASSERT(n == bA.size(2), "BA must be a square");
    
// And now call the function

    int r = batch_solver_single_GPU(d_bA, d_bb, d_bx, b, n);

//    print_em_single(d_bA, d_bb, d_bx, b, n);    

    return bx;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("demo", &demo, "Demo of batch solver (CUDA)");
  m.def("batchSolveDouble", &batchSolveDouble, "Solve batch of CUDADouble linear equations.");
  m.def("batchSolveSingle", &batchSolveSingle, "Solve batch of CUDASingle linear equations.");
}
