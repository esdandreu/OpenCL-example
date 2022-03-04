// Matrix multiplication A x B = C
__kernel void matmul(__global float* output,
    int widthB,  // Output width is the same as B width
    int heightB, // Must be equal to widthA
    __global float* A,
    __global float* B) {
    int col   = get_global_id(0); // Get global position in X direction
    int row   = get_global_id(1); // Get global position in Y direction
    float sum = 0.0f;
    for (int i = 0; i < heightB; i++) {
        sum += A[row * heightB + i] * B[i * widthB + col];
    }
    output[row * widthB + col] = sum;
}