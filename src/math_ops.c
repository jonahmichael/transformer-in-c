#include <math.h>
#include <stdlib.h>
#include "math_ops.h"
#include "tensor.h"
Tensor* mat_mul(Tensor* A, Tensor* B) {
    Tensor* C = tensor_create(A->rows, B->cols);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A->cols; k++) {
                sum += tensor_get(A, i, k) * tensor_get(B, k, j);
            }
            tensor_set(C, i, j, sum);
        }}
    return C;
}
Tensor* mat_add(Tensor* A, Tensor* B) {
    Tensor* C = tensor_create(A->rows, A->cols);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            float val = tensor_get(A, i, j) + tensor_get(B, i, j);
            tensor_set(C, i, j, val);
        }
    }
    return C;
}

void softmax(float* x, int len) {
    float max = x[0];
    for (int i = 1; i < len; i++)
        if (x[i] > max) max = x[i];

    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }

    for (int i = 0; i < len; i++)
        x[i] /= sum;
}



// ---------- HELPER -------------------------

float compute_mean(float* x, int len) {  // FOR MEAN
    float sum = 0.0f;
    for (int i = 0; i < len; i++)
        sum += x[i];
    return sum / len;
}
float compute_variance(float* x, int len, float mean) {  // FOR VARIANCE
    float avg = 0.0f;  
    for (int i = 0; i < len; i++) {
        float val = x[i] - mean;
        avg += val * val;
    }
    return avg / len;
}
void layer_norm(float* x, float* gamma, float* beta, int len) {    // LAYER NORM 
// REFER DOC FOR WHY LAYER NORM IS PREFFERED OVER BATCH NORM IN TRANSFORMERS
    float mean     = compute_mean(x, len);
    float variance = compute_variance(x, len, mean);
    float epsilon  = 1e-5f;

    for (int i = 0; i < len; i++) {
        x[i] = gamma[i] * (x[i] - mean) / sqrtf(variance + epsilon) + beta[i];
    }
}
// 0.7978845608 = sqrt(2/pi)
float gelu(float x) {  //// GELU activation used in Transformer FFN
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}


/*
So out of all of the above i can explain the maths and logic behind every fucntion except GELU and softmax. this is at the momemtn.

Once i have a better understanding of those two i will update the comments in the code to explain them as well.

ciao :) 
Day 1 was very fun and prodcutive - 05-05-2026

*/