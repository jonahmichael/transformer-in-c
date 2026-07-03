#include <math.h>
#include "utils.h"

Tensor* positional_encoding(int seq_len, int d_model) {
    Tensor* out = tensor_create(seq_len, d_model);
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < d_model / 2; i++) {
            float divisor = powf(10000.0f, (2.0f * i) / d_model);
            tensor_set(out, pos, 2 * i,     sinf(pos / divisor));
            tensor_set(out, pos, 2 * i + 1, cosf(pos / divisor));
        }
    }
    return out;
}