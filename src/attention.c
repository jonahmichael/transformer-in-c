#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "tensor.h"
#include "math_ops.h"
#include "attention.h"

// global flag — set to 1 to dump attention weights to file
// this is addded for Attention Visialisation to create a heatmap of the attention weights for each head and each layer in the transformer model. This can help us understand what the model is focusing on when making predictions.

int g_dump_attention = 0;
FILE* g_attention_file = NULL;

/* 
 * Flip rows and columns of a matrix
 * Example: (3,4) matrix becomes (4,3)
 * We need this because QK^T requires K to be transposed
 */
Tensor* transpose(Tensor* K) {
    Tensor* out = tensor_create(K->cols, K->rows);
    for (int i = 0; i < K->rows; i++) {
        for (int j = 0; j < K->cols; j++) {
            // what was at row i, col j → goes to row j, col i
            tensor_set(out, j, i, tensor_get(K, i, j));
        }
    }
    return out;
}


/*
 * Scaled Dot-Product Attention
 * Formula: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
 *
 * Q = what we are looking for (query)
 * K = what we are matching against (key)
 * V = what we actually return (value)
 *
 * Think of it like a search engine:
 * Q is your search query, K are page titles, V are page contents
 */
Tensor* attention(Tensor* Q, Tensor* K, Tensor* V, int causal) {
    // Step 1: QK^T — how much does each query match each key?
    // Result shape: (seq_len, seq_len)
    Tensor* Kt = transpose(K);
    Tensor* scores = mat_mul(Q, Kt);

    // Step 2: Scale by √d_k
    // Large dot products push softmax into tiny gradient regions
    // Dividing by √d_k keeps values in a stable range
    for (int i = 0; i < scores->rows; i++) {
        for (int j = 0; j < scores->cols; j++) {
            tensor_set(scores, i, j,
                tensor_get(scores, i, j) / sqrtf((float)K->cols));
        }
    }

    // welcome back guyss!! we are in phase 8 and we are here again to add masking to the attention mechanism.. so that the decoder can not see the future tokens while predicting the next token in the sequence.. so let's do it

    // Step 2.5: Apply causal mask — prevent attending to future positions
    if (causal){
        for (int i = 0; i < scores->rows; i++) {
            for (int j = 0; j < scores->cols; j++) {
                if (j > i) {
                    tensor_set(scores, i, j, -1e9f);
                }
            }
        }
    }

    // Step 3: Softmax each row
    // Converts raw scores into probabilities (each row sums to 1)
    // Higher score = more attention paid to that position
    for (int i = 0; i < scores->rows; i++) {
        softmax(scores->data + i * scores->cols, scores->cols);
    }

    // dump attention weights if flag is set
    if (g_dump_attention && g_attention_file) {
        for (int i = 0; i < scores->rows; i++) {
            for (int j = 0; j < scores->cols; j++) {
                fprintf(g_attention_file, "%.6f",
                        tensor_get(scores, i, j));
                if (j < scores->cols - 1) fprintf(g_attention_file, ",");
            }
            fprintf(g_attention_file, "\n");
        }
        fprintf(g_attention_file, "---\n");
    }

    // Step 4: Multiply by V
    // Use attention weights to get weighted sum of values
    // Result shape: (seq_len, d_k)
    Tensor* out = mat_mul(scores, V);

    // Free intermediate tensors we no longer need
    tensor_free(Kt);
    tensor_free(scores);

    return out;
}

/*
 * Extract one head's slice from a projected matrix
 * Example: if d_model=8, h=2, d_k=4
 *   head 0 gets columns 0,1,2,3
 *   head 1 gets columns 4,5,6,7
 * Each head sees a different part of the representation
 */
Tensor* get_head_slice(Tensor* X, int head, int d_k) {
    Tensor* out = tensor_create(X->rows, d_k);
    for (int i = 0; i < X->rows; i++) {
        for (int j = 0; j < d_k; j++) {
            // offset into X's columns based on which head we are
            tensor_set(out, i, j, tensor_get(X, i, head * d_k + j));
        }
    }
    return out;
}

/*
 * Concatenate all head outputs back into one matrix
 * Reverse of get_head_slice
 * Each head's (seq_len, d_k) output gets placed back
 * into its column slice → final shape (seq_len, h*d_k)
 */
Tensor* concat_heads(Tensor** heads, int h, int d_k) {
    Tensor* out = tensor_create(heads[0]->rows, h * d_k);
    for (int i = 0; i < heads[0]->rows; i++) {
        for (int hd = 0; hd < h; hd++) {
            for (int j = 0; j < d_k; j++) {
                // place head hd's values into the right column slice
                tensor_set(out, i, hd * d_k + j,
                    tensor_get(heads[hd], i, j));
            }
        }
    }
    return out;
}

/*
 * Allocate and initialize a MultiHeadAttention block
 * Creates 4 weight matrices W_Q, W_K, W_V, W_O
 * all of shape (d_model, d_model)
 * In real training these would be randomly initialized
 * and learned via backprop — here we just allocate them
 */
MultiHeadAttention* mha_create(int h, int d_model) {
    MultiHeadAttention* mha = malloc(sizeof(MultiHeadAttention));
    mha->h = h;
    mha->d_model = d_model;
    mha->W_Q = tensor_create(d_model, d_model);
    mha->W_K = tensor_create(d_model, d_model);
    mha->W_V = tensor_create(d_model, d_model);
    mha->W_O = tensor_create(d_model, d_model);
    return mha;
}

/*
 * Free all memory used by a MultiHeadAttention block
 * Must free the 4 weight tensors first, then the struct itself
 */
void mha_free(MultiHeadAttention* mha) {
    tensor_free(mha->W_Q);
    tensor_free(mha->W_K);
    tensor_free(mha->W_V);
    tensor_free(mha->W_O);
    free(mha);
}

/*
 * Multi-Head Attention Forward Pass
 *
 * The idea: instead of one big attention, run h smaller attentions
 * in parallel. Each head learns to focus on different things
 * (syntax, position, coreference, etc.)
 *
 * Steps:
 * 1. Project Q, K, V using learned weight matrices
 * 2. Split into h heads and run attention on each
 * 3. Concatenate all head outputs
 * 4. Final linear projection using W_O
 */
Tensor* mha_forward(MultiHeadAttention* mha, Tensor* Q, Tensor* K, Tensor* V, int causal){
    int d_k = mha->d_model / mha->h;  // dimension per head

    // Step 1: Linear projections
    // Transform Q, K, V into better representations for attention
    Tensor* Q_proj = mat_mul(Q, mha->W_Q);
    Tensor* K_proj = mat_mul(K, mha->W_K);
    Tensor* V_proj = mat_mul(V, mha->W_V);

    // Step 2: Run attention for each head independently
    Tensor* head_outputs[mha->h];
    for (int i = 0; i < mha->h; i++) {
        // Each head gets a different slice of the projected Q, K, V
        Tensor* Q_i = get_head_slice(Q_proj, i, d_k);
        Tensor* K_i = get_head_slice(K_proj, i, d_k);
        Tensor* V_i = get_head_slice(V_proj, i, d_k);
        head_outputs[i] = attention(Q_i, K_i, V_i, causal);  // Apply causal mask
        tensor_free(Q_i);
        tensor_free(K_i);
        tensor_free(V_i);
    }

    // Step 3: Concatenate all head outputs → (seq_len, d_model)
    Tensor* concat = concat_heads(head_outputs, mha->h, d_k);

    // Step 4: Final projection using W_O
    // Mixes information across all heads
    Tensor* out = mat_mul(concat, mha->W_O);

    // Free all intermediate tensors
    tensor_free(Q_proj);
    tensor_free(K_proj);
    tensor_free(V_proj);
    tensor_free(concat);
    for (int i = 0; i < mha->h; i++)
        tensor_free(head_outputs[i]);

    return out;
}


/*
 * fill_random - fills a tensor with small random values
 * divides by 10 to keep values small and stable
 */
void fill_random(Tensor* t) {
    for (int i = 0; i < t->rows * t->cols; i++) {
        t->data[i] = ((float)rand() / RAND_MAX) / 10.0f;
    }
}