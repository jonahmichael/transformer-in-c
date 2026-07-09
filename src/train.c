#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor.h"
#include "math_ops.h"
#include "transformer.h"
#include "utils.h"
#include "train.h"

/*
 * embedding_lookup
 * converts token ids to float vectors by looking up rows in embedding matrix
 * token id 3 → row 3 of embedding matrix
 * output shape: (seq_len, d_model)
 */
Tensor* embedding_lookup(int* tokens, Tensor* emb, int seq_len) {
    int d_model = emb->cols;
    Tensor* out = tensor_create(seq_len, d_model);
    for (int i = 0; i < seq_len; i++) {
        int token_id = tokens[i];
        for (int j = 0; j < d_model; j++) {
            tensor_set(out, i, j, tensor_get(emb, token_id, j));
        }
    }
    return out;
}

/*
 * linear_project
 * projects decoder output to vocabulary logits
 * (seq_len, d_model) x (d_model, vocab_size) = (seq_len, vocab_size)
 * each row now has one score per token in the vocabulary
 */
Tensor* linear_project(Tensor* decoder_out, Tensor* W_out) {
    return mat_mul(decoder_out, W_out);
}

/*
 * cross_entropy_loss
 * measures how wrong the model is at each position
 * formula: loss = -log(probability of correct token)
 * perfect prediction → loss = 0
 * random guessing on 10 tokens → loss = log(10) = 2.302
 */
float cross_entropy_loss(Tensor* logits, int* targets, int seq_len) {
    float total_loss = 0.0f;
    int vocab_size = logits->cols;

    for (int i = 0; i < seq_len; i++) {
        // get pointer to this position's score row
        float* row = logits->data + i * vocab_size;

        // convert raw scores to probabilities
        softmax(row, vocab_size);

        // pick out the probability assigned to the correct token
        float correct_prob = row[targets[i]];

        // floor to avoid log(0) = -infinity
        if (correct_prob < 1e-7f) correct_prob = 1e-7f;

        total_loss += -logf(correct_prob);
    }

    // average loss across all positions in the sequence
    return total_loss / seq_len;
}

/*
 * perturb_tensor
 * saves current weights, then nudges them by a small random amount
 * sigma controls how big the nudge is — smaller = more conservative
 */
static void perturb_tensor(Tensor* t, float* saved, float sigma) {
    for (int i = 0; i < t->rows * t->cols; i++) {
        saved[i] = t->data[i];
        t->data[i] += sigma * ((float)rand() / RAND_MAX - 0.5f);
    }
}

/*
 * revert_tensor
 * restores weights to their saved pre-nudge values
 * called when the nudge made the loss worse
 */
static void revert_tensor(Tensor* t, float* saved) {
    for (int i = 0; i < t->rows * t->cols; i++)
        t->data[i] = saved[i];
}

/*
 * compute_loss
 * runs one full forward pass for a single sequence and returns the loss
 * pipeline: tokens → embeddings → +PE → transformer → linear → loss
 */
static float compute_loss(Transformer* t, Tensor* emb, Tensor* W_out,
                           int* tokens, int seq_len) {
    // step 1: convert token ids to float vectors
    Tensor* src = embedding_lookup(tokens, emb, seq_len);

    // step 2: add positional encoding so model knows word order
    Tensor* pe     = positional_encoding(seq_len, t->d_model);
    Tensor* src_pe = mat_add(src, pe);
    tensor_free(src);
    tensor_free(pe);

    // step 3: run through transformer (copy task: src = tgt)
    Tensor* out = transformer_forward(t, src_pe, src_pe);
    tensor_free(src_pe);

    // step 4: project to vocabulary scores
    Tensor* logits = linear_project(out, W_out);
    tensor_free(out);

    // step 5: compute cross entropy loss
    float loss = cross_entropy_loss(logits, tokens, seq_len);
    tensor_free(logits);

    return loss;
}

/*
 * train_copy_task
 * trains the transformer to copy its input sequence exactly
 * example: input [1,3,2,4] → output [1,3,2,4]
 *
 * optimization strategy: random perturbation
 *   - nudge the output weights by a small random amount
 *   - if loss improves, keep the nudge
 *   - if loss gets worse, revert to previous weights
 *   - repeat for many epochs
 *
 * this is not backprop — it's a zeroth order optimizer
 * but it's enough to show the loss curve decreasing
 */
void train_copy_task(int epochs) {
    // model hyperparameters
    int N          = 2;     // number of encoder/decoder layers
    int d_model    = 8;     // embedding dimension
    int d_ff       = 32;    // feedforward hidden dimension
    int h          = 2;     // number of attention heads
    int seq_len    = 4;     // length of each sequence
    int vocab_size = 10;    // tokens 0-9
    float sigma    = 0.01f; // perturbation size — smaller = more careful

    // create transformer and randomly initialize all weights
    Transformer* t = transformer_create(N, d_model, d_ff, h);
    transformer_init_random(t);

    // embedding matrix: maps each token id to a d_model vector
    // shape: (vocab_size, d_model) = (10, 8)
    Tensor* emb = tensor_create(vocab_size, d_model);
    fill_random(emb);

    // output projection: maps decoder output to vocabulary scores
    // shape: (d_model, vocab_size) = (8, 10)
    Tensor* W_out = tensor_create(d_model, vocab_size);
    fill_random(W_out);

    // training data: 4 sequences the model must learn to copy
    int dataset[4][4] = {
        {1, 3, 2, 4},
        {5, 2, 7, 1},
        {3, 8, 1, 6},
        {2, 4, 9, 3}
    };
    int n_samples = 4;

    // buffer to save W_out weights before each perturbation
    float* saved_W_out = malloc(W_out->rows * W_out->cols * sizeof(float));

    printf("starting copy task training...\n");
    printf("random baseline loss = %.4f (log(vocab_size) = log(%d))\n\n",
           logf(vocab_size), vocab_size);
    printf("%-8s %s\n", "epoch", "loss");
    printf("%-8s %s\n", "-----", "----");

    // open csv file to record loss curve for plotting
    FILE* loss_file = fopen("loss_curve.csv", "w");
    fprintf(loss_file, "epoch,loss\n");

    for (int epoch = 0; epoch < epochs; epoch++) {

        // compute average loss across all training sequences
        float current_loss = 0.0f;
        for (int s = 0; s < n_samples; s++)
            current_loss += compute_loss(t, emb, W_out, dataset[s], seq_len);
        current_loss /= n_samples;

        // nudge W_out by a small random amount
        perturb_tensor(W_out, saved_W_out, sigma);

        // recompute loss after nudge
        float new_loss = 0.0f;
        for (int s = 0; s < n_samples; s++)
            new_loss += compute_loss(t, emb, W_out, dataset[s], seq_len);
        new_loss /= n_samples;

        // if nudge made things worse, undo it
        if (new_loss >= current_loss)
            revert_tensor(W_out, saved_W_out);

        // best loss this epoch
        float best = new_loss < current_loss ? new_loss : current_loss;

        // write to csv every epoch for smooth loss curve
        fprintf(loss_file, "%d,%.6f\n", epoch, best);

        // print to terminal every 100 epochs
        if (epoch % 100 == 0)
            printf("%-8d %.6f\n", epoch, best);
    }

    fclose(loss_file);
    printf("\ntraining complete!\n");
    printf("loss curve saved to loss_curve.csv\n");

    // clean up everything
    free(saved_W_out);
    tensor_free(emb);
    tensor_free(W_out);
    transformer_free(t);
}