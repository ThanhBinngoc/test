#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// Activation
float swoosh_r(float x) {
    return logf(1.0f + expf(x - 1.0f)) - 0.08f * x - 0.313261687f;
}
float swooshL(float x) {
    return logf(1.0f + expf(x - 4.0f)) - 0.08f * x - 0.035f;
}
// Conv2D
void conv2d(float* input, float* output, float* weight, float* bias,
            int in_c, int out_c,
            int in_h, int in_w,
            int k_h, int k_w,
            int stride_h, int stride_w,
            int pad_h, int pad_w) {
    int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;
    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = bias[oc];
                for (int ic = 0; ic < in_c; ic++) {
                    for (int kh = 0; kh < k_h; kh++) {
                        for (int kw = 0; kw < k_w; kw++) {
                            int ih = oh * stride_h - pad_h + kh;
                            int iw = ow * stride_w - pad_w + kw;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                int in_idx = (ic * in_h + ih) * in_w + iw;
                                int w_idx = (((oc * in_c + ic) * k_h + kh) * k_w + kw);
                                sum += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                }
                int out_idx = (oc * out_h + oh) * out_w + ow;
                output[out_idx] = swoosh_r(sum);
            }
        }
    }
}
// Depthwise Conv
void depthwise_conv2d(float* input, float* output, float* weight, float* bias,
                      int C, int H, int W, int K, int stride, int pad) {
    int out_h = (H + 2 * pad - K) / stride + 1;
    int out_w = (W + 2 * pad - K) / stride + 1;
    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = bias[c];
                for (int kh = 0; kh < K; kh++) {
                    for (int kw = 0; kw < K; kw++) {
                        int ih = oh * stride - pad + kh;
                        int iw = ow * stride - pad + kw;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            int in_idx = c * H * W + ih * W + iw;
                            int w_idx = c * K * K + kh * K + kw;
                            sum += input[in_idx] * weight[w_idx];
                        }
                    }
                }
                int out_idx = c * out_h * out_w + oh * out_w + ow;
                output[out_idx] = swoosh_r(sum);
            }
        }
    }
}
// Pointwise Conv (1x1)
void pointwise_conv2d(float* input, float* output, float* weight, float* bias,
                      int C_in, int C_out, int H, int W) {
    for (int co = 0; co < C_out; co++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float sum = bias[co];
                for (int ci = 0; ci < C_in; ci++) {
                    int in_idx = ci * H * W + h * W + w;
                    int w_idx = co * C_in + ci;
                    sum += input[in_idx] * weight[w_idx];
                }
                int out_idx = co * H * W + h * W + w;
                output[out_idx] = sum;
            }
        }
    }
}
// ConvNeXt Block
void convnext_block(float* x) {
    int C = 128, H = 29, W = 19;
    float* dw_w = malloc(C * 7 * 7 * sizeof(float));
    float* dw_b = malloc(C * sizeof(float));
    float* y_dw = malloc(C * H * W * sizeof(float));
    for (int i = 0; i < C * 7 * 7; i++) dw_w[i] = 0.01f;
    for (int i = 0; i < C; i++) dw_b[i] = 0.0f;
    depthwise_conv2d(x, y_dw, dw_w, dw_b, C, H, W, 7, 1, 3);
    // Pointwise 1 (128 → 384)
    int C_mid = 384;
    float* pw1_w = malloc(C_mid * C * sizeof(float));
    float* pw1_b = malloc(C_mid * sizeof(float));
    float* y_pw1 = malloc(C_mid * H * W * sizeof(float));
    for (int i = 0; i < C_mid * C; i++) pw1_w[i] = 0.01f;
    for (int i = 0; i < C_mid; i++) pw1_b[i] = 0.0f;
    pointwise_conv2d(y_dw, y_pw1, pw1_w, pw1_b, C, C_mid, H, W);
    // Activation
    for (int i = 0; i < C_mid * H * W; i++) {
        y_pw1[i] = swooshL(y_pw1[i]);
    }
    // Pointwise 2 (384 → 128)
    float* pw2_w = malloc(C * C_mid * sizeof(float));
    float* pw2_b = malloc(C * sizeof(float));
    float* y_pw2 = malloc(C * H * W * sizeof(float));
    for (int i = 0; i < C * C_mid; i++) pw2_w[i] = 0.01f;
    for (int i = 0; i < C; i++) pw2_b[i] = 0.0f;
    pointwise_conv2d(y_pw1, y_pw2, pw2_w, pw2_b, C_mid, C, H, W);
    // Residual
    for (int i = 0; i < C * H * W; i++) {
        x[i] += y_pw2[i];
    }
    // Free memory
    free(dw_w);
    free(dw_b);
    free(y_dw);
    free(pw1_w);
    free(pw1_b);
    free(y_pw1);
    free(pw2_w);
    free(pw2_b);
    free(y_pw2);
    printf("ConvNeXt block done!\n");
}
// MAIN
int main() {
    // Input [1,65,80] → [1,1,65,80]
    int C0 = 1, H0 = 65, W0 = 80;
    float* x = malloc(C0 * H0 * W0 * sizeof(float));
    for (int i = 0; i < C0 * H0 * W0; i++) {
        x[i] = (float)rand() / RAND_MAX;
    }
    // Conv1
    int C1 = 8;
    float* y1 = malloc(C1 * 63 * 80 * sizeof(float));
    float* w1 = malloc(C1 * C0 * 3 * 3 * sizeof(float));
    float* b1 = malloc(C1 * sizeof(float));
    conv2d(x, y1, w1, b1, C0, C1, H0, W0, 3, 3, 1, 1, 0, 1);
    // Conv2
    int C2 = 32;
    float* y2 = malloc(C2 * 31 * 39 * sizeof(float));
    float* w2 = malloc(C2 * C1 * 3 * 3 * sizeof(float));
    float* b2 = malloc(C2 * sizeof(float));
    conv2d(y1, y2, w2, b2, C1, C2, 63, 80, 3, 3, 2, 2, 0, 1);
    // Conv3
    int C3 = 128;
    float* y3 = malloc(C3 * 29 * 19 * sizeof(float));
    float* w3 = malloc(C3 * C2 * 3 * 3 * sizeof(float));
    float* b3 = malloc(C3 * sizeof(float));
    conv2d(y2, y3, w3, b3, C2, C3, 31, 39, 3, 3, 2, 2, 0, 1);
    printf("Conv embed done!\n");
    // ConvNeXt block
    convnext_block(y3);
    // Free memory
    free(x);
    free(y1);
    free(w1);
    free(b1);
    free(y2);
    free(w2);
    free(b2);
    free(y3);
    free(w3);
    free(b3);
    return 0;
}