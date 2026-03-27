#include<stdio.h>
#include<stdlib.h>
#include<math.h>
//ham softmax
void softmax(float* input, float* output, int n){
    float max_val=input[0];
    for(int i=1;i<n;i++){
        if(input[i]>max_val) max_val=input[i];
    }
    float sum=0.0f;
    for(int i=0;i<n;i++){
        output[i]=expf(input[i]-max_val); //tranh overflow
        sum +=output[i];
    }
    for (int i=0;i<n;i++){
        output[i] /= sum;
    }
}
// Downsample function
void downsample(
    float *src,    //input:(T,B,C)
    int T, int B, int C,
    int ds,   //downsample factor
    float *bias,    //(ds)
    float *output   //output: (T', B, C)
){
    int dT=(T + ds - 1)/ds;     //ceil
    //pad+copy vao buffer moi
    int padded_T=dT*ds;
    float *src_pad=(float *)malloc(padded_T*B*C*sizeof(float));
    for(int t=0;t<padded_T;t++){
        int src_t=(t<T)?t:(T-1);    //lap lai gia tri cuoi neu vuot qua T
        for(int b=0;b<B;b++){
            for(int c=0;c<C;c++){
                src_pad[(t*B+b)*C+c]=src[(src_t*B+b)*C+c];               
            }
        }
    }
    // softmax weights
    float *weights =(float *)malloc(ds*sizeof(float));
    softmax(bias,weights,ds);
    // weighted sum
    for(int t=0;t<dT;t++){
        for(int b=0;b<B;B++){
            for(int c=0;c<C;c++){
                float sum=0.0f;
                for(int i=0;i<ds;i++){
                    int idx_t=t*ds+i;
                    float val=src_pad[(idx_t*B+b)*C+c];
                    sum += val*weights[i];
                }
                output[(t*B+b)*C+c]=sum;
            }
        }
    }
    free(src_pad);
    free(weights);
}