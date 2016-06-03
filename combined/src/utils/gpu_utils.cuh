#include "utils.h"


void callLinCombOfVectors(float a, float *v1, float b, float *v2,
                          int length, float *v_result);

void callAddConstantToVector(float a, float *v1, int length, float *v_result);

void callMultVectsCompwise(float *v1, float *v2, int length, float *v_result);

float callCalcVectDist(float *v1, float *v2, int length);

void callMatrixTranspose(float *in, float *out, int r, int c);

void callMatrixMultiply(float *m1, float *m2, int m1_rows,
                        int m1_cols, int m2_cols, float *result);
