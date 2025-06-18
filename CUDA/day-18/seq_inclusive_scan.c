#include <stdio.h>

void sequential_scan(float *x, float *y, unsigned int N) {
	y[0] = x[0];
	for (unsigned int i = 1; i < N; ++i) {
		y[i] = y[i - 1] + x[i];
	}
}
