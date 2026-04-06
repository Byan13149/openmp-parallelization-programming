#include <stdio.h>                                                                                                                                                      
#include <stdlib.h>                                                                                                                                                     
#include "mphil_dis_cholesky.h"                           
                                                                                                                                                                          
int main() {                                              
    int n = 5;
    double c[25] = {
        12,  3,  3,  2, -1,
        3, 12,  4,  4,  2,
        3,  4, 16,  5,  2,
        2,  4,  5, 12,  4,
        -1,  2,  2,  4, 17
    };
    double elapsed = cholesky(c, n);

    printf("L =\n");
    for (int i = 0; i < n; i++) {
        printf("  [");
        for (int j = 0; j <= i; j++)
            printf("%8.4f", c[i * n + j]);
        printf("]\n");
    }
    printf("Time: %.6f s\n", elapsed);                                         
}  