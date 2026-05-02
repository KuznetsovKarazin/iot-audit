/*
 * =========================================================================
 * Decision Tree (max_depth=5) — export m2cgen per Arduino Mega 2560
 * =========================================================================
 * Generato da: m2cgen.export_to_c(dt_model) — soglie REALI
 * Dataset:     TON_IoT (train_test_network.csv)
 * Features:    95 (output DataFramePreprocessor: StandardScaler + OHE)
 * Target:      binario (0=normal, 1=attack)
 *
 * Dimensione sorgente: 4.76 KB  |  SRAM stimata: 4.76 KB
 * Limite Arduino Mega 2560:  8 KB SRAM  →  RIENTRA ✓
 *
 * F1 = 0.9943  |  ROC-AUC = 0.9856
 * Foglie totali: 26  |  Profondità massima: 5
 *
 * NOTA TECNICA — firma m2cgen per DecisionTreeClassifier:
 *   void score(double* input, double* output)
 *   output[0] = P(normal)   output[1] = P(attack)
 *   Predizione: attack se output[1] > output[0]
 *   Questa firma è DIVERSA da logreg.c: vedi predict() sotto.
 *
 * Deploy Arduino: copiare in progetto, includere in main.ino.
 * =========================================================================
 */

#include <string.h>

/*
 * score() — valuta l'albero e scrive le probabilità di classe in output.
 *
 * input:    array double di 95 feature preprocessate
 * output:   array double di 2 elementi
 *             output[0] = probabilità classe 0 (normal)
 *             output[1] = probabilità classe 1 (attack)
 */
void score(double * input, double * output) {
    double var0[2];
    if (input[17] <= 0.5) {
        if (input[46] <= 0.5) {
            if (input[2] <= -0.0005463569250423461) {
                if (input[4] <= -0.01473047723993659) {
                    if (input[0] <= -0.311388298869133) {
                        memcpy(var0, (double[]){0.010526315789473684, 0.9894736842105263}, 2 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){0.0, 1.0}, 2 * sizeof(double));
                    }
                } else {
                    memcpy(var0, (double[]){1.0, 0.0}, 2 * sizeof(double));
                }
            } else {
                if (input[12] <= 3.7529101371765137) {
                    memcpy(var0, (double[]){1.0, 0.0}, 2 * sizeof(double));
                } else {
                    memcpy(var0, (double[]){0.0, 1.0}, 2 * sizeof(double));
                }
            }
        } else {
            if (input[44] <= 0.5) {
                if (input[2] <= -0.013936611358076334) {
                    if (input[0] <= 1.1380477547645569) {
                        memcpy(var0, (double[]){0.024290112897707834, 0.9757098871022921}, 2 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){1.0, 0.0}, 2 * sizeof(double));
                    }
                } else {
                    if (input[8] <= -0.007050476968288422) {
                        memcpy(var0, (double[]){0.973170731707317, 0.026829268292682926}, 2 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){0.05759162303664921, 0.9424083769633508}, 2 * sizeof(double));
                    }
                }
            } else {
                if (input[1] <= -0.30616264045238495) {
                    if (input[0] <= -2.1080856323242188) {
                        memcpy(var0, (double[]){1.0, 0.0}, 2 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){0.06818181818181818, 0.9318181818181818}, 2 * sizeof(double));
                    }
                } else {
                    if (input[3] <= -0.01489240862429142) {
                        memcpy(var0, (double[]){0.9876650523614388, 0.0123349476385612}, 2 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){0.16666666666666666, 0.8333333333333334}, 2 * sizeof(double));
                    }
                }
            }
        }
    } else {
        if (input[6] <= 0.9597744941711426) {
            if (input[53] <= 0.5) {
                if (input[54] <= 0.5) {
                    memcpy(var0, (double[]){1.0, 0.0}, 2 * sizeof(double));
                } else {
                    memcpy(var0, (double[]){0.0, 1.0}, 2 * sizeof(double));
                }
            } else {
                if (input[15] <= 11.247903928160667) {
                    if (input[83] <= 0.5) {
                        memcpy(var0, (double[]){0.9826086956521739, 0.017391304347826087}, 2 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){0.007467195743337692, 0.9925328042566623}, 2 * sizeof(double));
                    }
                } else {
                    if (input[80] <= 0.5) {
                        memcpy(var0, (double[]){0.9757575757575757, 0.024242424242424242}, 2 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){0.0, 1.0}, 2 * sizeof(double));
                    }
                }
            }
        } else {
            if (input[6] <= 1.1225665807724) {
                if (input[28] <= 0.5) {
                    if (input[8] <= 0.14442132413387299) {
                        memcpy(var0, (double[]){0.9998742928975487, 0.0001257071024512885}, 2 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){0.0, 1.0}, 2 * sizeof(double));
                    }
                } else {
                    memcpy(var0, (double[]){0.0, 1.0}, 2 * sizeof(double));
                }
            } else {
                if (input[4] <= 0.03080033604055643) {
                    if (input[38] <= 0.5) {
                        memcpy(var0, (double[]){0.016286644951140065, 0.9837133550488599}, 2 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){1.0, 0.0}, 2 * sizeof(double));
                    }
                } else {
                    if (input[1] <= 0.14698276668787003) {
                        memcpy(var0, (double[]){1.0, 0.0}, 2 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){0.0, 1.0}, 2 * sizeof(double));
                    }
                }
            }
        }
    }
    memcpy(output, var0, 2 * sizeof(double));
}

/*
 * predict() — classificazione binaria dall'output dell'albero.
 * return: 1 = attack (output[1] > output[0]), 0 = normal
 */
int predict(double * input) {
    double out[2];
    score(input, out);
    return out[1] > out[0] ? 1 : 0;
}
