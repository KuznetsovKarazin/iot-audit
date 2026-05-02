/*
 * =========================================================================
 * Logistic Regression — export m2cgen per Arduino Mega 2560
 * =========================================================================
 * Generato da: m2cgen.export_to_c(lr_model)  — coefficienti REALI
 * Dataset:     TON_IoT (train_test_network.csv)
 * Features:    95 (output DataFramePreprocessor: StandardScaler + OHE)
 * Target:      binario (0=normal, 1=attack)
 *
 * Metriche sul test set (38.095 campioni):
 *   F1 = 0.9900  |  ROC-AUC = 0.9945
 *
 * Dimensione sorgente: 3.20 KB  |  SRAM usata (Wokwi): 985 B
 * Latenza inferenza (Wokwi):    ~1.330 µs per predizione
 * Limite Arduino Mega 2560:      8 KB SRAM  →  RIENTRA
 *
 * NOTA m2cgen + LogisticRegression:
 *   score() restituisce il logit grezzo (non sigmoid).
 *   Soglia corretta: score(x) > 0.0 -> attack, altrimenti normal.
 *   score_proba() aggiunge la sigmoid per ottenere P in [0,1].
 * =========================================================================
 */

#include <math.h>

/*
 * score() — logit grezzo (output diretto m2cgen).
 * input: array double[95] — feature preprocessate nello stesso
 *        ordine di DataFramePreprocessor (StandardScaler + OHE).
 * return: double  > 0.0 -> attack  |  <= 0.0 -> normal
 */
double score(double * input) {
    return -0.9811727014860384 + input[0] * 0.5309552663748482 + input[1] * -0.48346069844703343 + input[2] * 0.17220979424114038 + input[3] * 0.06996109922824466 + input[4] * 0.1399096960752882 + input[5] * -0.0004084461010809996 + input[6] * -5.7856934035225756 + input[7] * 6.653928203009451 + input[8] * 0.08445594068000209 + input[9] * -0.20570025072086406 + input[10] * -1.2251944114695252 + input[11] * -3.156648442184649 + input[12] * -0.4834877134289006 + input[13] * -0.16209556629573318 + input[14] * -0.8747840621009135 + input[15] * -0.21485740722533175 + input[16] * -1.3050198007185387 + input[17] * 4.4038344903309605 + input[18] * -3.905428868357565 + input[19] * 0.24295626490996095 + input[20] * 1.022863554454617 + input[21] * 1.6726161032216302 + input[22] * 0.30524404279373635 + input[23] * -0.8410609714949767 + input[24] * 0.10217827961439678 + input[25] * 0.030195677377062285 + input[26] * -3.3416071296213783 + input[27] * 0.11255633748867165 + input[28] * 5.107708169070037 + input[29] * -1.0323648371321938 + input[30] * -3.575626449861103 + input[31] * 0.5146371921181678 + input[32] * -1.958446753278553 + input[33] * -1.3514423160002889 + input[34] * 1.7430882174367583 + input[35] * 0.6727951511312519 + input[36] * 2.5462783322173737 + input[37] * 0.10773336376165316 + input[38] * 0.12416884227384911 + input[39] * -3.817699427970578 + input[40] * 0.7084359264127683 + input[41] * -1.5150501051575291 + input[42] * -0.5763501345280115 + input[43] * -0.23026404421675656 + input[44] * -2.583335465555288 + input[45] * 1.7767212868104842 + input[46] * -5.736256148669929 + input[47] * 4.929641969925035 + input[48] * 0.4918527968600768 + input[49] * 0.5886138897756533 + input[50] * -1.8870808653804618 + input[51] * -1.8870808653804618 + input[52] * -0.22324228531633875 + input[53] * 1.9532218839284672 + input[54] * 0.4918527968600768 + input[55] * -1.1413657088364944 + input[56] * 0.8697610229706937 + input[57] * -1.6763752017154663 + input[58] * 1.0430332608846518 + input[59] * -1.8496474396294462 + input[60] * 0.4360953407129828 + input[61] * -0.06751040618742582 + input[62] * -0.39726484716269245 + input[63] * -0.14279303638151825 + input[64] * -0.14278408110461802 + input[65] * -0.13916545989057189 + input[66] * -0.13915322945837896 + input[67] * -0.07475196741045333 + input[68] * -0.13928649186209566 + input[69] * -1.0322075493129812 + input[70] * 0.2658912396856053 + input[71] * -0.040297869117405946 + input[72] * -0.8062529372390206 + input[73] * -0.0003612415057422447 + input[74] * -0.4077575744197152 + input[75] * -1.527626087848164 + input[76] * -0.00005178521862834535 + input[77] * -0.11725198478850023 + input[78] * -0.5339314794919244 + input[79] * -1.2150073514937545 + input[80] * 4.6019814861055695 + input[81] * -0.3093328182259488 + input[82] * -1.2976365833636698 + input[83] * 6.149972064219139 + input[84] * -0.2645937797162765 + input[85] * -3.044259974227406 + input[86] * -0.4250299050933572 + input[87] * -0.015564825754346204 + input[88] * -0.04745665875979442 + input[89] * -2.983496712823342 + input[90] * -0.0007089037556247181 + input[91] * -0.5709306748149912 + input[92] * 0.3954551919813439 + input[93] * -0.8064602041176704 + input[94] * -0.000153974627716403;
}

/* score_proba() — probabilita' sigmoid in [0, 1]. */
double score_proba(double * input) {
    return 1.0 / (1.0 + exp(-score(input)));
}

/*
 * predict() — classificazione binaria.
 * return: 1 = attack, 0 = normal
 */
int predict(double * input) {
    return score(input) > 0.0 ? 1 : 0;
}
