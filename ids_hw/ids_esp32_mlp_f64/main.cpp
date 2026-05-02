/*
 * IDS Embedded — MLP float64 (pure C, no TFLite)
 * Board  : ESP32-C3 SuperMini  (RISC-V 160 MHz, 400 KB SRAM)
 * Weights: mlp_weights.h  (extracted from binary_mlp_(dl_baseline).joblib)
 * Architecture: 95 → Dense(64,ReLU) → Dense(32,ReLU) → Dense(1,Sigmoid)
 * F1=0.9816  ROC-AUC=0.9843  (float64 baseline, TON_IoT binary)
 *
 * NOTE: This is Variant B (fallback).
 * Variant A (TFLite INT8) was validated on Wokwi: F1=0.9959, ~5.25 us, arena 1668 B.
 * This variant measures physical hardware latency for the float64 baseline.
 * The quantization speedup (INT8 vs float64) can be inferred by comparing
 * Wokwi INT8 (~5 us) with physical float64 results here.
 */

#include <Arduino.h>
#include <math.h>
#include "mlp_weights.h"

#define N_IN  95
#define H1    64
#define H2    32

/* Test samples — same as all other sketches */
static const double ATTACK[N_IN] = {
   1.23, -0.45,  2.10,  3.85,  0.39, -0.10,
   1.47,  2.92,  0.38,  1.93,  0.04, -0.02,
   0.08,  0.04, -0.03,  0.02,
   0,1,0,0,0,0,0,0, 0,0,0,0,1,0,0,0,
   0,0,0,0,0,0,0,0, 0,0,0,0,1,0,0,0,
   0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0
};
static const double NORMAL[N_IN] = {
  -0.12, -0.45,  0.82, -0.91, -0.38,  0.00,
  -0.75, -0.82, -0.71, -0.63, -0.02,  0.01,
  -0.04, -0.02,  0.01, -0.01,
   0,0,1,0,0,0,0,0, 0,0,0,0,0,1,0,0,
   0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0
};

static double h1[H1], h2[H2];

static double mlp_proba(const double* x) {
  for (int j = 0; j < H1; j++) {
    double s = MLP_b0[j];
    for (int i = 0; i < N_IN; i++) s += x[i] * MLP_W0[i][j];
    h1[j] = s > 0.0 ? s : 0.0;
  }
  for (int j = 0; j < H2; j++) {
    double s = MLP_b1[j];
    for (int i = 0; i < H1; i++) s += h1[i] * MLP_W1[i][j];
    h2[j] = s > 0.0 ? s : 0.0;
  }
  double s = MLP_b2[0];
  for (int i = 0; i < H2; i++) s += h2[i] * MLP_W2[i][0];
  return 1.0 / (1.0 + exp(-s));
}

static void send_meta() {
  Serial.println("READY");
  Serial.println("MODEL=MLP_float64_95feat");
  Serial.println("BOARD=ESP32-C3-SuperMini");
  Serial.println("F1=0.9816");
  Serial.println("ROC_AUC=0.9843");
  Serial.println("SIZE_KB=41.7");
  Serial.println("SRAM_LIMIT_BYTES=409600");
  Serial.println("NOTE=VariantB_float64_TFLite_INT8_on_Wokwi_F1=0.9959_5us");
  Serial.println("HEADER:label,pred,prob,latency_us,free_heap_bytes,correct");
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  send_meta();
  Serial.print("FREE_HEAP_START="); Serial.println(ESP.getFreeHeap());
}

static void classify(const double* x, const char* label, int expected) {
  unsigned long t0 = micros();
  double prob      = mlp_proba(x);
  unsigned long dt = micros() - t0;
  int    pred      = (prob > 0.5) ? 1 : 0;
  uint32_t heap    = ESP.getFreeHeap();
  int    ok        = (pred == expected) ? 1 : 0;
  Serial.print(label); Serial.print(",");
  Serial.print(pred);  Serial.print(",");
  Serial.print(prob,4);Serial.print(",");
  Serial.print(dt);    Serial.print(",");
  Serial.print(heap);  Serial.print(",");
  Serial.println(ok);
}

void loop() {
  static uint32_t _cyc = 0;
  if (++_cyc % 20 == 1) send_meta();
  classify(ATTACK, "ATTACK", 1);
  classify(NORMAL, "NORMAL", 0);
  delay(2000);
}
