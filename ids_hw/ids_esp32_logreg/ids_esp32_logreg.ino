/*
 * IDS Embedded -- Logistic Regression on ESP32-C3
 * Board  : ESP32-C3 SuperMini  F1=0.9900  ROC-AUC=0.9945
 * Size   : 3.20 KB
 * Same model as Arduino Mega -- compare latencies across platforms.
 */

#include <math.h>
#include <Arduino.h>
#define N_FEATURES 95

extern "C" {
  int    predict(double* input);
  double score(double* input);
  double score_proba(double* input);
}

double sample_attack[N_FEATURES] = {
   1.23, -0.45,  2.10,  3.85,  0.39, -0.10,
   1.47,  2.92,  0.38,  1.93,  0.04, -0.02,
   0.08,  0.04, -0.03,  0.02,
   0,1,0,0,0,0,0,0, 0,0,0,0,1,0,0,0,
   0,0,0,0,0,0,0,0, 0,0,0,0,1,0,0,0,
   0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0
};
double sample_normal[N_FEATURES] = {
  -0.12, -0.45,  0.82, -0.91, -0.38,  0.00,
  -0.75, -0.82, -0.71, -0.63, -0.02,  0.01,
  -0.04, -0.02,  0.01, -0.01,
   0,0,1,0,0,0,0,0, 0,0,0,0,0,1,0,0,
   0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0
};

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("READY");
  Serial.println("MODEL=LogisticRegression");
  Serial.println("BOARD=ESP32-C3-SuperMini");
  Serial.println("F1=0.9900");
  Serial.println("ROC_AUC=0.9945");
  Serial.println("SIZE_KB=3.20");
  Serial.println("SRAM_LIMIT_BYTES=409600");
  Serial.print("FREE_HEAP_START="); Serial.println(ESP.getFreeHeap());
  Serial.println("HEADER:label,pred,prob,latency_us,free_heap_bytes,correct");
}

static void classify(double* features, const char* label, int expected) {
  unsigned long t0 = micros();
  int    pred      = predict(features);
  unsigned long dt = micros() - t0;
  double prob      = score_proba(features);
  uint32_t heap    = ESP.getFreeHeap();
  int    ok        = (pred == expected) ? 1 : 0;
  Serial.print(label);  Serial.print(",");
  Serial.print(pred);   Serial.print(",");
  Serial.print(prob,4); Serial.print(",");
  Serial.print(dt);     Serial.print(",");
  Serial.print(heap);   Serial.print(",");
  Serial.println(ok);
}

void loop() {
  classify(sample_attack, "ATTACK", 1);
  classify(sample_normal, "NORMAL", 0);
  delay(2000);
}
