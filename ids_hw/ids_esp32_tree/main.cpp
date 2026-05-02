/*
 * IDS Embedded -- Decision Tree depth=5 on ESP32-C3
 * Board  : ESP32-C3 SuperMini  F1=0.9943  ROC-AUC=0.9856
 * Same model as Arduino Mega -- compare latencies.
 */

#include <string.h>
#include <Arduino.h>
#define N_FEATURES 95

extern "C" {
  void score(double* input, double* output);
  int  predict(double* input);
}

double sample_attack[N_FEATURES] = {
   0, 0, -1.0, 0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
double sample_normal[N_FEATURES] = {
   0, 0,  1.0, 0,  0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

static void send_meta() {
  Serial.println("READY");
  Serial.println("MODEL=DecisionTree_depth5");
  Serial.println("BOARD=ESP32-C3-SuperMini");
  Serial.println("F1=0.9943");
  Serial.println("ROC_AUC=0.9856");
  Serial.println("SIZE_KB=4.76");
  Serial.println("SRAM_LIMIT_BYTES=409600");
  /* DTree: model in Flash (4.76 KB). Runtime SRAM: stack only */
  Serial.println("SRAM_MODEL_BYTES=0");
  Serial.println("HEADER:label,pred,prob,latency_us,free_heap_bytes,correct");
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  send_meta();
  Serial.print("FREE_HEAP_START="); Serial.println(ESP.getFreeHeap());
}

static void classify(double* features, const char* label, int expected) {
  double out[2];
  unsigned long t0 = micros();
  int    pred      = predict(features);
  unsigned long dt = micros() - t0;
  score(features, out);
  uint32_t heap    = ESP.getFreeHeap();
  int    ok        = (pred == expected) ? 1 : 0;
  Serial.print(label);     Serial.print(",");
  Serial.print(pred);      Serial.print(",");
  Serial.print(out[1], 4); Serial.print(",");
  Serial.print(dt);        Serial.print(",");
  Serial.print(heap);      Serial.print(",");
  Serial.println(ok);
}

void loop() {
  static uint32_t _cyc = 0;
  if (++_cyc % 20 == 1) { send_meta(); }
  classify(sample_attack, "ATTACK", 1);
  classify(sample_normal, "NORMAL", 0);
  delay(2000);
}
