/*
 * IDS Embedded -- Decision Tree depth=5
 * Board  : Arduino Mega 2560  F1=0.9943  ROC-AUC=0.9856
 * Size   : 4.76 KB            SRAM ~1163 B / 8192 B
 * Wokwi  : 36-48 us latency
 */

#include <string.h>
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

static long freeRam() {
  extern int __heap_start, *__brkval;
  int v;
  return (long)&v - ((__brkval==0)?(long)&__heap_start:(long)__brkval);
}

static void send_meta() {
  Serial.println(F("READY"));
  Serial.println(F("MODEL=DecisionTree_depth5"));
  Serial.println(F("BOARD=ArduinoMega2560"));
  Serial.println(F("F1=0.9943"));
  Serial.println(F("ROC_AUC=0.9856"));
  Serial.println(F("SIZE_KB=4.76"));
  Serial.println(F("SRAM_LIMIT_BYTES=8192"));
  Serial.println(F("HEADER:label,pred,prob,latency_us,sram_free_bytes,correct"));
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  send_meta();
}

static void classify(double* features, const char* label, int expected) {
  double out[2];
  unsigned long t0 = micros();
  int    pred      = predict(features);
  unsigned long dt = micros() - t0;
  score(features, out);
  long   sram      = freeRam();
  int    ok        = (pred == expected) ? 1 : 0;
  Serial.print(label);     Serial.print(F(","));
  Serial.print(pred);      Serial.print(F(","));
  Serial.print(out[1], 4); Serial.print(F(","));
  Serial.print(dt);        Serial.print(F(","));
  Serial.print(sram);      Serial.print(F(","));
  Serial.println(ok);
}

void loop() {
  static uint32_t _cyc = 0;
  if (++_cyc % 20 == 1) send_meta();
  classify(sample_attack, "ATTACK", 1);
  classify(sample_normal, "NORMAL", 0);
  delay(2000);
}
