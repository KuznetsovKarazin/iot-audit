/*
 * IDS Embedded -- Logistic Regression
 * Board  : Arduino Mega 2560  (ATmega2560, 16 MHz, 8 KB SRAM)
 * Model  : logreg.c           F1=0.9900  ROC-AUC=0.9945
 * Size   : 3.20 KB            SRAM ~985 B / 8192 B
 * Wokwi  : ~1.3 us latency
 */

#include <math.h>
#define N_FEATURES 95

/* extern "C" required: .ino is C++, logreg.c is C.
   Without this, C++ name mangling breaks the linker. */
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

static long freeRam() {
  extern int __heap_start, *__brkval;
  int v;
  return (long)&v - ((__brkval==0)?(long)&__heap_start:(long)__brkval);
}

static void send_meta() {
  Serial.println(F("READY"));
  Serial.println(F("MODEL=LogisticRegression"));
  Serial.println(F("BOARD=ArduinoMega2560"));
  Serial.println(F("F1=0.9900"));
  Serial.println(F("ROC_AUC=0.9945"));
  Serial.println(F("SIZE_KB=3.20"));
  Serial.println(F("SRAM_LIMIT_BYTES=8192"));
  Serial.println(F("HEADER:label,pred,prob,latency_us,sram_free_bytes,correct"));
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  send_meta();
}

static void classify(double* features, const char* label, int expected) {
  unsigned long t0 = micros();
  int    pred      = predict(features);
  unsigned long dt = micros() - t0;
  double prob      = score_proba(features);
  long   sram      = freeRam();
  int    ok        = (pred == expected) ? 1 : 0;
  Serial.print(label);  Serial.print(F(","));
  Serial.print(pred);   Serial.print(F(","));
  Serial.print(prob,4); Serial.print(F(","));
  Serial.print(dt);     Serial.print(F(","));
  Serial.print(sram);   Serial.print(F(","));
  Serial.println(ok);
}

void loop() {
  static uint32_t _cyc = 0;
  if (++_cyc % 20 == 1) send_meta();
  classify(sample_attack, "ATTACK", 1);
  classify(sample_normal, "NORMAL", 0);
  delay(2000);
}
