/*
 * IDS Embedded -- MLP INT8 (TFLite Micro)
 * Board  : ESP32-C3 SuperMini  F1=0.9959  ROC-AUC=0.9993
 * Model  : mlp.h  (g_mlp_model, 13344 bytes)
 * Wokwi (simulation) : 5.25 us  arena 1668 B
 * Hardware (physical): ~837 us  arena 1676 B
 *   Note: Wokwi does not model RISC-V soft-float; hardware measures are accurate.
 */

#include <Arduino.h>
#include "mlp.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define N_FEATURES      95
#define TENSOR_ARENA_KB 48

static uint8_t tensor_arena[TENSOR_ARENA_KB * 1024];

using MlpResolver = tflite::MicroMutableOpResolver<6>;
static MlpResolver resolver;

static const tflite::Model*      tfl_model    = nullptr;
static tflite::MicroInterpreter* interpreter  = nullptr;
static TfLiteTensor*             input_tensor = nullptr;
static TfLiteTensor*             output_tensor= nullptr;

/* Quantization params -- read from model tensor at runtime (not hardcoded) */
static float   in_scale  = 0.0f;
static int32_t in_zp     = 0;
static float   out_scale = 0.0f;
static int32_t out_zp    = 0;
static bool    is_int8   = false;

static const float ATTACK[N_FEATURES] = {
    1.6905f, -0.4659f, 0.0328f, 0.4075f, -0.7889f, 0.0021f, -0.0009f, -1.7547f, 1.0177f, 0.6005f, -0.6254f, -0.1715f, 0.5053f, -0.2614f, -0.2427f, -1.4532f, 0.0000f, 0.0000f, 1.0000f, 1.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 1.0000f, 1.0000f, 0.0000f, 0.0000f, 1.0000f, 1.0000f, 0.0000f, 0.0000f, 1.0000f, 1.0000f, 0.0000f, 0.0000f, 1.0000f, 1.0000f, 0.0000f, 0.0000f, 1.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 1.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 1.0000f, 1.0000f, 0.0000f, 0.0000f, 1.0000f, 1.0000f, 0.0000f, 1.0000f, 1.0000f, 0.0000f, 1.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 1.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 1.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 1.0000f, 1.0000f, 0.0000f
};
static const float NORMAL[N_FEATURES] = {
    -1.5882f, 0.2580f, 0.9327f, -0.1483f, -0.0705f, 1.3929f, -1.2497f, -1.4976f, -0.6540f, -1.1966f, 1.3463f, 0.3097f, -0.8641f, -0.6145f, 2.8611f, -0.6101f, 0.0000f, 0.0000f, 1.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 1.0000f, 0.0000f, 1.0000f, 1.0000f, 0.0000f, 1.0000f, 0.0000f, 0.0000f, 1.0000f, 1.0000f, 0.0000f, 1.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 1.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 1.0000f, 0.0000f, 1.0000f, 0.0000f, 0.0000f, 1.0000f, 1.0000f, 1.0000f, 0.0000f, 1.0000f, 1.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 1.0000f, 0.0000f, 1.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 1.0000f, 1.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 1.0000f, 0.0000f, 0.0000f, 0.0000f, 1.0000f
};

static uint32_t g_arena_bytes = 0;   /* set after AllocateTensors */

static void send_meta() {
  Serial.println("READY");
  Serial.println("MODEL=MLP_TFLite_INT8");
  Serial.println("BOARD=ESP32-C3-SuperMini");
  Serial.println("F1=0.9959");
  Serial.println("ROC_AUC=0.9993");
  Serial.println("SIZE_KB=13.03");
  Serial.println("SRAM_LIMIT_BYTES=409600");
  /* Repeat arena/heap every call so the monitor captures it even on late connect */
  if (interpreter != nullptr) {
    g_arena_bytes = interpreter->arena_used_bytes();
  Serial.print("ARENA_USED_BYTES="); Serial.println(g_arena_bytes);
  }
  Serial.print("FREE_HEAP="); Serial.println(ESP.getFreeHeap());
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  send_meta();

  tfl_model = tflite::GetModel(g_mlp_model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR:schema_mismatch"); while (true) {}
  }

  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddLogistic();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddReshape();

  static tflite::MicroInterpreter static_interp(
      tfl_model, resolver, tensor_arena, sizeof(tensor_arena));
  interpreter = &static_interp;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR:AllocateTensors"); while (true) {}
  }

  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);

  /* Read quantization params from the model -- never hardcode these */
  is_int8   = (input_tensor->type == kTfLiteInt8);
  in_scale  = input_tensor->params.scale;
  in_zp     = input_tensor->params.zero_point;
  out_scale = output_tensor->params.scale;
  out_zp    = output_tensor->params.zero_point;

  Serial.print("INPUT_TYPE=");       Serial.println((int)input_tensor->type);
  Serial.print("IN_SCALE=");         Serial.println(in_scale, 8);
  Serial.print("IN_ZP=");            Serial.println(in_zp);
  Serial.print("OUT_SCALE=");        Serial.println(out_scale, 8);
  Serial.print("OUT_ZP=");           Serial.println(out_zp);
  send_meta();  /* also prints ARENA_USED_BYTES and FREE_HEAP */
  Serial.println("HEADER:label,pred,prob,latency_us,arena_bytes,correct");
}

static void classify(const float* features, const char* label, int expected) {
  if (is_int8 && in_scale > 0.0f) {
    for (int i = 0; i < N_FEATURES; i++) {
      int q = (int)roundf(features[i] / in_scale) + in_zp;
      if (q < -128) q = -128;
      if (q >  127) q =  127;
      input_tensor->data.int8[i] = (int8_t)q;
    }
  } else {
    for (int i = 0; i < N_FEATURES; i++)
      input_tensor->data.f[i] = features[i];
  }

  unsigned long t0 = micros();
  TfLiteStatus s   = interpreter->Invoke();
  unsigned long dt = micros() - t0;

  if (s != kTfLiteOk) {
    Serial.print("ERROR:invoke="); Serial.println((int)s); return;
  }

  float prob; int pred;
  if (output_tensor->type == kTfLiteInt8 && out_scale > 0.0f) {
    int8_t raw = output_tensor->data.int8[0];
    prob = (float)(raw - out_zp) * out_scale;
    pred = (raw > out_zp) ? 1 : 0;
  } else {
    prob = output_tensor->data.f[0];
    pred = (prob > 0.5f) ? 1 : 0;
  }

  int ok = (pred == expected) ? 1 : 0;
  Serial.print(label);  Serial.print(",");
  Serial.print(pred);   Serial.print(",");
  Serial.print(prob,4); Serial.print(",");
  Serial.print(dt);     Serial.print(",");
  Serial.print(interpreter->arena_used_bytes()); Serial.print(",");
  Serial.println(ok);
}

void loop() {
  static uint32_t _cyc = 0;
  if (++_cyc % 20 == 1) send_meta();
  classify(ATTACK, "ATTACK", 1);
  classify(NORMAL, "NORMAL", 0);
  delay(2000);
}
