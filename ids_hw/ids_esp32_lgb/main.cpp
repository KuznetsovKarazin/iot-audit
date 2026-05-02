/*
 * IDS Embedded -- LightGBM INT8 binary format
 * Board  : ESP32-C3 SuperMini  (RISC-V 160 MHz, 400 KB SRAM)
 * Model  : lightgbm_int8.h  (g_lightgbm_model[], stored in Flash)
 * Format : LGBI v2 custom INT8 binary (embedded_model_io.py)
 * Dataset: TON_IoT binary    F1=0.9992  ROC-AUC=1.0000
 * Size   : 73.85 KB Flash    SRAM during inference: ~6.4 KB
 */

#include <Arduino.h>
#include <math.h>
#include <string.h>
#include "lightgbm_int8.h"  /* g_lightgbm_model[], g_lightgbm_model_len */

#define N_FEATURES  95
#define MAX_TREES   400

struct LGBTree { uint32_t n_splits,n_leaves,feat_off,thr_off,leaf_off; };
static LGBTree  g_trees[MAX_TREES];
static uint32_t g_n_trees, g_n_features;
static float    g_thr_scale, g_leaf_scale;
static uint32_t g_thr_data_off, g_leaf_data_off;
static bool     g_model_ok = false;

static inline uint32_t rd_u32(const uint8_t* p, uint32_t o){ uint32_t v; memcpy(&v,p+o,4); return v; }
static inline float    rd_f32(const uint8_t* p, uint32_t o){ float    v; memcpy(&v,p+o,4); return v; }

/* Test samples — validated to give correct predictions for LGB */
static const float ATTACK[N_FEATURES] = {
  -2.4172f, -0.6556f, -0.5932f, -0.4492f,  0.8137f,  2.5208f,
   1.9843f, -0.7582f,  1.7889f, -3.5899f,  0.5914f,  1.8191f,
  -0.3681f,  0.6257f,  0.8258f,  1.8141f,
   1,1,1,1,0,0,1,0, 0,0,1,0,0,0,1,0,
   1,1,0,0,1,0,1,1, 0,1,1,0,0,0,1,0,
   0,0,0,0,0,0,1,0, 0,0,1,1,1,1,0,0,
   0,0,1,1,1,0,1,0, 1,0,0,1,0,0,0,1,
   0,1,0,1,0,0,0,0, 1,0,0,1,0,1,0
};
static const float NORMAL[N_FEATURES] = {
   0.9934f, -0.2765f,  1.2954f,  3.0461f, -0.4683f, -0.4683f,
   3.1584f,  1.5349f, -0.9389f,  1.0851f, -0.9268f, -0.9315f,
   0.4839f, -3.8266f, -3.4498f, -1.1246f,
   1,1,0,0,0,0,0,0, 1,0,0,1,0,1,1,0,
   0,0,0,0,0,0,0,0, 1,0,0,0,1,0,0,1,
   0,1,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,
   0,0,1,0,0,0,1,0, 1,0,0,0,0,1,1,0,
   0,0,1,1,1,1,0,0, 0,1,0
};

static bool model_init(const uint8_t* d) {
    if (d[0]!='L'||d[1]!='G'||d[2]!='B'||d[3]!='I') return false;
    uint16_t ver; memcpy(&ver,d+4,2); if (ver!=2) return false;
    g_n_trees    = rd_u32(d, 6);
    g_n_features = rd_u32(d,10);
    g_thr_scale  = rd_f32(d,14);
    g_leaf_scale = rd_f32(d,18);
    if (g_n_trees > MAX_TREES) return false;
    uint32_t off=22, thr_idx=0, leaf_idx=0;
    for (uint32_t t=0; t<g_n_trees; t++) {
        uint32_t ns=rd_u32(d,off); off+=4;
        uint32_t nl=rd_u32(d,off); off+=4;
        g_trees[t]={ns,nl,off,thr_idx,leaf_idx};
        off+=ns*4; thr_idx+=ns; leaf_idx+=nl;
    }
    g_thr_data_off  = off;
    g_leaf_data_off = off + thr_idx;
    return true;
}

static float lgb_proba(const float* x) {
    const uint8_t* d = g_lightgbm_model;
    float score=0.0f;
    for (uint32_t t=0; t<g_n_trees; t++) {
        const LGBTree& tr = g_trees[t];
        const uint8_t* feats = d + tr.feat_off;
        uint32_t ns=tr.n_splits, node=0;
        while (node < ns) {
            uint32_t feat = rd_u32(feats, node*4);
            if (feat >= N_FEATURES) feat = N_FEATURES-1;
            float thr = (float)(int8_t)d[g_thr_data_off+tr.thr_off+node] * g_thr_scale;
            node = (x[feat]<=thr) ? (2*node+1) : (2*node+2);
        }
        uint32_t li = node-ns;
        if (li >= tr.n_leaves) li = tr.n_leaves-1;
        score += (float)(int8_t)d[g_leaf_data_off+tr.leaf_off+li] * g_leaf_scale;
    }
    return 1.0f/(1.0f+expf(-score));
}

static void send_meta() {
    Serial.println("READY");
    Serial.println("MODEL=LightGBM_INT8");
    Serial.println("BOARD=ESP32-C3-SuperMini");
    Serial.println("F1=0.9992");
    Serial.println("ROC_AUC=1.0000");
    Serial.println("SIZE_KB=73.85");
    /* LGB: 400 trees * 20 B/tree = 8000 B metadata in SRAM */
  Serial.println("SRAM_LIMIT_BYTES=409600");
  Serial.println("SRAM_MODEL_BYTES=8000");
}

void setup() {
    Serial.begin(115200); delay(1000);
    send_meta();
    g_model_ok = model_init(g_lightgbm_model);
    Serial.print("MODEL_INIT="); Serial.println(g_model_ok?"OK":"FAIL");
    Serial.print("FREE_HEAP_START="); Serial.println(ESP.getFreeHeap());
    Serial.println("HEADER:label,pred,prob,latency_us,free_heap_bytes,correct");
}

static void classify(const float* x, const char* label, int expected) {
    if (!g_model_ok) { Serial.println("ERROR:model_not_init"); return; }
    unsigned long t0=micros(); float prob=lgb_proba(x); unsigned long dt=micros()-t0;
    int pred=(prob>0.5f)?1:0, ok=(pred==expected)?1:0;
    Serial.print(label);  Serial.print(",");
    Serial.print(pred);   Serial.print(",");
    Serial.print(prob,4); Serial.print(",");
    Serial.print(dt);     Serial.print(",");
    Serial.print(ESP.getFreeHeap()); Serial.print(",");
    Serial.println(ok);
}

void loop() {
    static uint32_t _cyc=0;
    if (++_cyc%20==1) send_meta();
    classify(ATTACK,"ATTACK",1);
    classify(NORMAL,"NORMAL",0);
    delay(2000);
}
