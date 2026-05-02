/*
 * IDS Embedded -- XGBoost INT8 binary format
 * Board  : ESP32-C3 SuperMini  (RISC-V 160 MHz, 400 KB SRAM)
 * Model  : xgboost_int8.h   (g_xgboost_model[], stored in Flash)
 * Format : XGBI v2 custom INT8 binary (embedded_model_io.py)
 * Dataset: TON_IoT binary    F1=0.9989  ROC-AUC=1.0000
 * Size   : 369.52 KB Flash   SRAM during inference: ~25 KB
 */

#include <Arduino.h>
#include <math.h>
#include <string.h>
#include "xgboost_int8.h"   /* g_xgboost_model[], g_xgboost_model_len */

#define N_FEATURES  95
#define MAX_TREES   300

/* Parsed metadata in SRAM (~2.4 KB) */
struct XGBTree { uint32_t offset; int32_t n_nodes; };
static XGBTree  g_trees[MAX_TREES];
static uint32_t g_n_trees, g_n_features;
static float    g_leaf_scale, g_thr_scale;
static uint32_t g_leaf_data_offset, g_thr_data_offset;
static bool     g_model_ok = false;

static inline int32_t  rd_i32(const uint8_t* p, uint32_t o){ int32_t  v; memcpy(&v,p+o,4); return v; }
static inline uint32_t rd_u32(const uint8_t* p, uint32_t o){ uint32_t v; memcpy(&v,p+o,4); return v; }
static inline float    rd_f32(const uint8_t* p, uint32_t o){ float    v; memcpy(&v,p+o,4); return v; }

/* Test samples — validated to give correct predictions for XGB */
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
    if (d[0]!='X'||d[1]!='G'||d[2]!='B'||d[3]!='I') return false;
    uint16_t ver; memcpy(&ver,d+4,2); if (ver!=2) return false;
    g_n_trees    = rd_u32(d, 6);
    g_n_features = rd_u32(d,10);
    g_leaf_scale = rd_f32(d,14);
    g_thr_scale  = rd_f32(d,18);
    if (g_n_trees > MAX_TREES) return false;
    uint32_t off = 22;
    for (uint32_t t = 0; t < g_n_trees; t++) {
        g_trees[t].offset  = off;
        int32_t n          = (int32_t)rd_u32(d, off); off += 4;
        g_trees[t].n_nodes = n;
        off += (uint32_t)n * (4+4+4+1+4+4);
    }
    g_leaf_data_offset = off;
    uint32_t tot_leaves=0, tot_splits=0;
    for (uint32_t t = 0; t < g_n_trees; t++) {
        int32_t n = g_trees[t].n_nodes;
        const uint8_t* is_leaf = d + g_trees[t].offset + 4 + (uint32_t)n*12;
        for (int32_t i=0; i<n; i++) { if(is_leaf[i]) tot_leaves++; else tot_splits++; }
    }
    g_thr_data_offset = g_leaf_data_offset + tot_leaves;
    return true;
}

static float predict_tree(const uint8_t* d, uint32_t ti, const float* x) {
    const uint8_t* tb = d + g_trees[ti].offset;
    int32_t  n  = (int32_t)rd_u32(tb,0);
    const uint8_t *lc  = tb+4, *rc  = tb+4+(uint32_t)n*4,
                  *si  = tb+4+(uint32_t)n*8,
                  *isl = tb+4+(uint32_t)n*12,
                  *lpp = tb+4+(uint32_t)n*12+(uint32_t)n,
                  *tpp = tb+4+(uint32_t)n*13+(uint32_t)n*4;
    int32_t node=0;
    while (!isl[node]) {
        uint32_t feat = rd_u32(si,(uint32_t)node*4);
        if (feat>=N_FEATURES) break;
        float thr = (float)(int8_t)d[g_thr_data_offset+(uint32_t)rd_i32(tpp,(uint32_t)node*4)] * g_thr_scale;
        node = (x[feat]<=thr) ? rd_i32(lc,(uint32_t)node*4) : rd_i32(rc,(uint32_t)node*4);
        if (node<0||node>=n) break;
    }
    return (float)(int8_t)d[g_leaf_data_offset+(uint32_t)rd_i32(lpp,(uint32_t)node*4)] * g_leaf_scale;
}

static float xgb_proba(const float* x) {
    float s=0;
    for (uint32_t t=0; t<g_n_trees; t++) s += predict_tree(g_xgboost_model,t,x);
    return 1.0f/(1.0f+expf(-s));
}

static void send_meta() {
    Serial.println("READY");
    Serial.println("MODEL=XGBoost_INT8");
    Serial.println("BOARD=ESP32-C3-SuperMini");
    Serial.println("F1=0.9989");
    Serial.println("ROC_AUC=1.0000");
    Serial.println("SIZE_KB=369.52");
    /* XGB: 300 trees * 8 B/tree = 2400 B metadata in SRAM */
  Serial.println("SRAM_LIMIT_BYTES=409600");
  Serial.println("SRAM_MODEL_BYTES=2400");
}

void setup() {
    Serial.begin(115200); delay(1000);
    send_meta();
    g_model_ok = model_init(g_xgboost_model);
    Serial.print("MODEL_INIT="); Serial.println(g_model_ok?"OK":"FAIL");
    Serial.print("FREE_HEAP_START="); Serial.println(ESP.getFreeHeap());
    Serial.println("HEADER:label,pred,prob,latency_us,free_heap_bytes,correct");
}

static void classify(const float* x, const char* label, int expected) {
    if (!g_model_ok) { Serial.println("ERROR:model_not_init"); return; }
    unsigned long t0=micros(); float prob=xgb_proba(x); unsigned long dt=micros()-t0;
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
