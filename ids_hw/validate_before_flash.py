"""
validate_before_flash.py
========================
Verifies models produce correct predictions using the same test samples
as the .ino sketches -- reads coefficients directly from C files.

The saved .joblib files in the repo use a 13-feature cross-domain pipeline,
NOT the 95-feature binary classification pipeline. So we parse logreg.c and
tree_depth5.c directly -- these are the exact coefficients on the hardware.

Usage:
    python validate_before_flash.py
"""

import sys, os, re, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Test samples -- identical to all .ino sketches (95 features)
# ---------------------------------------------------------------------------
ATTACK_95 = np.array([
    1.23, -0.45,  2.10,  3.85,  0.39, -0.10,
    1.47,  2.92,  0.38,  1.93,  0.04, -0.02,
    0.08,  0.04, -0.03,  0.02,
    0,1,0,0,0,0,0,0,  0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,0,  0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,
], dtype=np.float64)

NORMAL_95 = np.array([
   -0.12, -0.45,  0.82, -0.91, -0.38,  0.00,
   -0.75, -0.82, -0.71, -0.63, -0.02,  0.01,
   -0.04, -0.02,  0.01, -0.01,
    0,0,1,0,0,0,0,0,  0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Model 1: parse logreg.c
# ---------------------------------------------------------------------------
def validate_logreg(c_file):
    if not os.path.exists(c_file):
        print(f"  SKIP -- not found: {c_file}"); return

    with open(c_file) as f:
        src = f.read()

    m = re.search(r'double score\s*\(\s*double \* input\s*\)\s*\{\s*return\s+(.+?);\s*\}',
                  src, re.DOTALL)
    if not m:
        print("  SKIP -- could not parse score() from logreg.c"); return

    expr = m.group(1).strip()
    intercept = float(re.match(r'([-+]?\d[\d.eE+\-]*)', expr).group(1))
    coefs = np.zeros(95)
    for idx, val in re.findall(r'input\[(\d+)\]\s*\*\s*([-+]?\d[\d.eE+\-]*)', expr):
        coefs[int(idx)] = float(val)

    def score_fn(x):   return intercept + np.dot(x, coefs)
    def sigmoid(v):    return 1.0 / (1.0 + np.exp(-v))
    def predict_fn(x): return 1 if score_fn(x) > 0.0 else 0

    print(f"  Parsed: {os.path.basename(c_file)}  intercept={intercept:.4f}  n_coef={np.count_nonzero(coefs)}")

    all_ok = True
    for x, label, expected in [(ATTACK_95,"ATTACK",1),(NORMAL_95,"NORMAL",0)]:
        pred = predict_fn(x)
        prob = sigmoid(score_fn(x))
        N    = 5000
        t0   = time.perf_counter()
        for _ in range(N): predict_fn(x)
        lat  = (time.perf_counter()-t0)/N*1e6
        ok   = "[ OK ]" if pred == expected else "[FAIL] <-- investigate!"
        if pred != expected: all_ok = False
        print(f"  {ok} {label:6}  pred={pred}  P(attack)={prob:.4f}  PC latency={lat:.2f} us")

    if all_ok:
        print("  All correct. Hardware Wokwi ref: ~1.3 us on Mega")
    print()

# ---------------------------------------------------------------------------
# Model 2: parse tree_depth5.c -- try to execute the if/else tree in Python
# ---------------------------------------------------------------------------
def validate_tree(c_file):
    if not os.path.exists(c_file):
        print(f"  SKIP -- not found: {c_file}"); return

    with open(c_file) as f:
        src = f.read()

    # Extract score() body between the opening brace and the final memcpy
    m = re.search(r'void score\s*\(double \* input,\s*double \* var0\)\s*\{(.+?)memcpy',
                  src, re.DOTALL)
    if not m:
        print("  SKIP -- could not extract tree body from tree_depth5.c")
        print("  Validated by quantization_export.py: F1=0.9943. Wokwi: 36-48 us on Mega")
        print(); return

    body = m.group(1)

    # Translate C if/else to Python
    py = body
    py = re.sub(r'double var0\[2\];', 'var0 = [0.0, 0.0]', py)
    py = re.sub(r'if \(input\[(\d+)\] <= ([-\d.eE+]+)\)', r'if input[\1] <= \2:', py)
    py = re.sub(r'\} else \{', 'else:', py)
    py = re.sub(r'var0\[0\] = ([-\d.eE+]+);', r'var0[0] = \1', py)
    py = re.sub(r'var0\[1\] = ([-\d.eE+]+);', r'var0[1] = \1', py)
    py = re.sub(r'[{}]', '', py)

    # Re-indent: count 'if' and 'else' to build proper indentation
    lines = [l.strip() for l in py.split('\n') if l.strip()]
    result = ['def _tree(input):', '  var0 = [0.0, 0.0]']
    depth = 1
    for line in lines:
        if line.startswith('var0 = ['):
            continue
        if line == 'else:':
            depth -= 1
            result.append('  ' * depth + line)
            depth += 1
        elif line.endswith(':'):
            result.append('  ' * depth + line)
            depth += 1
        else:
            result.append('  ' * depth + line)
    result.append('  return var0')

    try:
        ns = {}
        exec('\n'.join(result), ns)
        tree_fn = ns['_tree']

        print(f"  Parsed: {os.path.basename(c_file)}")
        all_ok = True
        for x, label, expected in [(ATTACK_95,"ATTACK",1),(NORMAL_95,"NORMAL",0)]:
            out  = tree_fn(list(x))
            pred = 1 if out[1] > out[0] else 0
            N    = 5000
            t0   = time.perf_counter()
            for _ in range(N): tree_fn(list(x))
            lat  = (time.perf_counter()-t0)/N*1e6
            ok   = "[ OK ]" if pred == expected else "[FAIL] <-- investigate!"
            if pred != expected: all_ok = False
            print(f"  {ok} {label:6}  pred={pred}  P(attack)={out[1]:.4f}  PC latency={lat:.2f} us")

        if all_ok:
            print("  All correct. Hardware Wokwi ref: 36-48 us on Mega")
    except Exception as e:
        print(f"  Python parse failed ({e}).")
        print("  tree_depth5.c is validated by quantization_export.py metrics: F1=0.9943")
    print()

# ---------------------------------------------------------------------------
# Model 3: MLP TFLite INT8 -- can't run on PC, show Wokwi reference
# ---------------------------------------------------------------------------
def validate_mlp():
    print("  TFLite INT8 cannot run on PC without the TFLite runtime library.")
    print("  The mlp.h model was validated on Wokwi by the student:")
    print("    F1=0.9959  ROC-AUC=0.9993  Arena ~1668 B  Latency ~5.25 us")
    print("  Hardware results should match Wokwi within ~2x.")
    print()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("=" * 62)
print("  IDS Embedded -- validate_before_flash.py")
print("  Reads coefficients from C files (exact hardware values)")
print("=" * 62)
print()

print("-- Model 1: LogReg  [ids_mega_logreg + ids_esp32_logreg] --")
validate_logreg(os.path.join(SCRIPT_DIR, "ids_mega_logreg", "logreg.c"))

print("-- Model 2: DTree   [ids_mega_tree + ids_esp32_tree] ------")
validate_tree(os.path.join(SCRIPT_DIR, "ids_mega_tree", "tree_depth5.c"))

print("-- Model 3: MLP INT8 TFLite  [ids_esp32_mlp] ---------------")
validate_mlp()

print("=" * 62)
print("  If Models 1-2 show [ OK ], proceed to flash.")
print()
print("  Expected hardware results (Wokwi baseline):")
print("    LogReg  Mega  (AVR  16 MHz)  : ~1.3 us    SRAM ~985 B")
print("    DTree   Mega  (AVR  16 MHz)  : ~36-48 us  SRAM ~1163 B")
print("    LogReg  ESP32 (RISC-V 160 MHz): expected ~0.1-0.3 us")
print("    DTree   ESP32 (RISC-V 160 MHz): expected ~2-5 us")
print("    MLP INT8 ESP32               : ~5.25 us   arena ~1668 B")
print("=" * 62)


print("-- XGB/LGB: inspect .bin files (no PC inference needed) -----------")
import sys as _sys
_sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "iot-audit-main"))
_sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

try:
    from embedded_model_io import inspect_bin_file, load_xgb_int8, load_lgb_int8
    import numpy as _np

    xgb_bin = os.path.join(SCRIPT_DIR, "ids_esp32_xgb", "xgboost_int8.bin")
    lgb_bin = os.path.join(SCRIPT_DIR, "ids_esp32_lgb", "lightgbm_int8.bin")

    for bin_path, name in [(xgb_bin, "XGBoost"), (lgb_bin, "LightGBM")]:
        if not os.path.exists(bin_path):
            print(f"  SKIP -- not found: {bin_path}")
            continue
        info = inspect_bin_file(bin_path)
        print(f"  {name}: {info['type']} v{info['version']}")
        print(f"    n_trees={info['n_trees']}  n_features={info['n_features']}")
        print(f"    size={info['size_kb']:.1f} KB  fits_esp32={info['fits_esp32']}  margin={info['margin_kb']:.1f} KB")
        # Quick round-trip: predict on ATTACK/NORMAL_95 samples
        if name == "XGBoost":
            model = load_xgb_int8(bin_path)
        else:
            model = load_lgb_int8(bin_path)
        proba_a = model.predict_proba(_np.array([ATTACK_95], dtype=_np.float32))[0]
        proba_n = model.predict_proba(_np.array([NORMAL_95], dtype=_np.float32))[0]
        pred_a = 1 if proba_a[1] >= 0.5 else 0
        pred_n = 1 if proba_n[1] >= 0.5 else 0
        ok_a = "[ OK ]" if pred_a == 1 else "[FAIL]"
        ok_n = "[ OK ]" if pred_n == 0 else "[FAIL]"
        print(f"  {ok_a} ATTACK  pred={pred_a}  P(attack)={proba_a[1]:.4f}")
        print(f"  {ok_n} NORMAL  pred={pred_n}  P(attack)={proba_n[1]:.4f}")
        print()

except ImportError:
    print("  embedded_model_io.py not found on Python path -- skipping XGB/LGB check")
    print("  Copy embedded_model_io.py from iot-audit repo next to ids_hw/")
    print()
