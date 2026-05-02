# ids_hw/ -- IDS Embedded Hardware Benchmark

Flash ML intrusion detection models to real MCU hardware and capture
inference latency + SRAM usage for the Electronics journal paper.

## Models and boards

| Model | Mega | ESP32-C3 | F1 | ROC-AUC | Size | Wokwi latency |
|---|---|---|---|---|---|---|
| Logistic Regression | YES | YES | 0.9900 | 0.9945 | 3.20 KB | ~1.3 us (Mega) |
| Decision Tree d=5 | YES | YES | 0.9943 | 0.9856 | 4.76 KB | ~36-48 us (Mega) |
| MLP INT8 (TFLite) | no | YES | 0.9959 | 0.9993 | 13.03 KB | ~5.25 us (ESP32) |

**Why LogReg and Tree on both boards?**
Same model, two platforms: AVR 8-bit 16 MHz vs RISC-V 160 MHz.
Shows the latency benefit of a faster CPU with the same algorithm.
ESP32-C3 has 400 KB SRAM -- all three models fit easily.

---

## Quick start

```powershell
# Step 1: verify models on PC first
pip install joblib scikit-learn numpy
python validate_before_flash.py --repo E:\path\to\iot-audit-main

# Step 2: flash and benchmark
.\flash.ps1 -Device mega    -Model logreg  -Collect 20
.\flash.ps1 -Device mega    -Model tree    -Collect 20
.\flash.ps1 -Device esp32c3 -Model logreg  -Collect 20
.\flash.ps1 -Device esp32c3 -Model tree    -Collect 20
.\flash.ps1 -Device esp32c3 -Model mlp     -Collect 20

# Just monitor (no auto-collect)
.\flash.ps1 -Device mega    -Model logreg  -Monitor

# Flash both devices sequentially (same model)
.\flash.ps1 -Device both    -Model logreg  -Collect 20
```

**First run** downloads arduino-cli + ESP32 core (~300 MB) and PlatformIO + TFLite Micro (~50 MB).
Subsequent runs use the cache and are fast (~15 s).

---

## Alternative: PlatformIO only

```powershell
pip install platformio
cd ids_hw

pio run -e mega_logreg  -t upload
pio run -e mega_tree    -t upload
pio run -e esp32_logreg -t upload
pio run -e esp32_tree   -t upload
pio run -e esp32_mlp    -t upload    # downloads TFLite Micro on first run

pio device monitor -b 115200
```

---

## File structure

```
ids_hw/
|-- flash.ps1                       main script
|-- platformio.ini                  PlatformIO config (5 envs)
|-- validate_before_flash.py        PC verification
|-- README.md
|
|-- ids_mega_logreg/
|   |-- ids_mega_logreg.ino
|   `-- logreg.c                    m2cgen export
|
|-- ids_mega_tree/
|   |-- ids_mega_tree.ino
|   `-- tree_depth5.c               m2cgen export (26-leaf if/else)
|
|-- ids_esp32_logreg/
|   |-- ids_esp32_logreg.ino        same model, ESP32-C3 platform
|   `-- logreg.c
|
|-- ids_esp32_tree/
|   |-- ids_esp32_tree.ino
|   `-- tree_depth5.c
|
`-- ids_esp32_mlp/
    |-- ids_esp32_mlp.ino           TFLite Micro sketch
    `-- mlp.h                       g_mlp_model[] 13344 bytes INT8
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| avrdude: timeout | Wrong COM port -- check Device Manager |
| Failed to connect to ESP32 | Hold BOOT, press RESET, release BOOT, retry |
| Serial Monitor empty (ESP32) | CDCOnBoot=cdc already set in FQBN -- check data cable |
| execution of scripts is disabled | Set-ExecutionPolicy -Scope CurrentUser RemoteSigned |
| pio not found | pip install platformio, restart PowerShell |
| TFLite schema mismatch | pio pkg update in ids_hw folder |
| validate_before_flash: models not found | python validate_before_flash.py --repo E:\iot-audit-main |
