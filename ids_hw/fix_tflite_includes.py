# fix_tflite_includes.py
# PlatformIO pre-script: adds all third_party include paths bundled inside
# esp-tflite-micro that are NOT exported via library.json.
#
# Missing headers and their locations inside esp-tflite-micro/:
#   flatbuffers/flatbuffers.h  -> third_party/flatbuffers/include/
#   kiss_fft.h                 -> third_party/kissfft/
#   fixedpoint/fixedpoint.h    -> third_party/gemmlowp/
#   ruy/ruy.h                  -> third_party/ruy/

Import("env")
import os, sys

libdeps_dir = env.get("PROJECT_LIBDEPS_DIR", "")
pio_env     = env.get("PIOENV", "esp32_mlp")
tflite_root = os.path.join(libdeps_dir, pio_env, "esp-tflite-micro")

if not os.path.isdir(tflite_root):
    print(f"[fix_tflite] WARNING: esp-tflite-micro not found at {tflite_root}")
    print(f"[fix_tflite] Run once without upload to let PIO download deps first:")
    print(f"[fix_tflite]   pio run -e esp32_mlp")
    Return()

# All third_party subdirs that need to be on the include path
third_party = os.path.join(tflite_root, "third_party")

# Known required paths
candidates = [
    os.path.join(third_party, "flatbuffers", "include"),  # flatbuffers/flatbuffers.h
    os.path.join(third_party, "kissfft"),                 # kiss_fft.h
    os.path.join(third_party, "gemmlowp"),                # fixedpoint/fixedpoint.h
    os.path.join(third_party, "ruy"),                     # ruy/ruy.h
    tflite_root,                                          # tensorflow/... root includes
]

added = []
for path in candidates:
    if os.path.isdir(path):
        env.Append(CPPPATH=[path])
        added.append(path)
    else:
        print(f"[fix_tflite] not found (may be OK): {path}")

print(f"[fix_tflite] Added {len(added)} include paths:")
for p in added:
    print(f"  {p}")
