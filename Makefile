# ═══════════════════════════════════════════════════════════════
# S-MoE Engine — Makefile
# ═══════════════════════════════════════════════════════════════
# Target : Apple Silicon (arm64)  |  C++20  |  Metal  |  macOS 14+
#
# First-time setup (any new clone):
#   make setup          ← creates .venv and installs Python deps
#
# Then, every day:
#   make shatter MODEL=./checkpoints/qwen3-235b-instruct OUT=./vault --validate
#   make probe   MODEL=./checkpoints/qwen3-235b-instruct
#   make all             ← build the C++ engine
# ═══════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────
SRC_DIR     := src
BUILD_DIR   := build
SCRIPTS_DIR := scripts
VENV_DIR    := .venv
PYTHON      := $(VENV_DIR)/bin/python3
PIP         := $(VENV_DIR)/bin/pip

# ── C++ / Metal toolchain ────────────────────────────────────
CXX      := clang++
XCRUN    := xcrun
CXXFLAGS := -std=c++20 -g -O3 -Wall -Wextra -Wno-unused-parameter \
             -Wno-unused-private-field \
             -arch arm64 -mmacosx-version-min=14.0
MMFLAGS  := -std=c++20 -ObjC++ -g -O3 -arch arm64 -mmacosx-version-min=14.0
LDFLAGS  := -framework Metal -framework Foundation \
             -framework MetalPerformanceShaders

SRCS_CXX  := $(SRC_DIR)/main.cpp \
              $(SRC_DIR)/engine.cpp \
              $(SRC_DIR)/io/streamer.cpp \
              $(SRC_DIR)/scout/scout.cpp \
              $(SRC_DIR)/prefill.cpp \
              $(SRC_DIR)/decode.cpp
SRCS_MM   := $(SRC_DIR)/compute/metal_bridge.mm
METAL_SRC := $(SRC_DIR)/compute/kernels.metal
METAL_LIB := $(BUILD_DIR)/kernels.metallib
MSL_HEADER:= $(BUILD_DIR)/kernels_msl.h
OBJS_CXX  := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS_CXX))
OBJS_MM   := $(patsubst $(SRC_DIR)/%.mm,  $(BUILD_DIR)/%.o, $(SRCS_MM))
TARGET    := $(BUILD_DIR)/smoe-engine

# ─────────────────────────────────────────────────────────────
# PYTHON ENVIRONMENT
# ─────────────────────────────────────────────────────────────

# Guard: abort any python target if .venv has not been created yet.
_require_venv:
	@test -f $(PYTHON) || ( \
	  echo ""; \
	  echo "  ✗  .venv not found — run 'make setup' first."; \
	  echo ""; \
	  exit 1 )

## setup: Create .venv and install all Python dependencies.
##        Run this once after cloning the repo.
.PHONY: setup
setup:
	@echo "  🐍  Creating Python virtual environment in $(VENV_DIR)/ …"
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip -q
	$(PIP) install -r $(SCRIPTS_DIR)/requirements.txt
	@echo ""
	@echo "  ✓  Environment ready.  Activate with:"
	@echo "       source $(VENV_DIR)/bin/activate"
	@echo ""

# ─────────────────────────────────────────────────────────────
# OFFLINE SCULPTOR
# ─────────────────────────────────────────────────────────────

## shatter: Slice a supported MoE monolith into a .smoe vault.
##   Usage : make shatter MODEL=<model_dir> OUT=<output_dir>
##   Flags : set ARGS to pass extra flags (e.g. ARGS="--max-layers 2")
.PHONY: shatter
shatter: _require_venv
	@test -n "$(MODEL)" || (echo "  ✗  Usage: make shatter MODEL=<dir> OUT=<dir>"; exit 1)
	@test -n "$(OUT)"   || (echo "  ✗  Usage: make shatter MODEL=<dir> OUT=<dir>"; exit 1)
	$(PYTHON) $(SCRIPTS_DIR)/shatter_moe.py $(MODEL) $(OUT) --validate $(ARGS)

## probe: Dry-run topology detection without writing any files.
##   Usage: make probe MODEL=<model_dir>
.PHONY: probe
probe: _require_venv
	@test -n "$(MODEL)" || (echo "  ✗  Usage: make probe MODEL=<dir>"; exit 1)
	$(PYTHON) $(SCRIPTS_DIR)/shatter_moe.py $(MODEL) /tmp/smoe_dry_run --dry-run

## test-quant: Quick smoke-test of the SMOE-Q2 quantiser on a tiny random tensor.
.PHONY: test-quant
test-quant: _require_venv
	$(PYTHON) -c "\
import numpy as np, sys; sys.path.insert(0,'$(SCRIPTS_DIR)'); \
from shatter_moe import quantize_smoeq2, dequantize_smoeq2, measure_q2_error; \
w = np.random.randn(128, 64).astype('float32'); \
p, s = quantize_smoeq2(w); \
err = measure_q2_error(w, p, s); \
print(f'  RMSE={err[\"rmse\"]:.5f}  absmax={err[\"absmax_err\"]:.5f}  SNR={err[\"snr_db\"]:.1f} dB'); \
print('  ✓  SMOE-Q2 smoke-test passed') \
"

# ─────────────────────────────────────────────────────────────
# C++ / METAL ENGINE
# ─────────────────────────────────────────────────────────────

## all: Build the C++ engine binary.
.PHONY: all
all: $(TARGET)
	@echo "  ⚡  smoe-engine → $(TARGET)"

$(TARGET): $(OBJS_CXX) $(OBJS_MM) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -c -o $@ $<

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.mm | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(MMFLAGS) -I$(SRC_DIR) -I$(BUILD_DIR) -c -o $@ $<

# kernels.metal is the single source of truth for the GPU kernels: the
# bridge JIT-compiles it at boot from this generated raw-string header.
$(BUILD_DIR)/compute/metal_bridge.o: $(MSL_HEADER)

$(MSL_HEADER): $(METAL_SRC) | $(BUILD_DIR)
	@printf '// Auto-generated from %s by make — do not edit.\n' $(METAL_SRC) >  $@
	@printf 'static const char* kMetalSource = R"SMOE_MSL(\n'                 >> $@
	@cat $(METAL_SRC)                                                         >> $@
	@printf '\n)SMOE_MSL";\n'                                                 >> $@
	@echo "  ⚙   MSL source embedded → $@"

$(METAL_LIB): $(METAL_SRC) | $(BUILD_DIR)
	$(XCRUN) -sdk macosx metal    -c $(METAL_SRC) -o $(BUILD_DIR)/kernels.air
	$(XCRUN) -sdk macosx metallib    $(BUILD_DIR)/kernels.air -o $@
	@echo "  ⚙   Metal kernels → $@"

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)/io $(BUILD_DIR)/scout $(BUILD_DIR)/compute

# ─────────────────────────────────────────────────────────────
# HOUSEKEEPING
# ─────────────────────────────────────────────────────────────

## clean: Remove the C++ build directory.
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	@echo "  🧹  Build artefacts cleaned."

## nuke: Remove both the build directory and the .venv.
.PHONY: nuke
nuke: clean
	rm -rf $(VENV_DIR)
	@echo "  💣  .venv removed. Run 'make setup' to start fresh."

## rebuild: Clean + full rebuild.
.PHONY: rebuild
rebuild: clean all

## help: Print available targets.
.PHONY: help
help:
	@grep -E '^## ' Makefile | sed 's/## /  /'
