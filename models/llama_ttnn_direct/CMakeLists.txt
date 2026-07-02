# ===- CMakeLists.txt - TTNN Direct Llama package -----------*- cmake -*-===//
#
# Phase 13 adds a lightweight package-directory target for the additive
# Buddy-TTNN Direct Python program path. It intentionally does not register a
# buddy-cli runner and does not modify existing flatbuffer package targets.

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(BUDDY_LLAMA31_TTNN_DIRECT_MODEL_PATH "" CACHE PATH
  "Optional local Hugging Face Llama 3.1 model path for TTNN Direct packaging")
set(BUDDY_LLAMA31_TTNN_DIRECT_CONFIG
  "${CMAKE_CURRENT_SOURCE_DIR}/buddy_ttnn_direct/configs/p150a_llama31_8b_b32.json"
  CACHE FILEPATH "Template config JSON for the TTNN Direct Llama 3.1 package")

set(_BUDDY_LLAMA31_TTNN_DIRECT_ROOT
  "${CMAKE_BINARY_DIR}/models/llama31_ttnn_direct")
set(_BUDDY_LLAMA31_TTNN_DIRECT_FAKE_MODEL_DIR
  "${_BUDDY_LLAMA31_TTNN_DIRECT_ROOT}/config_only_model")
set(_BUDDY_LLAMA31_TTNN_DIRECT_PROGRAM_DIR
  "${_BUDDY_LLAMA31_TTNN_DIRECT_ROOT}/program")
set(_BUDDY_LLAMA31_TTNN_DIRECT_PACKAGE_DIR
  "${_BUDDY_LLAMA31_TTNN_DIRECT_ROOT}/llama31_ttnn_direct_package")
set(_BUDDY_LLAMA31_TTNN_DIRECT_MANIFEST
  "${_BUDDY_LLAMA31_TTNN_DIRECT_PACKAGE_DIR}/manifest.json")

file(MAKE_DIRECTORY "${_BUDDY_LLAMA31_TTNN_DIRECT_FAKE_MODEL_DIR}")
file(WRITE
  "${_BUDDY_LLAMA31_TTNN_DIRECT_FAKE_MODEL_DIR}/config.json"
  "{\n"
  "  \"_name_or_path\": \"meta-llama/Llama-3.1-8B-Instruct\",\n"
  "  \"model_type\": \"llama\",\n"
  "  \"num_hidden_layers\": 32,\n"
  "  \"hidden_size\": 4096,\n"
  "  \"intermediate_size\": 14336,\n"
  "  \"num_attention_heads\": 32,\n"
  "  \"num_key_value_heads\": 8,\n"
  "  \"vocab_size\": 128256,\n"
  "  \"rms_norm_eps\": 0.00001,\n"
  "  \"rope_theta\": 500000.0,\n"
  "  \"tie_word_embeddings\": false\n"
  "}\n")

set(_BUDDY_LLAMA31_TTNN_DIRECT_MODEL_INPUT
  "${_BUDDY_LLAMA31_TTNN_DIRECT_FAKE_MODEL_DIR}")
if(BUDDY_LLAMA31_TTNN_DIRECT_MODEL_PATH)
  set(_BUDDY_LLAMA31_TTNN_DIRECT_MODEL_INPUT
    "${BUDDY_LLAMA31_TTNN_DIRECT_MODEL_PATH}")
endif()

add_custom_command(
  OUTPUT "${_BUDDY_LLAMA31_TTNN_DIRECT_MANIFEST}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${_BUDDY_LLAMA31_TTNN_DIRECT_ROOT}"
  COMMAND "${Python3_EXECUTABLE}" -m
          models.llama_ttnn_direct.buddy_ttnn_direct.cli
          build-program
          --model-path "${_BUDDY_LLAMA31_TTNN_DIRECT_MODEL_INPUT}"
          --config "${BUDDY_LLAMA31_TTNN_DIRECT_CONFIG}"
          --out-dir "${_BUDDY_LLAMA31_TTNN_DIRECT_PROGRAM_DIR}"
  COMMAND "${Python3_EXECUTABLE}" -m
          models.llama_ttnn_direct.buddy_ttnn_direct.cli
          package-program
          --program-dir "${_BUDDY_LLAMA31_TTNN_DIRECT_PROGRAM_DIR}"
          --out-dir "${_BUDDY_LLAMA31_TTNN_DIRECT_PACKAGE_DIR}"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/buddy_ttnn_direct/cli.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/buddy_ttnn_direct/codegen/package.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/buddy_ttnn_direct/codegen/program.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/buddy_ttnn_direct/codegen/python_ttnn.py"
    "${BUDDY_LLAMA31_TTNN_DIRECT_CONFIG}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Generating TTNN Direct Llama 3.1 Python program package"
  VERBATIM
)

add_custom_target(llama31_ttnn_direct_program
  DEPENDS "${_BUDDY_LLAMA31_TTNN_DIRECT_MANIFEST}"
  COMMENT "TTNN Direct package -> ${_BUDDY_LLAMA31_TTNN_DIRECT_PACKAGE_DIR}"
)
