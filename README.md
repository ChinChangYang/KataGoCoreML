# KataGoCoreML
Provides a C++ interface to convert a KataGo model into a CoreML model using Apple’s native CoreML libraries.

---

## 🚧 Status

This repository is a **work in progress**. Currently focused on:
- Building convolution operations

---

## 🔧 Build Instructions

1. **Build the native CoreML libraries**

   Clone and build CoreMLTools using the included script:
   ```bash
   ./scripts/build_coremltools.sh
   ```

2. **Configure and build the C++ interface**
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

3. **Run tests**
   ```bash
   ./katagocoreml_tests
   ```

   Expected output:
   ```
   ✅ Successfully built minimal CoreML model at test_output.mlpackage
   ```

## 📜 License

This project is licensed under the **MIT License**. See `LICENSE` for details.
