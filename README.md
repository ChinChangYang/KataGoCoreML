# KataGoCoreML
Provides a C++ interface to convert a KataGo model into a CoreML model using Apple’s native CoreML libraries.

---

## 🚧 Status

This repository is a **work in progress**. Currently focused on:
- Building convolution operations

---

## 🔧 Build Instructions

1. **Configure and build the C++ interface**
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

2. **Run tests**
   ```bash
   ctest
   ```

   Expected output:
   ```
   100% tests passed, 0 tests failed out of 1
   ```

## 📦 Installation

To install this library, run:
```bash
make install
```

To use this in your KataGo project, add the following lines to your `CMakeLists.txt`:

```
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(KataGoCoreML REQUIRED)

target_link_libraries(katago KataGoCoreML ${Python3_LIBRARIES})
```

## 📜 License

This project is licensed under the **MIT License**. See `LICENSE` for details.
