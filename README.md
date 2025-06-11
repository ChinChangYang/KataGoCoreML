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

## 🛠️ Usage in KataGo Project

1. **CMake Configuration**

Add the following lines to your `CMakeLists.txt`:

```cmake
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(KataGoCoreML REQUIRED)

target_link_libraries(katago KataGoCoreML ${Python3_LIBRARIES})
```

2. **Model Conversion Workflow**

In your C++ code, perform the following steps:

* Convert KataGo’s `ModelDesc` object into a `KataGoCoreML::ModelDesc` object.
* Use this to construct a `KataGoCoreML::ModelBuilder` object.
* Call the `createMLPackage(outputPath)` member function to generate a CoreML model package.

See `test/test_main.cpp` for an example.

## 📜 License

This project is licensed under the **MIT License**. See `LICENSE` for details.
