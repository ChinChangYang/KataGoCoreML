// The definitions in this file shall be consistent with coremltools/coremltools/__init__.py

#pragma once

// Core ML specification versions

// Basic Core ML specification understood by iOS 11.0
#define SPECIFICATION_VERSION 1

// iOS 11.2 features
#define MINIMUM_CUSTOM_LAYER_SPEC_VERSION 2
#define MINIMUM_FP16_SPEC_VERSION 2

// iOS 12.0 features
#define MINIMUM_CUSTOM_MODEL_SPEC_VERSION 3
#define MINIMUM_QUANTIZED_MODEL_SPEC_VERSION 3
#define MINIMUM_FLEXIBLE_SHAPES_SPEC_VERSION 3

// iOS 13.0 features
#define MINIMUM_NDARRAY_SPEC_VERSION 4
#define MINIMUM_NEAREST_NEIGHBORS_SPEC_VERSION 4
#define MINIMUM_LINKED_MODELS_SPEC_VERSION 4
#define MINIMUM_UPDATABLE_SPEC_VERSION 4
#define SPECIFICATION_VERSION_IOS_13 4

// iOS 14.0
#define SPECIFICATION_VERSION_IOS_14 5

// iOS 15.0
#define SPECIFICATION_VERSION_IOS_15 6

// iOS 16.0
#define SPECIFICATION_VERSION_IOS_16 7

// iOS 17.0
#define SPECIFICATION_VERSION_IOS_17 8

// iOS 18.0
#define SPECIFICATION_VERSION_IOS_18 9

// Backend-specific minimum supported specification versions
#define LOWEST_ALLOWED_SPECIFICATION_VERSION_FOR_NEURALNETWORK SPECIFICATION_VERSION_IOS_13
#define LOWEST_ALLOWED_SPECIFICATION_VERSION_FOR_MILPROGRAM SPECIFICATION_VERSION_IOS_15

// Compute unit configuration options
enum ComputeUnit {
    COMPUTE_UNIT_ALL = 1,          // Use CPU, GPU, and Neural Engine
    COMPUTE_UNIT_CPU_AND_GPU = 2,  // Use CPU and GPU only
    COMPUTE_UNIT_CPU_ONLY = 3,     // Use CPU only
    COMPUTE_UNIT_CPU_AND_NE = 4    // Use CPU and Neural Engine only (macOS >= 13.0)
};

// Model reshape frequency hint
enum ReshapeFrequency {
    RESHAPE_FREQUENCY_FREQUENT = 1,
    RESHAPE_FREQUENCY_INFREQUENT = 2
};

// Specialization strategy for model optimization
enum SpecializationStrategy {
    SPECIALIZATION_STRATEGY_DEFAULT = 1,
    SPECIALIZATION_STRATEGY_FAST_PREDICTION = 2
};

// Mapping from specification version to MIL opset string (defined elsewhere as needed)
#define OPSET_SPECIFICATION_VERSION_IOS_13 "CoreML3"
#define OPSET_SPECIFICATION_VERSION_IOS_14 "CoreML4"
#define OPSET_SPECIFICATION_VERSION_IOS_15 "CoreML5"
#define OPSET_SPECIFICATION_VERSION_IOS_16 "CoreML6"
#define OPSET_SPECIFICATION_VERSION_IOS_17 "CoreML7"
#define OPSET_SPECIFICATION_VERSION_IOS_18 "CoreML8"
