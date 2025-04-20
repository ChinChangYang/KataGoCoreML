#include "ModelBuilder.hpp"

#include <fstream>
#include <memory>
#include <Model.pb.h>
#include "ModelVersion.hpp"

using namespace CoreML::Specification;
using namespace CoreML::Specification::MILSpec;

namespace katagocoreml
{
    ModelBuilder::ModelBuilder() = default;
    ModelBuilder::~ModelBuilder() = default;

    NamedValueType *addReLUOperation(Block &block, const NamedValueType &input, const char *name)
    {
        // Create a new ReLU operation.
        Operation *relu_op = block.add_operations();
        relu_op->set_type("relu");

        // Bind the input: set argument "x" to be the provided input's name.
        Argument::Binding *binding = (*relu_op->mutable_inputs())["x"].add_arguments();
        binding->set_name(input.name());

        // Create the operation's output "relu_out" and copy the input type.
        NamedValueType *relu_output = relu_op->add_outputs();
        relu_output->set_name(name);
        *relu_output->mutable_type() = input.type(); // Use same type as input

        return relu_output;
    }

    Program createProgram()
    {
        Program program;

        // Set version
        program.set_version(1);

        // Create Function
        Function func;

        // Define input tensor (FLOAT32, shape [1, 22, 19, 19])
        NamedValueType *input = func.add_inputs();
        input->set_name(INPUT_SPATIAL_NAME);

        TensorType *input_tensor = input->mutable_type()->mutable_tensortype();
        input_tensor->set_datatype(DataType::FLOAT32);
        input_tensor->set_rank(4);
        input_tensor->add_dimensions()->mutable_constant()->set_size(1);
        input_tensor->add_dimensions()->mutable_constant()->set_size(22);
        input_tensor->add_dimensions()->mutable_constant()->set_size(19);
        input_tensor->add_dimensions()->mutable_constant()->set_size(19);

        // Define Block (opset = "v1")
        func.set_opset("CoreML5");
        Block &block = (*func.mutable_block_specializations())["CoreML5"];

        // Call the helper function to add a ReLU operation to the block.
        NamedValueType *relu_output_0 = addReLUOperation(block, *input, "relu_output_0");
        NamedValueType *relu_output_1 = addReLUOperation(block, *relu_output_0, "relu_output_1");

        block.add_outputs("relu_output_1");

        // Add the function to the program
        (*program.mutable_functions())["main"] = func;

        return program;
    }

    // Populate model I/O
    void addModelIOFeatures(ModelDescription *desc, int batchSize, int nnXLen, int nnYLen, int modelVersion)
    {
        const auto dataType = ArrayFeatureType_ArrayDataType_FLOAT32;

        // Input Spatial
        {
            auto *feature = desc->add_input();
            feature->set_name(INPUT_SPATIAL_NAME);
            auto *array = feature->mutable_type()->mutable_multiarraytype();
            int numSpatial = getNumSpatialFeatures(modelVersion);
            assert(numSpatial > 0);
            array->add_shape(batchSize);
            array->add_shape(numSpatial);
            array->add_shape(nnYLen);
            array->add_shape(nnXLen);
            array->set_datatype(dataType);
        }

        // Input Global
        {
            auto *feature = desc->add_input();
            feature->set_name(INPUT_GLOBAL_NAME);
            auto *array = feature->mutable_type()->mutable_multiarraytype();
            int numGlobal = getNumGlobalFeatures(modelVersion);
            assert(numGlobal > 0);
            array->add_shape(batchSize);
            array->add_shape(numGlobal);
            array->set_datatype(dataType);
        }

        // Output Policy
        {
            auto *feature = desc->add_output();
            feature->set_name(OUTPUT_POLICY_NAME);
            auto *array = feature->mutable_type()->mutable_multiarraytype();
            int numPolicy = getNumPolicyChannel(modelVersion);
            array->add_shape(batchSize);
            array->add_shape(numPolicy);
            array->add_shape(nnYLen);
            array->add_shape(nnXLen);
            array->set_datatype(dataType);
        }

        // Output Policy Pass
        {
            auto *feature = desc->add_output();
            feature->set_name(OUTPUT_POLICY_PASS_NAME);
            auto *array = feature->mutable_type()->mutable_multiarraytype();
            array->add_shape(batchSize);
            array->add_shape(getNumPolicyChannel(modelVersion));
            array->set_datatype(dataType);
        }

        // Output Value
        {
            auto *feature = desc->add_output();
            feature->set_name(OUTPUT_VALUE_NAME);
            auto *array = feature->mutable_type()->mutable_multiarraytype();
            array->add_shape(batchSize);
            array->add_shape(3); // channels
            array->set_datatype(dataType);
        }

        // Output Score Value
        {
            auto *feature = desc->add_output();
            feature->set_name(OUTPUT_SCORE_VALUE_NAME);
            auto *array = feature->mutable_type()->mutable_multiarraytype();
            array->add_shape(batchSize);
            array->add_shape(getNumScoreValueChannel(modelVersion));
            array->set_datatype(dataType);
        }

        // Output Ownership
        {
            auto *feature = desc->add_output();
            feature->set_name(OUTPUT_OWNERSHIP_NAME);
            auto *array = feature->mutable_type()->mutable_multiarraytype();
            array->add_shape(batchSize);
            array->add_shape(1);
            array->add_shape(nnYLen);
            array->add_shape(nnXLen);
            array->set_datatype(dataType);
        }
    }

    bool ModelBuilder::buildMinimalModel(const std::string &outputPath)
    {
        const int batchSize = 1;
        const int nnXLen = 19;
        const int nnYLen = 19;
        const int modelVersion = 3;

        Model model;
        model.set_specificationversion(6);

        ModelDescription *desc = model.mutable_description();
        addModelIOFeatures(desc, batchSize, nnXLen, nnYLen, modelVersion);

        Program *program = new Program(createProgram());
        model.set_allocated_mlprogram(program);

        std::ofstream ofs(outputPath, std::ios::binary);
        if (!ofs)
        {
            return false;
        }
        return model.SerializeToOstream(&ofs);
    }

} // namespace katagocoreml
