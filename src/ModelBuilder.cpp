#include "ModelBuilder.hpp"

#include <fstream>
#include <memory>
#include <Model.pb.h>

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
        input->set_name("input_spatial");

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

    bool ModelBuilder::buildMinimalModel(const std::string &outputPath)
    {
        Model model;

        model.set_specificationversion(6);

        ModelDescription *desc = model.mutable_description();
        FeatureDescription *inputFeature = desc->add_input();

        inputFeature->set_name("input_spatial");

        CoreML::Specification::ArrayFeatureType *inputArray = inputFeature->mutable_type()->mutable_multiarraytype();

        inputArray->add_shape(1);
        inputArray->add_shape(22);
        inputArray->add_shape(19);
        inputArray->add_shape(19);
        inputArray->set_datatype(CoreML::Specification::ArrayFeatureType_ArrayDataType_FLOAT32);

        FeatureDescription *outputFeature = desc->add_output();

        outputFeature->set_name("relu_out");

        CoreML::Specification::ArrayFeatureType *outputArray = outputFeature->mutable_type()->mutable_multiarraytype();

        outputArray->add_shape(1);
        outputArray->add_shape(22);
        outputArray->add_shape(19);
        outputArray->add_shape(19);
        outputArray->set_datatype(CoreML::Specification::ArrayFeatureType_ArrayDataType_FLOAT32);

        Program *program = new Program(createProgram());

        model.set_allocated_mlprogram(program);

        // === Write to file ===
        std::ofstream ofs(outputPath, std::ios::binary);
        if (!ofs)
        {
            return false;
        }

        return model.SerializeToOstream(&ofs);
    }

} // namespace katagocoreml
