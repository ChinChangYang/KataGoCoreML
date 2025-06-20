#include "ModelBuilder.hpp"

#include <fstream>
#include <memory>
#include <MILBlob/Blob/StorageWriter.hpp>
#include <Model.pb.h>
#include <ModelPackage.hpp>
#include "ModelVersion.hpp"
#include "UtilTempDir.hpp"
#include "CoremltoolsDefines.hpp"

using namespace MILBlob;
using namespace CoreML::Specification;
using namespace CoreML::Specification::MILSpec;
using namespace MPL;
namespace fs = std::filesystem;

namespace KataGoCoreML
{
    NamedValueType *addReLUOperation(Block &block, const NamedValueType &input, const char *name)
    {
        // Create a new ReLU operation.
        Operation *relu_op = block.add_operations();
        relu_op->set_type("relu");

        // Bind the input: set argument "x" to be the provided input's name.
        auto *binding = (*relu_op->mutable_inputs())["x"].add_arguments();
        binding->set_name(input.name());

        // Create the operation's output "relu_out" and copy the input type.
        NamedValueType *relu_output = relu_op->add_outputs();
        relu_output->set_name(name);
        *relu_output->mutable_type() = input.type(); // Use same type as input

        return relu_output;
    }

    NamedValueType *addConvOperation(Block &block,
                                     const NamedValueType &input,
                                     const int numOutputChannel,
                                     const int numInputChannel,
                                     std::string name,
                                     Blob::StorageWriter &weightWriter)
    {
        const int kernelSize = 3;
        const std::string weightName = name + "_weight";

        // === Constant operation for weight ===
        Operation *constWeightOp = block.add_operations();
        constWeightOp->set_type("const");
        auto *output_weight = constWeightOp->add_outputs();
        output_weight->set_name(weightName);
        auto *weight_output_tensor_type = output_weight->mutable_type()->mutable_tensortype();
        weight_output_tensor_type->set_datatype(DataType::FLOAT32);
        weight_output_tensor_type->set_rank(4);
        weight_output_tensor_type->add_dimensions()->mutable_constant()->set_size(numOutputChannel);
        weight_output_tensor_type->add_dimensions()->mutable_constant()->set_size(numInputChannel);
        weight_output_tensor_type->add_dimensions()->mutable_constant()->set_size(kernelSize);
        weight_output_tensor_type->add_dimensions()->mutable_constant()->set_size(kernelSize);

        Value &weight_attribute_val = (*constWeightOp->mutable_attributes())["val"];
        auto *weight_attribute_tensor_type = weight_attribute_val.mutable_type()->mutable_tensortype();
        weight_attribute_tensor_type->set_datatype(DataType::FLOAT32);
        weight_attribute_tensor_type->set_rank(4);
        weight_attribute_tensor_type->add_dimensions()->mutable_constant()->set_size(numOutputChannel);
        weight_attribute_tensor_type->add_dimensions()->mutable_constant()->set_size(numInputChannel);
        weight_attribute_tensor_type->add_dimensions()->mutable_constant()->set_size(kernelSize);
        weight_attribute_tensor_type->add_dimensions()->mutable_constant()->set_size(kernelSize);

        const std::vector<float> weightData(numOutputChannel * numInputChannel * kernelSize * kernelSize, 0.0f);
        auto span = Util::MakeSpan(weightData);
        auto offset = weightWriter.WriteData(span);

        auto *weight_attribute_val_blobfile = weight_attribute_val.mutable_blobfilevalue();
        weight_attribute_val_blobfile->set_filename("@model_path/weights/weight.bin");
        weight_attribute_val_blobfile->set_offset(offset);

        Value &weight_attribute_name = (*constWeightOp->mutable_attributes())["name"];
        weight_attribute_name.mutable_type()
            ->mutable_tensortype()
            ->set_datatype(DataType::STRING);
        weight_attribute_name.mutable_immediatevalue()
            ->mutable_tensor()
            ->mutable_strings()
            ->add_values(weightName);

        // === Constant operation for padding type ===
        Operation *constPadTypeOp = block.add_operations();
        constPadTypeOp->set_type("const");
        auto *output_pad_type = constPadTypeOp->add_outputs();
        const std::string padTypeName = name + "_pad_type";
        output_pad_type->set_name(padTypeName);
        auto *pad_type_output_tensor_type = output_pad_type->mutable_type()->mutable_tensortype();
        pad_type_output_tensor_type->set_datatype(DataType::STRING);

        auto &pad_type_attribute_val = (*constPadTypeOp->mutable_attributes())["val"];
        auto *pad_type_attribute_tensor_type = pad_type_attribute_val.mutable_type()->mutable_tensortype();
        pad_type_attribute_tensor_type->set_datatype(DataType::STRING);
        pad_type_attribute_val.mutable_immediatevalue()
            ->mutable_tensor()
            ->mutable_strings()
            ->add_values("same");

        auto &pad_type_attribute_name = (*constPadTypeOp->mutable_attributes())["name"];
        pad_type_attribute_name.mutable_type()
            ->mutable_tensortype()
            ->set_datatype(DataType::STRING);
        pad_type_attribute_name.mutable_immediatevalue()
            ->mutable_tensor()
            ->mutable_strings()
            ->add_values(padTypeName);

        // === Constant operation for strides ===
        Operation *constStridesOp = block.add_operations();
        constStridesOp->set_type("const");
        auto *output_strides = constStridesOp->add_outputs();
        const std::string stridesName = name + "_strides";
        output_strides->set_name(stridesName);
        auto *strides_output_tensor_type = output_strides->mutable_type()->mutable_tensortype();
        strides_output_tensor_type->set_datatype(DataType::INT32);
        strides_output_tensor_type->set_rank(1);
        strides_output_tensor_type->add_dimensions()->mutable_constant()->set_size(2);

        auto &strides_attribute_val = (*constStridesOp->mutable_attributes())["val"];
        auto *strides_attribute_tensor_type = strides_attribute_val.mutable_type()->mutable_tensortype();
        strides_attribute_tensor_type->set_datatype(DataType::INT32);
        strides_attribute_tensor_type->set_rank(1);
        strides_attribute_tensor_type->add_dimensions()
            ->mutable_constant()
            ->set_size(2);
        auto *strides_attributes_val_ints = strides_attribute_val.mutable_immediatevalue()
                                                ->mutable_tensor()
                                                ->mutable_ints();

        strides_attributes_val_ints->add_values(1);
        strides_attributes_val_ints->add_values(1);

        auto &strides_attribute_name = (*constStridesOp->mutable_attributes())["name"];
        strides_attribute_name.mutable_type()
            ->mutable_tensortype()
            ->set_datatype(DataType::STRING);

        strides_attribute_name.mutable_immediatevalue()
            ->mutable_tensor()
            ->mutable_strings()
            ->add_values(stridesName);

        // === Constant operation for pad ===
        Operation *constPadOp = block.add_operations();
        constPadOp->set_type("const");
        auto *output_pad = constPadOp->add_outputs();
        const std::string padName = name + "_pad";
        output_pad->set_name(padName);
        auto *pad_output_tensor_type = output_pad->mutable_type()->mutable_tensortype();
        pad_output_tensor_type->set_datatype(DataType::INT32);
        pad_output_tensor_type->set_rank(1);
        pad_output_tensor_type->add_dimensions()
            ->mutable_constant()
            ->set_size(4);

        auto &pad_attribute_val = (*constPadOp->mutable_attributes())["val"];
        auto *pad_attribute_tensor_type = pad_attribute_val.mutable_type()->mutable_tensortype();
        pad_attribute_tensor_type->set_datatype(DataType::INT32);
        pad_attribute_tensor_type->set_rank(1);
        pad_attribute_tensor_type->add_dimensions()
            ->mutable_constant()
            ->set_size(4);

        auto *pad_attributes_val_ints = pad_attribute_val.mutable_immediatevalue()
                                            ->mutable_tensor()
                                            ->mutable_ints();
        pad_attributes_val_ints->add_values(0);
        pad_attributes_val_ints->add_values(0);
        pad_attributes_val_ints->add_values(0);
        pad_attributes_val_ints->add_values(0);

        auto &pad_attribute_name = (*constPadOp->mutable_attributes())["name"];
        pad_attribute_name.mutable_type()
            ->mutable_tensortype()
            ->set_datatype(DataType::STRING);

        pad_attribute_name.mutable_immediatevalue()
            ->mutable_tensor()
            ->mutable_strings()
            ->add_values(padName);

        // === Constant operation for dilations ===
        Operation *constDilationsOp = block.add_operations();
        constDilationsOp->set_type("const");
        auto *output_dilations = constDilationsOp->add_outputs();
        const std::string dilationsName = name + "_dilations";
        output_dilations->set_name(dilationsName);
        auto *dilations_output_tensor_type = output_dilations->mutable_type()->mutable_tensortype();
        dilations_output_tensor_type->set_datatype(DataType::INT32);
        dilations_output_tensor_type->set_rank(1);
        dilations_output_tensor_type->add_dimensions()
            ->mutable_constant()
            ->set_size(2);

        auto &dilations_attribute_val = (*constDilationsOp->mutable_attributes())["val"];
        auto *dilations_attribute_tensor_type = dilations_attribute_val.mutable_type()->mutable_tensortype();
        dilations_attribute_tensor_type->set_datatype(DataType::INT32);
        dilations_attribute_tensor_type->set_rank(1);
        dilations_attribute_tensor_type->add_dimensions()
            ->mutable_constant()
            ->set_size(2);

        auto *dilations_attributes_val_ints = dilations_attribute_val.mutable_immediatevalue()
                                                  ->mutable_tensor()
                                                  ->mutable_ints();

        dilations_attributes_val_ints->add_values(1);
        dilations_attributes_val_ints->add_values(1);

        auto &dilations_attribute_name = (*constDilationsOp->mutable_attributes())["name"];

        dilations_attribute_name.mutable_type()
            ->mutable_tensortype()
            ->set_datatype(DataType::STRING);

        dilations_attribute_name.mutable_immediatevalue()
            ->mutable_tensor()
            ->mutable_strings()
            ->add_values(dilationsName);

        // === Constant operation for groups ===
        Operation *constGroupsOp = block.add_operations();
        constGroupsOp->set_type("const");
        auto *output_groups = constGroupsOp->add_outputs();
        const std::string groupsName = name + "_groups";
        output_groups->set_name(groupsName);
        auto *groups_output_tensor_type = output_groups->mutable_type()->mutable_tensortype();
        groups_output_tensor_type->set_datatype(DataType::INT32);

        auto &groups_attribute_val = (*constGroupsOp->mutable_attributes())["val"];
        auto *groups_attribute_tensor_type = groups_attribute_val.mutable_type()->mutable_tensortype();
        groups_attribute_tensor_type->set_datatype(DataType::INT32);

        auto *groups_attributes_val_ints = groups_attribute_val.mutable_immediatevalue()
                                               ->mutable_tensor()
                                               ->mutable_ints();

        groups_attributes_val_ints->add_values(1);

        auto &groups_attribute_name = (*constGroupsOp->mutable_attributes())["name"];
        groups_attribute_name.mutable_type()
            ->mutable_tensortype()
            ->set_datatype(DataType::STRING);

        groups_attribute_name.mutable_immediatevalue()
            ->mutable_tensor()
            ->mutable_strings()
            ->add_values(groupsName);

        // === Convolution operation ===
        Operation *convOp = block.add_operations();
        convOp->set_type("conv");

        // === Inputs ===
        // x (the input tensor)
        auto *input_x = (*convOp->mutable_inputs())["x"].add_arguments();
        input_x->set_name(input.name());

        // weight
        auto *input_weight = (*convOp->mutable_inputs())["weight"].add_arguments();
        input_weight->set_name(weightName);

        // strides
        auto *input_strides = (*convOp->mutable_inputs())["strides"].add_arguments();
        input_strides->set_name(stridesName);

        // pad
        auto *input_pad = (*convOp->mutable_inputs())["pad"].add_arguments();
        input_pad->set_name(padName);

        // pad_type
        auto *input_pad_type = (*convOp->mutable_inputs())["pad_type"].add_arguments();
        input_pad_type->set_name(padTypeName);

        // dilations
        auto *input_dilations = (*convOp->mutable_inputs())["dilations"].add_arguments();
        input_dilations->set_name(dilationsName);

        // groups
        auto *input_groups = (*convOp->mutable_inputs())["groups"].add_arguments();
        input_groups->set_name(groupsName);

        // === Outputs ===
        auto output = convOp->add_outputs();
        output->set_name(name);
        auto *outputTensor = output->mutable_type()->mutable_tensortype();
        outputTensor->set_datatype(DataType::FLOAT32);
        outputTensor->set_rank(4);
        const int batchSize = input.type().tensortype().dimensions(0).constant().size();
        const int nnYLen = input.type().tensortype().dimensions(2).constant().size();
        const int nnXLen = input.type().tensortype().dimensions(3).constant().size();
        outputTensor->add_dimensions()->mutable_constant()->set_size(batchSize);
        outputTensor->add_dimensions()->mutable_constant()->set_size(numOutputChannel);
        outputTensor->add_dimensions()->mutable_constant()->set_size(nnYLen);
        outputTensor->add_dimensions()->mutable_constant()->set_size(nnXLen);

        // === Attributes ===
        Value &attribute_name = (*convOp->mutable_attributes())["name"];
        attribute_name.mutable_type()
            ->mutable_tensortype()
            ->set_datatype(DataType::STRING);

        attribute_name.mutable_immediatevalue()
            ->mutable_tensor()
            ->mutable_strings()
            ->add_values(name);

        return output;
    }

    void setupProgram(ModelBuilder &mb, Program &program,
                      const std::string &weightsPath)
    {
        // Data type will be configurable, but for now we use float32
        const auto dataType = DataType::FLOAT32;

        // Version is set to a value that is consistent with coremltools
        program.set_version(1);

        // Create Function
        Function func;

        // For each input feature, add a input spatial tensor to the function
        for (const auto &inputFeature : mb.getInputFeatures())
        {
            auto *inputValue = func.add_inputs();
            inputValue->set_name(inputFeature.name);
            auto *inputTensor = inputValue->mutable_type()->mutable_tensortype();
            inputTensor->set_datatype(dataType);
            inputTensor->set_rank(inputFeature.shape.size());
            for (const auto &dim : inputFeature.shape)
            {
                inputTensor->add_dimensions()->mutable_constant()->set_size(dim);
            }
        }

        // Define a block for input, output, and operations
        func.set_opset(OPSET_SPECIFICATION_VERSION_IOS_15);
        Block &block = (*func.mutable_block_specializations())[OPSET_SPECIFICATION_VERSION_IOS_15];

        // The inputs(0) is the input spatial tensor
        NamedValueType inputSpatialValue = func.inputs(0);
        const int numSpatial = func.inputs(0).type().tensortype().dimensions(1).constant().size();

        // Create a writer for the weights
        Blob::StorageWriter weightWriter(weightsPath);

        NamedValueType *initial_conv = addConvOperation(
            block,
            inputSpatialValue,
            mb.getModelDesc().numPolicyChannels,
            numSpatial,
            OUTPUT_POLICY_NAME,
            weightWriter);

        block.add_outputs(initial_conv->name());

        // Add the function to the program
        (*program.mutable_functions())["main"] = func;
    }

    // Populate model I/O
    void addModelIOFeatures(ModelBuilder &mb, ModelDescription &desc)
    {
        // Data type will be configurable, but for now we use float32
        const auto dataType = ArrayFeatureType_ArrayDataType_FLOAT32;

        // For each input feature, add a feature to the model description
        for (const auto &inputFeature : mb.getInputFeatures())
        {
            auto *feature = desc.add_input();
            feature->set_name(inputFeature.name);
            auto *array = feature->mutable_type()->mutable_multiarraytype();
            for (const auto &dim : inputFeature.shape)
            {
                array->add_shape(dim);
            }
            array->set_datatype(dataType);
        }

        const int batchSize = mb.getBatchSize();
        const int nnXLen = mb.getNnXLen();
        const int nnYLen = mb.getNnYLen();
        const ModelDesc &modelDesc = mb.getModelDesc();
        const int numPolicy = modelDesc.numPolicyChannels;
        assert(numPolicy > 0);

        // Output Policy
        {
            auto *feature = desc.add_output();
            feature->set_name(OUTPUT_POLICY_NAME);
            auto *array = feature->mutable_type()->mutable_multiarraytype();
            array->add_shape(batchSize);
            array->add_shape(numPolicy);
            array->add_shape(nnYLen);
            array->add_shape(nnXLen);
            array->set_datatype(dataType);
        }

        // Output Policy Pass
        {
            auto *feature = desc.add_output();
            feature->set_name(OUTPUT_POLICY_PASS_NAME);
            auto *array = feature->mutable_type()->mutable_multiarraytype();
            array->add_shape(batchSize);
            array->add_shape(numPolicy);
            array->set_datatype(dataType);
        }

        // Output Value
        {
            auto *feature = desc.add_output();
            feature->set_name(OUTPUT_VALUE_NAME);
            auto *array = feature->mutable_type()->mutable_multiarraytype();
            const int numValue = modelDesc.numValueChannels;
            assert(numValue > 0);
            array->add_shape(batchSize);
            array->add_shape(numValue);
            array->set_datatype(dataType);
        }

        // Output Score Value
        {
            auto *feature = desc.add_output();
            feature->set_name(OUTPUT_SCORE_VALUE_NAME);
            auto *array = feature->mutable_type()->mutable_multiarraytype();
            const int numScoreValue = modelDesc.numScoreValueChannels;
            assert(numScoreValue > 0);
            array->add_shape(batchSize);
            array->add_shape(numScoreValue);
            array->set_datatype(dataType);
        }

        // Output Ownership
        {
            auto *feature = desc.add_output();
            feature->set_name(OUTPUT_OWNERSHIP_NAME);
            auto *array = feature->mutable_type()->mutable_multiarraytype();
            const int numOwnership = modelDesc.numOwnershipChannels;
            assert(numOwnership > 0);
            array->add_shape(batchSize);
            array->add_shape(numOwnership);
            array->add_shape(nnYLen);
            array->add_shape(nnXLen);
            array->set_datatype(dataType);
        }
    }

    void setupModel(ModelBuilder &mb, Model &model,
                    const std::string &weightsPath)
    {
        // Specification version is set to a value that is consistent with coremltools
        model.set_specificationversion(SPECIFICATION_VERSION_IOS_15);

        ModelDescription *desc = model.mutable_description();
        addModelIOFeatures(mb, *desc);

        Program *program = new Program();
        setupProgram(mb, *program, weightsPath);
        model.set_allocated_mlprogram(program);
    }

    std::string createTempFile(const std::string &templatePattern)
    {
        std::string path = templatePattern;
        int fd = mkstemp(const_cast<char *>(path.data()));
        if (fd < 0)
        {
            throw std::runtime_error("Failed to create temporary file: " + templatePattern);
        }

        std::cout << "Temporary file created: " << path << std::endl;
        return path;
    }

    std::string ModelBuilder::setupAndSerializeModel(const std::string &weightFile)
    {
        // Initialize and setup the model
        Model model;
        setupModel(*this, model, weightFile);

        // Serialize the model to a new temp file
        auto tmpPattern = (fs::temp_directory_path() / "modelXXXXXX").string();
        std::string modelPath = createTempFile(tmpPattern);
        std::ofstream ofs(modelPath, std::ios::binary);
        model.SerializeToOstream(&ofs);
        ofs.close();

        std::cout << "Model serialized to: " << modelPath << std::endl;
        return modelPath;
    }

    void cleanExistingPackage(const std::string &packagePath)
    {
        if (fs::exists(packagePath))
        {
            fs::remove_all(packagePath);
        }
    }

    void buildModelPackage(const std::string &packagePath,
                           const std::string &modelPath,
                           const TempDir &weightDir)
    {
        ModelPackage pkg(packagePath);
        pkg.setRootModel(modelPath,
                         "model.mlmodel",
                         "github.com/ChinChangYang/KataGoCoreML",
                         "KataGo CoreML Model Specification");
        pkg.addItem(weightDir.path(),
                    "weights",
                    "github.com/ChinChangYang/KataGoCoreML",
                    "KataGo CoreML Model Weights");
    }

    void ModelBuilder::addInputFeature(InputFeature &inputFeature)
    {
        inputFeatures.push_back(inputFeature);
    }

    void ModelBuilder::createMLPackage(const std::string &packagePath)
    {
        // Prepare a temp directory and weight file
        auto weightDir = TempDir("weights");
        auto weightFile = weightDir.path().string() + "/weight.bin";

        // Build and serialize the model
        auto modelFile = setupAndSerializeModel(weightFile);

        // Remove any existing package
        cleanExistingPackage(packagePath);

        // Assemble the final package
        buildModelPackage(packagePath, modelFile, weightDir);
    }

} // namespace KataGoCoreML
