#include "ModelBuilder.hpp"

#include <iostream>
#include <filesystem>

int main()
{
    KataGoCoreML::ModelDesc modelDesc;

    modelDesc.modelVersion = 3;
    modelDesc.numPolicyChannels = 1;
    modelDesc.numValueChannels = 3;
    modelDesc.numScoreValueChannels = 1;
    modelDesc.numOwnershipChannels = 1;

    const int nnXLen = 19;
    const int nnYLen = 19;

    KataGoCoreML::ModelBuilder builder(modelDesc, nnXLen, nnYLen);

    // Input spatial feature
    const int batchSize = 1;
    const int numSpatialFeatures = 22;

    KataGoCoreML::InputFeature inputSpatial("input_spatial",
                                            {batchSize, numSpatialFeatures, nnYLen, nnXLen});

    builder.addInputFeature(inputSpatial);

    // Input global feature
    const int numGlobalFeatures = 19;
    KataGoCoreML::InputFeature inputGlobal("input_global", {batchSize, numGlobalFeatures});
    builder.addInputFeature(inputGlobal);

    // Create a CoreML package with the model builder
    const std::string outputPath = "test_output.mlpackage";
    builder.createMLPackage(outputPath);

    if (!std::filesystem::exists(outputPath))
    {
        std::cerr << "❌ Output file was not created." << std::endl;
        return 1;
    }

    std::cout << "✅ Successfully built a CoreML package at " << outputPath << std::endl;
    return 0;
}
