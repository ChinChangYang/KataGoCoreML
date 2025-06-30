#include "ModelBuilder.hpp"

#include <iostream>
#include <filesystem>

int main()
{
    KataGoCoreML::ModelDesc modelDesc;

    modelDesc.numInputChannels = 22;
    modelDesc.numInputGlobalChannels = 19;
    modelDesc.modelVersion = 3;
    modelDesc.numPolicyChannels = 1;
    modelDesc.numValueChannels = 3;
    modelDesc.numScoreValueChannels = 1;
    modelDesc.numOwnershipChannels = 1;

    const int nnXLen = 19;
    const int nnYLen = 19;

    KataGoCoreML::ModelBuilder builder(modelDesc, nnXLen, nnYLen);

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
