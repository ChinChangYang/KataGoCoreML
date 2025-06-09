#include "ModelBuilder.hpp"

#include <iostream>
#include <filesystem>

int main()
{
    KataGoCoreML::ModelBuilder builder;
 
    // Input spatial feature
    const int batchSize = 1;
    const int numSpatialFeatures = 22;
    const int nnXLen = 19;
    const int nnYLen = 19;
 
    KataGoCoreML::InputFeature inputSpatial("input_spatial",
                                            {batchSize, numSpatialFeatures, nnYLen, nnXLen});

    builder.addInputFeature(inputSpatial);

    // Input global feature
    const int numGlobalFeatures = 19;
    KataGoCoreML::InputFeature inputGlobal("input_global", {1, numGlobalFeatures});
    builder.addInputFeature(inputGlobal);

    // Create a CoreML package with the model builder
    const std::string outputPath = "test_output.mlpackage";
    builder.createMLPackage(outputPath);

    if (!std::filesystem::exists(outputPath))
    {
        std::cerr << "❌ Output file was not created." << std::endl;
        return 1;
    }

    std::cout << "✅ Successfully built a CoreML model at " << outputPath << std::endl;
    return 0;
}
