#include "ModelBuilder.hpp"

#include <iostream>
#include <filesystem>

int main()
{
    const int batchSize = 1;
    const int numSpatialFeatures = 22;
    const int nnXLen = 19;
    const int nnYLen = 19;

    KataGoCoreML::ModelBuilder builder;
    KataGoCoreML::InputFeature inputSpatial("input_spatial",
                                            {batchSize, numSpatialFeatures, nnYLen, nnXLen});

    builder.addInputFeature(inputSpatial);
    const std::string outputPath = "test_output.mlpackage";
    builder.createMLPackage(outputPath);

    if (!std::filesystem::exists(outputPath))
    {
        std::cerr << "❌ Output file was not created." << std::endl;
        return 1;
    }

    std::cout << "✅ Successfully built minimal CoreML model at " << outputPath << std::endl;
    return 0;
}
