#include "katagocoreml/ModelBuilder.hpp"

#include <iostream>
#include <filesystem>

int main()
{
    katagocoreml::ModelBuilder builder;

    const std::string outputPath = "test_output.mlmodel";
    if (!builder.buildMinimalModel(outputPath))
    {
        std::cerr << "❌ Failed to write minimal model to: " << outputPath << std::endl;
        return 1;
    }

    if (!std::filesystem::exists(outputPath))
    {
        std::cerr << "❌ Output file was not created." << std::endl;
        return 1;
    }

    std::cout << "✅ Successfully built minimal CoreML model at " << outputPath << std::endl;
    return 0;
}
