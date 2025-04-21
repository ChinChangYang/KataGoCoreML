#include "ModelBuilder.hpp"

#include <iostream>
#include <filesystem>

int main()
{
    KataGoCoreML::ModelBuilder builder;

    const std::string outputPath = "test_output.mlpackage";
    builder.createMLPackage("", outputPath);

    if (!std::filesystem::exists(outputPath))
    {
        std::cerr << "❌ Output file was not created." << std::endl;
        return 1;
    }

    std::cout << "✅ Successfully built minimal CoreML model at " << outputPath << std::endl;
    return 0;
}
