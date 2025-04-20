#pragma once

#include <string>

namespace katagocoreml
{

    class ModelBuilder
    {
    public:
        ModelBuilder();
        ~ModelBuilder();

        // Build a minimal MLModel and serialize it to .mlmodel file
        bool buildMinimalModel(const std::string &outputPath);
    };

} // namespace katagocoreml
