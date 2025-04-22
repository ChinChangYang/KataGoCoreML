#pragma once

#include <string>

namespace KataGoCoreML
{
    const std::string INPUT_SPATIAL_NAME = "input_spatial";
    const std::string INPUT_GLOBAL_NAME = "input_global";
    const std::string INPUT_META_NAME = "input_meta";
    const std::string OUTPUT_POLICY_NAME = "output_policy";
    const std::string OUTPUT_POLICY_PASS_NAME = "output_policy_pass";
    const std::string OUTPUT_VALUE_NAME = "output_value";
    const std::string OUTPUT_SCORE_VALUE_NAME = "output_score_value";
    const std::string OUTPUT_OWNERSHIP_NAME = "output_ownership";

    class ModelBuilder
    {
    public:
        ModelBuilder();
        ~ModelBuilder();

        void createMLPackage(const std::string &packagePath);
    };

} // namespace KataGoCoreML
