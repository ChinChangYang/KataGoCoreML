#pragma once

#include <string>
#include <vector>
#include "UtilTempDir.hpp"
#include "ModelDescription.hpp"

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

    class InputFeature
    {
    public:
        std::string name;
        std::vector<int> shape;

        InputFeature(const std::string &name, const std::vector<int> &shape)
            : name(name), shape(shape) {}
    };

    class ModelBuilder
    {
    public:
        ModelBuilder(ModelDesc &modelDesc, int nnXLen, int nnYLen, int batchSize = 1)
            : modelDesc(modelDesc),
              nnXLen(nnXLen),
              nnYLen(nnYLen),
              batchSize(batchSize) {}

        void addInputFeature(InputFeature &inputFeature);
        void createMLPackage(const std::string &packagePath);

        const std::vector<InputFeature> &getInputFeatures() const
        {
            return inputFeatures;
        }

        ModelDesc &getModelDesc() const
        {
            return modelDesc;
        }

        int getNnXLen() const
        {
            return nnXLen;
        }

        int getNnYLen() const
        {
            return nnYLen;
        }

        int getBatchSize() const
        {
            return batchSize;
        }

    private:
        std::vector<InputFeature> inputFeatures;
        std::string packagePath;
        ModelDesc &modelDesc;
        int nnXLen;
        int nnYLen;
        int batchSize;

        std::string setupAndSerializeModel(const std::string &weightFile);
    };

} // namespace KataGoCoreML
