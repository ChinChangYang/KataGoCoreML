#pragma once

// Model versions
namespace KataGoCoreML
{
    const int NUM_FEATURES_SPATIAL_V3 = 22;
    const int NUM_FEATURES_GLOBAL_V3 = 14;

    const int NUM_FEATURES_SPATIAL_V4 = 22;
    const int NUM_FEATURES_GLOBAL_V4 = 14;

    const int NUM_FEATURES_SPATIAL_V5 = 13;
    const int NUM_FEATURES_GLOBAL_V5 = 12;

    const int NUM_FEATURES_SPATIAL_V6 = 22;
    const int NUM_FEATURES_GLOBAL_V6 = 16;

    const int NUM_FEATURES_SPATIAL_V7 = 22;
    const int NUM_FEATURES_GLOBAL_V7 = 19;

    constexpr int latestModelVersionImplemented = 16;
    constexpr int latestInputsVersionImplemented = 7;
    constexpr int defaultModelVersion = 16;

    constexpr int oldestModelVersionImplemented = 3;
    constexpr int oldestInputsVersionImplemented = 3;

    // Which V* feature version from NNInputs does a given model version consume?
    int getInputsVersion(int modelVersion)
    {
        if (modelVersion >= 8 && modelVersion <= 16)
            return 7;
        else if (modelVersion == 7)
            return 6;
        else if (modelVersion == 6)
            return 5;
        else if (modelVersion == 5)
            return 4;
        else if (modelVersion == 3 || modelVersion == 4)
            return 3;
        return -1;
    }

    // Convenience functions, feeds forward the number of features and the size of
    // the row vector that the net takes as input
    int getNumSpatialFeatures(int modelVersion)
    {
        if (modelVersion >= 8 && modelVersion <= 16)
            return NUM_FEATURES_SPATIAL_V7;
        else if (modelVersion == 7)
            return NUM_FEATURES_SPATIAL_V6;
        else if (modelVersion == 6)
            return NUM_FEATURES_SPATIAL_V5;
        else if (modelVersion == 5)
            return NUM_FEATURES_SPATIAL_V4;
        else if (modelVersion == 3 || modelVersion == 4)
            return NUM_FEATURES_SPATIAL_V3;
        return -1;
    }

    int getNumGlobalFeatures(int modelVersion)
    {
        if (modelVersion >= 8 && modelVersion <= 16)
            return NUM_FEATURES_GLOBAL_V7;
        else if (modelVersion == 7)
            return NUM_FEATURES_GLOBAL_V6;
        else if (modelVersion == 6)
            return NUM_FEATURES_GLOBAL_V5;
        else if (modelVersion == 5)
            return NUM_FEATURES_GLOBAL_V4;
        else if (modelVersion == 3 || modelVersion == 4)
            return NUM_FEATURES_GLOBAL_V3;

        return -1;
    }

    // SGF metadata encoder input versions
    int getNumInputMetaChannels(int metaEncoderVersion)
    {
        if (metaEncoderVersion == 0)
            return 0;
        if (metaEncoderVersion == 1)
            return 192;
        return -1;
    }

    int getNumPolicyChannel(int modelVersion)
    {
        if (modelVersion >= 16)
        {
            return 4;
        }
        else if (modelVersion >= 12)
        {
            return 2;
        }
        else
        {
            return 1;
        }
    }

    int getNumScoreValueChannel(int modelVersion)
    {
        if (modelVersion >= 9)
        {
            return 6;
        }
        else if (modelVersion >= 8)
        {
            return 4;
        }
        else if (modelVersion >= 4)
        {
            return 2;
        }
        else
        {
            return 1;
        }
    }

} // namespace NNModelVersion
