// The definitions in this file shall be consistent with katago/cpp/neuralnet/desc.h

#pragma once

#include <istream>
#include <string>
#include <vector>

using unique_ptr_void = std::unique_ptr<void, void (*)(const void *)>;

namespace KataGoCoreML
{
    struct ConvLayerDesc
    {
        std::string name;
        int convYSize;
        int convXSize;
        int inChannels;
        int outChannels;
        int dilationY;
        int dilationX;
        // outC x inC x H x W (col-major order - W has least stride, outC greatest)
        std::vector<float> weights;

        ConvLayerDesc();
    };

    struct BatchNormLayerDesc
    {
        std::string name;
        int numChannels;
        float epsilon;
        bool hasScale;
        bool hasBias;
        std::vector<float> mean;
        std::vector<float> variance;
        std::vector<float> scale;
        std::vector<float> bias;

        BatchNormLayerDesc();
    };

    static constexpr int ACTIVATION_IDENTITY = 0;
    static constexpr int ACTIVATION_RELU = 1;
    static constexpr int ACTIVATION_MISH = 2;

    struct ActivationLayerDesc
    {
        std::string name;
        int activation;

        ActivationLayerDesc();
    };

    struct MatMulLayerDesc
    {
        std::string name;
        int inChannels;
        int outChannels;
        std::vector<float> weights;

        MatMulLayerDesc();
    };

    struct MatBiasLayerDesc
    {
        std::string name;
        int numChannels;
        std::vector<float> weights;

        MatBiasLayerDesc();
    };

    struct ResidualBlockDesc
    {
        std::string name;
        BatchNormLayerDesc preBN;
        ActivationLayerDesc preActivation;
        ConvLayerDesc regularConv;
        BatchNormLayerDesc midBN;
        ActivationLayerDesc midActivation;
        ConvLayerDesc finalConv;

        ResidualBlockDesc();
    };

    struct GlobalPoolingResidualBlockDesc
    {
        std::string name;
        int modelVersion;
        BatchNormLayerDesc preBN;
        ActivationLayerDesc preActivation;
        ConvLayerDesc regularConv;
        ConvLayerDesc gpoolConv;
        BatchNormLayerDesc gpoolBN;
        ActivationLayerDesc gpoolActivation;
        MatMulLayerDesc gpoolToBiasMul;
        BatchNormLayerDesc midBN;
        ActivationLayerDesc midActivation;
        ConvLayerDesc finalConv;

        GlobalPoolingResidualBlockDesc();
    };

    struct NestedBottleneckResidualBlockDesc
    {
        std::string name;
        int numBlocks;

        BatchNormLayerDesc preBN;
        ActivationLayerDesc preActivation;
        ConvLayerDesc preConv;

        std::vector<std::pair<int, unique_ptr_void>> blocks;

        BatchNormLayerDesc postBN;
        ActivationLayerDesc postActivation;
        ConvLayerDesc postConv;

        NestedBottleneckResidualBlockDesc();
    };

    struct SGFMetadataEncoderDesc
    {
        std::string name;
        int metaEncoderVersion;
        int numInputMetaChannels;
        MatMulLayerDesc mul1;
        MatBiasLayerDesc bias1;
        ActivationLayerDesc act1;
        MatMulLayerDesc mul2;
        MatBiasLayerDesc bias2;
        ActivationLayerDesc act2;
        MatMulLayerDesc mul3;

        SGFMetadataEncoderDesc();
    };

    constexpr int ORDINARY_BLOCK_KIND = 0;
    constexpr int GLOBAL_POOLING_BLOCK_KIND = 2;
    constexpr int NESTED_BOTTLENECK_BLOCK_KIND = 3;

    struct TrunkDesc
    {
        std::string name;
        int modelVersion;
        int numBlocks;
        int trunkNumChannels;
        int midNumChannels;     // Currently every plain residual block must have the same number of mid conv channels
        int regularNumChannels; // Currently every gpool residual block must have the same number of regular conv hannels
        int gpoolNumChannels;   // Currently every gpooling residual block must have the same number of gpooling conv channels

        int metaEncoderVersion;

        ConvLayerDesc initialConv;
        MatMulLayerDesc initialMatMul;
        SGFMetadataEncoderDesc sgfMetadataEncoder;
        std::vector<std::pair<int, unique_ptr_void>> blocks;
        BatchNormLayerDesc trunkTipBN;
        ActivationLayerDesc trunkTipActivation;

        TrunkDesc();
    };

    struct PolicyHeadDesc
    {
        std::string name;
        int modelVersion;
        int policyOutChannels;
        ConvLayerDesc p1Conv;
        ConvLayerDesc g1Conv;
        BatchNormLayerDesc g1BN;
        ActivationLayerDesc g1Activation;
        MatMulLayerDesc gpoolToBiasMul;
        BatchNormLayerDesc p1BN;
        ActivationLayerDesc p1Activation;
        ConvLayerDesc p2Conv;
        MatMulLayerDesc gpoolToPassMul;
        MatBiasLayerDesc gpoolToPassBias;
        ActivationLayerDesc passActivation;
        MatMulLayerDesc gpoolToPassMul2;

        PolicyHeadDesc();
    };

    struct ValueHeadDesc
    {
        std::string name;
        int modelVersion;
        ConvLayerDesc v1Conv;
        BatchNormLayerDesc v1BN;
        ActivationLayerDesc v1Activation;
        MatMulLayerDesc v2Mul;
        MatBiasLayerDesc v2Bias;
        ActivationLayerDesc v2Activation;
        MatMulLayerDesc v3Mul;
        MatBiasLayerDesc v3Bias;
        MatMulLayerDesc sv3Mul;
        MatBiasLayerDesc sv3Bias;
        ConvLayerDesc vOwnershipConv;

        ValueHeadDesc();
    };

    struct ModelPostProcessParams
    {
        double tdScoreMultiplier;
        double scoreMeanMultiplier;
        double scoreStdevMultiplier;
        double leadMultiplier;
        double varianceTimeMultiplier;
        double shorttermValueErrorMultiplier;
        double shorttermScoreErrorMultiplier;

        ModelPostProcessParams();
    };

    struct ModelDesc
    {
        std::string name;
        std::string sha256;
        int modelVersion;
        int numInputChannels;
        int numInputGlobalChannels;
        int numInputMetaChannels;
        int numPolicyChannels;
        int numValueChannels;
        int numScoreValueChannels;
        int numOwnershipChannels;

        int metaEncoderVersion;

        ModelPostProcessParams postProcessParams;

        TrunkDesc trunk;
        PolicyHeadDesc policyHead;
        ValueHeadDesc valueHead;

        ModelDesc();
    };
} // namespace KataGoCoreML
