#include "ModelDescription.hpp"

namespace KataGoCoreML
{
    ConvLayerDesc::ConvLayerDesc()
        : convYSize(0),
          convXSize(0),
          inChannels(0),
          outChannels(0),
          dilationY(1),
          dilationX(1) {}

    BatchNormLayerDesc::BatchNormLayerDesc()
        : numChannels(0), epsilon(0.001f), hasScale(false), hasBias(false) {}

    ActivationLayerDesc::ActivationLayerDesc()
        : name(), activation(ACTIVATION_RELU) {}

    MatMulLayerDesc::MatMulLayerDesc()
        : name(), inChannels(0), outChannels(0), weights() {}

    MatBiasLayerDesc::MatBiasLayerDesc()
        : name(), numChannels(0), weights() {}

    ResidualBlockDesc::ResidualBlockDesc() {}

    GlobalPoolingResidualBlockDesc::GlobalPoolingResidualBlockDesc() {}

    NestedBottleneckResidualBlockDesc::NestedBottleneckResidualBlockDesc() {}

    SGFMetadataEncoderDesc::SGFMetadataEncoderDesc()
        : metaEncoderVersion(0), numInputMetaChannels(0) {}

    TrunkDesc::TrunkDesc()
        : modelVersion(-1),
          numBlocks(0),
          trunkNumChannels(0),
          midNumChannels(0),
          regularNumChannels(0),
          gpoolNumChannels(0),
          metaEncoderVersion(0) {}

    PolicyHeadDesc::PolicyHeadDesc() : modelVersion(-1) {}

    ValueHeadDesc::ValueHeadDesc() : modelVersion(-1) {}

    ModelPostProcessParams::ModelPostProcessParams()
        : tdScoreMultiplier(20.0),
          scoreMeanMultiplier(20.0),
          scoreStdevMultiplier(20.0),
          leadMultiplier(20.0),
          varianceTimeMultiplier(40.0),
          shorttermValueErrorMultiplier(0.25),
          shorttermScoreErrorMultiplier(30.0) {}

    ModelDesc::ModelDesc()
        : modelVersion(-1),
          numInputChannels(0),
          numInputGlobalChannels(0),
          numInputMetaChannels(0),
          numPolicyChannels(0),
          numValueChannels(0),
          numScoreValueChannels(0),
          numOwnershipChannels(0),
          metaEncoderVersion(0),
          postProcessParams() {}

} // namespace KataGoCoreML
