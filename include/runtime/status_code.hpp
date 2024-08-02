#ifndef __FREE_INFER_STATUS_CODE_HPP__
#define __FREE_INFER_STATUS_CODE_HPP__

namespace free_infer {
enum class RuntimeParameterType {
  kParameterUnknown = 0,
  kParameterInt,
  kParameterFloat,
  kParameterString,
  kParameterBool,
  kParameterIntArray,
  kParameterFloatArray,
  kParameterStringArray
};

enum class RuntimeDataType {
  kTypeUnknown = 0,
  kTypeFloat32 = 1,
  kTypeFloat64 = 2,
  kTypeFloat16 = 3,
  kTypeInt32 = 4,
  kTypeInt64 = 5,
  kTypeInt16 = 6,
  kTypeInt8 = 7,
  kTypeUInt8 = 8,
};

enum class InferStatus {
  kInferUnknown = -1,
  kInferSuccess = 0,

  kInferFailedInputEmpty = 1,
  kInferFailedWeightParameterError = 2,
  kInferFailedBiasParameterError = 3,
  kInferFailedStrideParameterError = 4,
  kInferFailedDimensionParameterError = 5,
  kInferFailedInputOutSizeMatchError = 6,

  kInferFailedOutputSizeError = 7,
  kInferFailedShapeParameterError = 9,
  kInferFailedChannelParameterError = 10,
  kInferFailedOutputEmpty = 11,

};

enum class ParseParameterAttrStatus {
  kParameterMissingUnknown = -1,
  kParameterMissingStride = 1,
  kParameterMissingPadding = 2,
  kParameterMissingKernel = 3,
  kParameterMissingUseBias = 4,
  kParameterMissingInChannel = 5,
  kParameterMissingOutChannel = 6,

  kParameterMissingEps = 7,
  kParameterMissingNumFeatures = 8,
  kParameterMissingDim = 9,
  kParameterMissingExpr = 10,
  kParameterMissingOutHW = 11,
  kParameterMissingShape = 12,
  kParameterMissingGroups = 13,
  kParameterMissingScale = 14,
  kParameterMissingResizeMode = 15,
  kParameterMissingDilation = 16,
  kParameterMissingPaddingMode = 16,

  kAttrMissingBias = 21,
  kAttrMissingWeight = 22,
  kAttrMissingRunningMean = 23,
  kAttrMissingRunningVar = 24,
  kAttrMissingOutFeatures = 25,
  kAttrMissingYoloStrides = 26,
  kAttrMissingYoloAnchorGrides = 27,
  kAttrMissingYoloGrides = 28,
  kAttrMissingInFeatures,

  kParameterAttrParseSuccess = 0
};
}  // namespace free_infer

#endif  // __FREE_INFER_STATUS_CODE_HPP__