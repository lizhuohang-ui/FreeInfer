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

}

#endif // __FREE_INFER_STATUS_CODE_HPP__