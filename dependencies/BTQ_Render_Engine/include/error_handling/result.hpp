#pragma once

#include <expected>
#include <string>
#include <string_view>
#include <format>

namespace btq {

// Comprehensive error codes for the BTQ Render Engine
enum class ErrorCode {
    // Generic errors
    kSuccess = 0,
    kUnknownError,
    kInvalidArgument,
    kOutOfRange,
    kNotImplemented,
    
    // Rendering errors
    kVulkanInitializationFailed,
    kSwapchainCreationFailed,
    kRenderPassCreationFailed,
    kGraphicsPipelineCreationFailed,
    kFramebufferCreationFailed,
    kCommandBufferAllocationFailed,
    kShaderCompilationFailed,
    kTextureLoadingFailed,
    kModelLoadingFailed,
    kVertexBufferCreationFailed,
    kIndexBufferCreationFailed,
    kUniformBufferCreationFailed,
    kDescriptorSetCreationFailed,
    kOutOfMemory,
    kDeviceLost,
    kSurfaceLost,
    
    // Resource management errors
    kResourceNotFound,
    kResourceAlreadyExists,
    kResourceInUse,
    kResourceCorrupted,
    
    // Network/Data errors
    kConnectionFailed,
    kDataTransferFailed,
    kTimeout,
    kAuthenticationFailed,
    kAuthorizationFailed,
    kNetworkUnreachable,
    
    // File system errors
    kFileNotFound,
    kFileAccessDenied,
    kFileCorrupted,
    kFileWriteFailed,
    kFileReadFailed,
    
    // Configuration errors
    kConfigNotFound,
    kConfigInvalidFormat,
    kConfigValueOutOfRange,
    kConfigMissingRequiredField,
    
    // Threading/Synchronization errors
    kThreadCreationFailed,
    kMutexLockFailed,
    kConditionVariableError,
    kDeadlockDetected,
    
    // UI/Interaction errors
    kWindowCreationFailed,
    kInputHandlingFailed,
    kUIElementNotFound,
    
    // Trading-specific errors
    kOrderSubmissionFailed,
    kPositionManagementFailed,
    kRiskManagementViolation,
    kMarketDataUnavailable,
    kInsufficientFunds,
    
    // Analytics/Calculation errors
    kCalculationOverflow,
    kDivisionByZero,
    kInvalidDataFormat,
    kMathDomainError,
};

// Error information structure
struct ErrorInfo {
    ErrorCode code;
    std::string message;
    std::string details;
    std::string file;
    int line;
    std::string function;

    ErrorInfo() : code(ErrorCode::kUnknownError), line(0) {}
    
    ErrorInfo(ErrorCode c, std::string_view msg, std::string_view det = "", 
              std::string_view f = "", int l = 0, std::string_view func = "")
        : code(c), message(msg), details(det), file(f), line(l), function(func) {}
};

// Custom Result type using std::expected
template<typename T>
using Result = std::expected<T, ErrorInfo>;

// Helper macros for creating errors
#define BTQ_MAKE_ERROR(code, msg) \
    btq::ErrorInfo(code, msg, "", __FILE__, __LINE__, __FUNCTION__)

#define BTQ_MAKE_ERROR_WITH_DETAILS(code, msg, details) \
    btq::ErrorInfo(code, msg, details, __FILE__, __LINE__, __FUNCTION__)

// Helper macro for propagating errors
#define BTQ_TRY(expr) ({ \
    auto result = (expr); \
    if (!result) { \
        return result; \
    } \
    std::move(result.value()); \
})

// Helper function to create a Result with an error
template<typename T>
Result<T> make_error(ErrorCode code, std::string_view msg,
                     std::string_view details = "",
                     std::string_view file = "",
                     int line = 0,
                     std::string_view func = "");

// Convert ErrorCode to human-readable string
std::string to_string(ErrorCode code);

// Convert ErrorInfo to human-readable string
std::string to_string(const ErrorInfo& error);

// Helper macro for converting exceptions to errors
#define BTQ_CATCH_TO_ERROR(block, error_code, error_msg) \
    [&]() -> ::btq::Result<decltype(block())> { \
        try { \
            return ::btq::Result<decltype(block())>::success(block()); \
        } catch (const std::exception& e) { \
            return ::btq::Result<decltype(block())>::unexpected( \
                BTQ_MAKE_ERROR_WITH_DETAILS(error_code, error_msg, e.what())); \
        } catch (...) { \
            return ::btq::Result<decltype(block())>::unexpected( \
                BTQ_MAKE_ERROR(error_code, error_msg)); \
        } \
    }()

} // namespace btq