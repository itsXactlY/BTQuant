#include "error_handling/result.hpp"
#include <sstream>
#include <iomanip>

namespace btq {

// Convert ErrorCode to human-readable string
std::string to_string(ErrorCode code) {
    switch (code) {
        case ErrorCode::kSuccess:
            return "Success";
        case ErrorCode::kUnknownError:
            return "Unknown Error";
        case ErrorCode::kInvalidArgument:
            return "Invalid Argument";
        case ErrorCode::kOutOfRange:
            return "Out of Range";
        case ErrorCode::kNotImplemented:
            return "Not Implemented";
        case ErrorCode::kVulkanInitializationFailed:
            return "Vulkan Initialization Failed";
        case ErrorCode::kSwapchainCreationFailed:
            return "Swapchain Creation Failed";
        case ErrorCode::kRenderPassCreationFailed:
            return "Render Pass Creation Failed";
        case ErrorCode::kGraphicsPipelineCreationFailed:
            return "Graphics Pipeline Creation Failed";
        case ErrorCode::kFramebufferCreationFailed:
            return "Framebuffer Creation Failed";
        case ErrorCode::kCommandBufferAllocationFailed:
            return "Command Buffer Allocation Failed";
        case ErrorCode::kShaderCompilationFailed:
            return "Shader Compilation Failed";
        case ErrorCode::kTextureLoadingFailed:
            return "Texture Loading Failed";
        case ErrorCode::kModelLoadingFailed:
            return "Model Loading Failed";
        case ErrorCode::kVertexBufferCreationFailed:
            return "Vertex Buffer Creation Failed";
        case ErrorCode::kIndexBufferCreationFailed:
            return "Index Buffer Creation Failed";
        case ErrorCode::kUniformBufferCreationFailed:
            return "Uniform Buffer Creation Failed";
        case ErrorCode::kDescriptorSetCreationFailed:
            return "Descriptor Set Creation Failed";
        case ErrorCode::kOutOfMemory:
            return "Out of Memory";
        case ErrorCode::kDeviceLost:
            return "Device Lost";
        case ErrorCode::kSurfaceLost:
            return "Surface Lost";
        case ErrorCode::kResourceNotFound:
            return "Resource Not Found";
        case ErrorCode::kResourceAlreadyExists:
            return "Resource Already Exists";
        case ErrorCode::kResourceInUse:
            return "Resource In Use";
        case ErrorCode::kResourceCorrupted:
            return "Resource Corrupted";
        case ErrorCode::kConnectionFailed:
            return "Connection Failed";
        case ErrorCode::kDataTransferFailed:
            return "Data Transfer Failed";
        case ErrorCode::kTimeout:
            return "Timeout";
        case ErrorCode::kAuthenticationFailed:
            return "Authentication Failed";
        case ErrorCode::kAuthorizationFailed:
            return "Authorization Failed";
        case ErrorCode::kNetworkUnreachable:
            return "Network Unreachable";
        case ErrorCode::kFileNotFound:
            return "File Not Found";
        case ErrorCode::kFileAccessDenied:
            return "File Access Denied";
        case ErrorCode::kFileCorrupted:
            return "File Corrupted";
        case ErrorCode::kFileWriteFailed:
            return "File Write Failed";
        case ErrorCode::kFileReadFailed:
            return "File Read Failed";
        case ErrorCode::kConfigNotFound:
            return "Configuration Not Found";
        case ErrorCode::kConfigInvalidFormat:
            return "Configuration Invalid Format";
        case ErrorCode::kConfigValueOutOfRange:
            return "Configuration Value Out of Range";
        case ErrorCode::kConfigMissingRequiredField:
            return "Configuration Missing Required Field";
        case ErrorCode::kThreadCreationFailed:
            return "Thread Creation Failed";
        case ErrorCode::kMutexLockFailed:
            return "Mutex Lock Failed";
        case ErrorCode::kConditionVariableError:
            return "Condition Variable Error";
        case ErrorCode::kDeadlockDetected:
            return "Deadlock Detected";
        case ErrorCode::kWindowCreationFailed:
            return "Window Creation Failed";
        case ErrorCode::kInputHandlingFailed:
            return "Input Handling Failed";
        case ErrorCode::kUIElementNotFound:
            return "UI Element Not Found";
        case ErrorCode::kOrderSubmissionFailed:
            return "Order Submission Failed";
        case ErrorCode::kPositionManagementFailed:
            return "Position Management Failed";
        case ErrorCode::kRiskManagementViolation:
            return "Risk Management Violation";
        case ErrorCode::kMarketDataUnavailable:
            return "Market Data Unavailable";
        case ErrorCode::kInsufficientFunds:
            return "Insufficient Funds";
        case ErrorCode::kCalculationOverflow:
            return "Calculation Overflow";
        case ErrorCode::kDivisionByZero:
            return "Division by Zero";
        case ErrorCode::kInvalidDataFormat:
            return "Invalid Data Format";
        case ErrorCode::kMathDomainError:
            return "Math Domain Error";
        default:
            return "Unknown Error Code";
    }
}

// Convert ErrorInfo to human-readable string
std::string to_string(const ErrorInfo& error) {
    std::ostringstream oss;
    oss << "[" << to_string(error.code) << "] " << error.message;
    
    if (!error.details.empty()) {
        oss << " - Details: " << error.details;
    }
    
    if (!error.file.empty()) {
        oss << " (File: " << error.file << ":" << error.line << ")";
    }
    
    if (!error.function.empty()) {
        oss << " [Function: " << error.function << "]";
    }
    
    return oss.str();
}

// Helper function to create a Result with an error
template<typename T>
Result<T> make_error(ErrorCode code, std::string_view msg,
                     std::string_view details,
                     std::string_view file,
                     int line,
                     std::string_view func) {
    return Result<T>(std::unexpect, ErrorInfo(code, msg, details, file, line, func));
}

// Explicit template instantiations for common types
template Result<int> make_error<int>(ErrorCode, std::string_view, std::string_view, std::string_view, int, std::string_view);
template Result<float> make_error<float>(ErrorCode, std::string_view, std::string_view, std::string_view, int, std::string_view);
template Result<double> make_error<double>(ErrorCode, std::string_view, std::string_view, std::string_view, int, std::string_view);
template Result<std::string> make_error<std::string>(ErrorCode, std::string_view, std::string_view, std::string_view, int, std::string_view);

} // namespace btq