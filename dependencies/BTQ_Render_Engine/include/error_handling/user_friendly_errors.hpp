#pragma once

#include "error_handling/result.hpp"
#include <unordered_map>
#include <string>

namespace btq {

// User-friendly error message generator
class UserFriendlyErrorMessage {
public:
    // Generate a user-friendly error message based on error code
    static std::string generate_user_friendly_message(ErrorCode code) {
        switch (code) {
            case ErrorCode::kSuccess:
                return "Operation completed successfully.";
                
            case ErrorCode::kUnknownError:
                return "An unknown error occurred. Please try again or contact support.";
                
            case ErrorCode::kInvalidArgument:
                return "Invalid input provided. Please check your input and try again.";
                
            case ErrorCode::kOutOfRange:
                return "Value is outside the acceptable range. Please adjust your input.";
                
            case ErrorCode::kNotImplemented:
                return "This feature is not yet implemented. Check for updates.";
                
            case ErrorCode::kVulkanInitializationFailed:
                return "Failed to initialize graphics system. Please ensure your graphics drivers are up to date.";
                
            case ErrorCode::kSwapchainCreationFailed:
                return "Failed to create display surface. Your graphics hardware may not support this feature.";
                
            case ErrorCode::kRenderPassCreationFailed:
                return "Failed to configure rendering pipeline. This may be due to graphics driver issues.";
                
            case ErrorCode::kGraphicsPipelineCreationFailed:
                return "Failed to create graphics rendering pipeline. Check your graphics settings.";
                
            case ErrorCode::kFramebufferCreationFailed:
                return "Failed to create frame buffer. This could be due to insufficient graphics memory.";
                
            case ErrorCode::kCommandBufferAllocationFailed:
                return "Failed to allocate command buffer. This may be due to insufficient system resources.";
                
            case ErrorCode::kShaderCompilationFailed:
                return "Failed to compile graphics shader. This may be due to graphics driver or hardware limitations.";
                
            case ErrorCode::kTextureLoadingFailed:
                return "Failed to load texture resource. The image file may be corrupted or incompatible.";
                
            case ErrorCode::kModelLoadingFailed:
                return "Failed to load 3D model. The model file may be corrupted or in an unsupported format.";
                
            case ErrorCode::kVertexBufferCreationFailed:
                return "Failed to create vertex buffer. This may be due to insufficient graphics memory.";
                
            case ErrorCode::kIndexBufferCreationFailed:
                return "Failed to create index buffer. This may be due to insufficient graphics memory.";
                
            case ErrorCode::kUniformBufferCreationFailed:
                return "Failed to create uniform buffer. This may be due to insufficient graphics memory.";
                
            case ErrorCode::kDescriptorSetCreationFailed:
                return "Failed to create descriptor set. This may be due to graphics driver issues.";
                
            case ErrorCode::kOutOfMemory:
                return "Insufficient memory available. Close other applications and try again.";
                
            case ErrorCode::kDeviceLost:
                return "Graphics device lost connection. Restart the application or check your graphics hardware.";
                
            case ErrorCode::kSurfaceLost:
                return "Display surface lost. Try restarting the application.";
                
            case ErrorCode::kResourceNotFound:
                return "Required resource not found. Reinstall the application or restore missing files.";
                
            case ErrorCode::kResourceAlreadyExists:
                return "Resource already exists. Choose a different name or remove the existing resource.";
                
            case ErrorCode::kResourceInUse:
                return "Resource is currently in use. Close any applications using this resource and try again.";
                
            case ErrorCode::kResourceCorrupted:
                return "Resource file is corrupted. Restore from backup or reinstall the application.";
                
            case ErrorCode::kConnectionFailed:
                return "Failed to establish network connection. Check your internet connection and firewall settings.";
                
            case ErrorCode::kDataTransferFailed:
                return "Data transfer failed. Check your network connection and try again.";
                
            case ErrorCode::kTimeout:
                return "Operation timed out. Check your network connection or try again later.";
                
            case ErrorCode::kAuthenticationFailed:
                return "Authentication failed. Check your credentials and try again.";
                
            case ErrorCode::kAuthorizationFailed:
                return "Authorization failed. You may not have permission to perform this action.";
                
            case ErrorCode::kNetworkUnreachable:
                return "Network is unreachable. Check your network connection and settings.";
                
            case ErrorCode::kFileNotFound:
                return "File not found. Check the file path and ensure the file exists.";
                
            case ErrorCode::kFileAccessDenied:
                return "Access denied to file. Check file permissions or run as administrator.";
                
            case ErrorCode::kFileCorrupted:
                return "File is corrupted. Restore from backup or obtain a fresh copy.";
                
            case ErrorCode::kFileWriteFailed:
                return "Failed to write to file. Check disk space and file permissions.";
                
            case ErrorCode::kFileReadFailed:
                return "Failed to read file. Check file permissions and file integrity.";
                
            case ErrorCode::kConfigNotFound:
                return "Configuration file not found. Restore default settings or reinstall the application.";
                
            case ErrorCode::kConfigInvalidFormat:
                return "Configuration file has invalid format. Reset to default settings.";
                
            case ErrorCode::kConfigValueOutOfRange:
                return "Configuration value is out of acceptable range. Adjust the setting and try again.";
                
            case ErrorCode::kConfigMissingRequiredField:
                return "Configuration is missing required fields. Reset to default settings.";
                
            case ErrorCode::kThreadCreationFailed:
                return "Failed to create thread. System may be low on resources.";
                
            case ErrorCode::kMutexLockFailed:
                return "Failed to acquire lock. This may indicate a system issue.";
                
            case ErrorCode::kConditionVariableError:
                return "Condition variable error occurred. This may indicate a system issue.";
                
            case ErrorCode::kDeadlockDetected:
                return "Deadlock detected. Application may need to be restarted.";
                
            case ErrorCode::kWindowCreationFailed:
                return "Failed to create application window. Check your display settings.";
                
            case ErrorCode::kInputHandlingFailed:
                return "Failed to process input. Check your input devices.";
                
            case ErrorCode::kUIElementNotFound:
                return "Requested UI element not found. Interface may need to be refreshed.";
                
            case ErrorCode::kOrderSubmissionFailed:
                return "Failed to submit order. Check market conditions and try again.";
                
            case ErrorCode::kPositionManagementFailed:
                return "Failed to manage position. Check your account status.";
                
            case ErrorCode::kRiskManagementViolation:
                return "Risk management rules violated. Check your position limits.";
                
            case ErrorCode::kMarketDataUnavailable:
                return "Market data unavailable. Check your data feed connection.";
                
            case ErrorCode::kInsufficientFunds:
                return "Insufficient funds for this operation. Check your account balance.";
                
            case ErrorCode::kCalculationOverflow:
                return "Calculation overflow occurred. Input values may be too large.";
                
            case ErrorCode::kDivisionByZero:
                return "Division by zero attempted. Check your input values.";
                
            case ErrorCode::kInvalidDataFormat:
                return "Invalid data format. Check your input data.";
                
            case ErrorCode::kMathDomainError:
                return "Mathematical domain error. Input values are outside valid range.";
                
            default:
                return "An unexpected error occurred. Please contact support.";
        }
    }
    
    // Generate a detailed error message with troubleshooting steps
    static std::string generate_detailed_error_message(ErrorCode code) {
        std::string base_message = generate_user_friendly_message(code);
        std::string troubleshooting_steps = generate_troubleshooting_steps(code);
        
        return base_message + "\n\nTroubleshooting:\n" + troubleshooting_steps;
    }
    
private:
    static std::string generate_troubleshooting_steps(ErrorCode code) {
        switch (code) {
            case ErrorCode::kVulkanInitializationFailed:
            case ErrorCode::kSwapchainCreationFailed:
            case ErrorCode::kRenderPassCreationFailed:
            case ErrorCode::kGraphicsPipelineCreationFailed:
                return "- Update your graphics drivers to the latest version\n"
                       "- Check if your GPU supports Vulkan API\n"
                       "- Try running the application with reduced graphics settings\n"
                       "- Restart your computer and try again";
                
            case ErrorCode::kOutOfMemory:
            case ErrorCode::kVertexBufferCreationFailed:
            case ErrorCode::kIndexBufferCreationFailed:
            case ErrorCode::kUniformBufferCreationFailed:
                return "- Close other applications to free up memory\n"
                       "- Reduce the complexity of your scene or data\n"
                       "- Increase virtual memory/page file size\n"
                       "- Add more RAM if possible";
                
            case ErrorCode::kConnectionFailed:
            case ErrorCode::kDataTransferFailed:
            case ErrorCode::kTimeout:
            case ErrorCode::kNetworkUnreachable:
                return "- Check your internet connection\n"
                       "- Verify firewall settings allow the application\n"
                       "- Try connecting to a different network\n"
                       "- Contact your network administrator";
                
            case ErrorCode::kFileNotFound:
            case ErrorCode::kFileAccessDenied:
            case ErrorCode::kFileCorrupted:
            case ErrorCode::kFileWriteFailed:
            case ErrorCode::kFileReadFailed:
                return "- Verify the file path is correct\n"
                       "- Check file permissions\n"
                       "- Ensure the file is not open in another application\n"
                       "- Restore from backup if the file is corrupted";
                
            case ErrorCode::kConfigNotFound:
            case ErrorCode::kConfigInvalidFormat:
            case ErrorCode::kConfigValueOutOfRange:
            case ErrorCode::kConfigMissingRequiredField:
                return "- Reset to default configuration settings\n"
                       "- Reinstall the application to restore defaults\n"
                       "- Manually edit the configuration file if you're experienced\n"
                       "- Contact support for configuration assistance";
                
            default:
                return "- Restart the application\n"
                       "- Check system requirements are met\n"
                       "- Update to the latest version\n"
                       "- Contact support if the problem persists";
        }
    }
};

// Helper function to create user-friendly error results
template<typename T>
Result<T> make_user_friendly_error(ErrorCode code, const std::string& custom_message = "") {
    std::string user_message = custom_message.empty() ? 
        UserFriendlyErrorMessage::generate_user_friendly_message(code) : custom_message;
    
    return Result<T>::unexpected(
        ErrorInfo(code, user_message, UserFriendlyErrorMessage::generate_detailed_error_message(code))
    );
}

// Macro for creating user-friendly error results
#define BTQ_USER_ERROR(code, custom_msg) btq::make_user_friendly_error<decltype(auto)>(code, custom_msg)

} // namespace btq