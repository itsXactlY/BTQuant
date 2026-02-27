#include "../include/input_validator.h"
#include <algorithm>
#include <cctype>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>

namespace btq {

// Sanitize string input by removing dangerous characters
std::string InputValidator::sanitizeString(const std::string& input) {
    std::string sanitized;
    
    // Remove potentially dangerous characters
    for (char c : input) {
        if (c != '\0' && c != '\\' && c != '\"' && c != '\'' && c != '<' && c != '>') {
            sanitized += c;
        }
    }
    
    return sanitized;
}

// Validate numeric input range
bool InputValidator::validateNumericRange(double value, double min, double max) {
    if (value < min || value > max) {
        return false;
    }
    return true;
}

// Validate integer input range
bool InputValidator::validateIntegerRange(int value, int min, int max) {
    if (value < min || value > max) {
        return false;
    }
    return true;
}

// Validate string length
bool InputValidator::validateStringLength(const std::string& str, size_t minLength, size_t maxLength) {
    if (str.length() < minLength || str.length() > maxLength) {
        return false;
    }
    return true;
}

// Check if string contains only alphanumeric characters
bool InputValidator::isAlphanumeric(const std::string& str) {
    if (str.empty()) {
        return false;
    }
    
    for (char c : str) {
        if (!std::isalnum(c)) {
            return false;
        }
    }
    return true;
}

// Validate email format
bool InputValidator::isValidEmail(const std::string& email) {
    const std::regex emailPattern(
        R"(^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$)"
    );
    return std::regex_match(email, emailPattern);
}

// Validate URL format
bool InputValidator::isValidURL(const std::string& url) {
    const std::regex urlPattern(
        R"(^(https?://)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(:[0-9]+)?(/.*)?$)"
    );
    return std::regex_match(url, urlPattern);
}

// Sanitize SQL input to prevent injection
std::string InputValidator::sanitizeSQL(const std::string& input) {
    std::string sanitized = input;
    
    // Remove potentially dangerous SQL keywords
    std::vector<std::string> dangerousPatterns = {
        "DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER", 
        "EXEC", "UNION", "SELECT", "FROM", "WHERE", "--", "/*", "*/"
    };
    
    for (const auto& pattern : dangerousPatterns) {
        size_t pos = 0;
        while ((pos = sanitized.find(pattern, pos)) != std::string::npos) {
            // Replace with safe alternative or remove
            sanitized.replace(pos, pattern.length(), "");
        }
    }
    
    return sanitized;
}

// Validate JSON-like structure
bool InputValidator::isValidJSONStructure(const std::string& jsonStr) {
    if (jsonStr.empty()) {
        return false;
    }
    
    int braceCount = 0;
    int bracketCount = 0;
    bool inQuotes = false;
    char quoteChar = '"';
    
    for (size_t i = 0; i < jsonStr.length(); ++i) {
        char c = jsonStr[i];
        
        if (c == '"' || c == '\'') {
            if (!inQuotes) {
                inQuotes = true;
                quoteChar = c;
            } else if (c == quoteChar) {
                inQuotes = false;
            }
        }
        
        if (!inQuotes) {
            if (c == '{') {
                braceCount++;
            } else if (c == '}') {
                braceCount--;
                if (braceCount < 0) {
                    return false;
                }
            } else if (c == '[') {
                bracketCount++;
            } else if (c == ']') {
                bracketCount--;
                if (bracketCount < 0) {
                    return false;
                }
            }
        }
    }
    
    return (braceCount == 0 && bracketCount == 0);
}

// Validate file path to prevent directory traversal
bool InputValidator::isValidFilePath(const std::string& path) {
    // Check for directory traversal attempts
    if (path.find("../") != std::string::npos || path.find("..\\") != std::string::npos) {
        return false;
    }
    
    // Basic path validation
    if (path.empty() || path.length() > 255) {
        return false;
    }
    
    return true;
}

// Validate user input with multiple checks
ValidationResult InputValidator::validateUserInput(const std::string& input, 
                                                  const ValidationOptions& options) {
    ValidationResult result;
    result.isValid = true;
    result.sanitizedValue = input;
    
    // Length validation
    if (options.validateLength) {
        if (!validateStringLength(input, options.minLength, options.maxLength)) {
            result.isValid = false;
            result.errors.push_back("Input length validation failed");
        }
    }
    
    // Character validation
    if (options.validateCharacters) {
        if (!isAlphanumeric(input)) {
            result.isValid = false;
            result.errors.push_back("Input contains invalid characters");
        }
    }
    
    // Sanitization
    if (options.shouldSanitize) {
        result.sanitizedValue = sanitizeString(input);
    }
    
    return result;
}

// Validate data from external sources
ValidationResult InputValidator::validateExternalData(const std::string& data,
                                                     const ExternalDataValidationOptions& options) {
    ValidationResult result;
    result.isValid = true;
    result.sanitizedValue = data;
    
    // Apply appropriate validation based on data type
    switch (options.dataType) {
        case DataType::EMAIL:
            if (!isValidEmail(data)) {
                result.isValid = false;
                result.errors.push_back("Invalid email format");
            }
            break;
            
        case DataType::URL:
            if (!isValidURL(data)) {
                result.isValid = false;
                result.errors.push_back("Invalid URL format");
            }
            break;
            
        case DataType::FILE_PATH:
            if (!isValidFilePath(data)) {
                result.isValid = false;
                result.errors.push_back("Invalid file path");
            }
            break;
            
        case DataType::JSON:
            if (!isValidJSONStructure(data)) {
                result.isValid = false;
                result.errors.push_back("Invalid JSON structure");
            }
            break;
            
        case DataType::SQL:
            if (options.shouldSanitize) {
                result.sanitizedValue = sanitizeSQL(data);
            }
            break;
            
        case DataType::GENERIC:
            // Apply generic validation
            if (data.empty()) {
                result.isValid = false;
                result.errors.push_back("Data is empty");
            }
            break;
    }
    
    // Additional sanitization if requested
    if (options.shouldSanitize && options.dataType != DataType::SQL) {
        result.sanitizedValue = sanitizeString(result.sanitizedValue);
    }
    
    return result;
}

// Validate numeric input from external sources
ValidationResult InputValidator::validateNumericInput(double value, 
                                                     const NumericValidationOptions& options) {
    ValidationResult result;
    result.isValid = true;
    result.numericValue = value;
    
    if (!validateNumericRange(value, options.minValue, options.maxValue)) {
        result.isValid = false;
        result.errors.push_back("Numeric value out of valid range");
    }
    
    return result;
}

// Validate integer input from external sources
ValidationResult InputValidator::validateIntegerInput(int value, 
                                                     const IntegerValidationOptions& options) {
    ValidationResult result;
    result.isValid = true;
    result.integerValue = value;
    
    if (!validateIntegerRange(value, options.minValue, options.maxValue)) {
        result.isValid = false;
        result.errors.push_back("Integer value out of valid range");
    }
    
    return result;
}

// Prevent crashes from bad data by checking buffer bounds
bool InputValidator::safeStringCopy(char* dest, const char* src, size_t destSize) {
    if (!dest || !src || destSize == 0) {
        return false;
    }
    
    size_t srcLen = strlen(src);
    size_t copyLen = std::min(srcLen, destSize - 1);  // Leave space for null terminator
    
    strncpy(dest, src, copyLen);
    dest[copyLen] = '\0';
    
    return true;
}

// Validate array bounds to prevent buffer overflows
bool InputValidator::validateArrayBounds(size_t index, size_t arraySize) {
    if (index >= arraySize) {
        return false;
    }
    return true;
}

// Validate pointer to prevent null pointer dereference
bool InputValidator::validatePointer(const void* ptr) {
    return ptr != nullptr;
}

// Validate that a number is finite (not NaN or infinity)
bool InputValidator::isFiniteNumber(double value) {
    return std::isfinite(value);
}

// Validate that a number is positive
bool InputValidator::isPositiveNumber(double value) {
    return value > 0;
}

} // namespace btq