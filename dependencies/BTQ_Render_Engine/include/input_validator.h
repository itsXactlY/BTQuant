#ifndef BTQ_INPUT_VALIDATOR_H
#define BTQ_INPUT_VALIDATOR_H

#include <string>
#include <vector>
#include <limits>
#include <cstring>  // for strlen

namespace btq {

// Enum for different data types that need validation
enum class DataType {
    GENERIC,
    EMAIL,
    URL,
    FILE_PATH,
    JSON,
    SQL
};

// Structure to hold validation results
struct ValidationResult {
    bool isValid;
    std::string sanitizedValue;
    double numericValue;
    int integerValue;
    std::vector<std::string> errors;
    
    ValidationResult() : isValid(true), numericValue(0.0), integerValue(0) {}
};

// Options for user input validation
struct ValidationOptions {
    bool validateLength = true;
    bool validateCharacters = false;
    bool shouldSanitize = true;
    size_t minLength = 0;
    size_t maxLength = 1000;
};

// Options for external data validation
struct ExternalDataValidationOptions {
    DataType dataType = DataType::GENERIC;
    bool shouldSanitize = true;
};

// Options for numeric validation
struct NumericValidationOptions {
    double minValue = -std::numeric_limits<double>::infinity();
    double maxValue = std::numeric_limits<double>::infinity();
};

// Options for integer validation
struct IntegerValidationOptions {
    int minValue = std::numeric_limits<int>::min();
    int maxValue = std::numeric_limits<int>::max();
};

class InputValidator {
public:
    // Sanitize string input by removing dangerous characters
    static std::string sanitizeString(const std::string& input);
    
    // Validate numeric input range
    static bool validateNumericRange(double value, double min, double max);
    
    // Validate integer input range
    static bool validateIntegerRange(int value, int min, int max);
    
    // Validate string length
    static bool validateStringLength(const std::string& str, size_t minLength, size_t maxLength);
    
    // Check if string contains only alphanumeric characters
    static bool isAlphanumeric(const std::string& str);
    
    // Validate email format
    static bool isValidEmail(const std::string& email);
    
    // Validate URL format
    static bool isValidURL(const std::string& url);
    
    // Sanitize SQL input to prevent injection
    static std::string sanitizeSQL(const std::string& input);
    
    // Validate JSON-like structure
    static bool isValidJSONStructure(const std::string& jsonStr);
    
    // Validate file path to prevent directory traversal
    static bool isValidFilePath(const std::string& path);
    
    // Validate user input with multiple checks
    static ValidationResult validateUserInput(const std::string& input, 
                                            const ValidationOptions& options);
    
    // Validate data from external sources
    static ValidationResult validateExternalData(const std::string& data,
                                               const ExternalDataValidationOptions& options);
    
    // Validate numeric input from external sources
    static ValidationResult validateNumericInput(double value, 
                                               const NumericValidationOptions& options);
    
    // Validate integer input from external sources
    static ValidationResult validateIntegerInput(int value, 
                                               const IntegerValidationOptions& options);
    
    // Prevent crashes from bad data by checking buffer bounds
    static bool safeStringCopy(char* dest, const char* src, size_t destSize);
    
    // Validate array bounds to prevent buffer overflows
    static bool validateArrayBounds(size_t index, size_t arraySize);
    
    // Validate pointer to prevent null pointer dereference
    static bool validatePointer(const void* ptr);
    
    // Validate that a number is finite (not NaN or infinity)
    static bool isFiniteNumber(double value);
    
    // Validate that a number is positive
    static bool isPositiveNumber(double value);
};

} // namespace btq

#endif // BTQ_INPUT_VALIDATOR_H