#include "../include/input_validator.h"
#include <iostream>
#include <cassert>
#include <limits>

int main() {
    std::cout << "Running Input Validator Tests..." << std::endl;

    // Test string sanitization
    std::string dirtyInput = "Hello<script>alert('xss')</script>";
    std::string sanitized = btq::InputValidator::sanitizeString(dirtyInput);
    std::cout << "Original: " << dirtyInput << std::endl;
    std::cout << "Sanitized: " << sanitized << std::endl;
    
    // Test string length validation
    assert(btq::InputValidator::validateStringLength("hello", 3, 10) == true);
    assert(btq::InputValidator::validateStringLength("hi", 3, 10) == false);
    std::cout << "String length validation passed." << std::endl;
    
    // Test numeric range validation
    assert(btq::InputValidator::validateNumericRange(5.0, 1.0, 10.0) == true);
    assert(btq::InputValidator::validateNumericRange(15.0, 1.0, 10.0) == false);
    std::cout << "Numeric range validation passed." << std::endl;
    
    // Test email validation
    assert(btq::InputValidator::isValidEmail("test@example.com") == true);
    assert(btq::InputValidator::isValidEmail("invalid-email") == false);
    std::cout << "Email validation passed." << std::endl;
    
    // Test URL validation
    assert(btq::InputValidator::isValidURL("https://example.com") == true);
    assert(btq::InputValidator::isValidURL("invalid-url") == false);
    std::cout << "URL validation passed." << std::endl;
    
    // Test file path validation
    assert(btq::InputValidator::isValidFilePath("valid/path/file.txt") == true);
    assert(btq::InputValidator::isValidFilePath("../../dangerous/path") == false);
    std::cout << "File path validation passed." << std::endl;
    
    // Test user input validation
    btq::ValidationOptions options;
    options.minLength = 1;
    options.maxLength = 100;
    options.validateLength = true;
    options.shouldSanitize = true;
    
    btq::ValidationResult result = btq::InputValidator::validateUserInput("test input", options);
    assert(result.isValid == true);
    std::cout << "User input validation passed." << std::endl;
    
    // Test external data validation
    btq::ExternalDataValidationOptions extOptions;
    extOptions.dataType = btq::DataType::EMAIL;
    extOptions.shouldSanitize = true;
    
    btq::ValidationResult extResult = btq::InputValidator::validateExternalData("test@example.com", extOptions);
    assert(extResult.isValid == true);
    std::cout << "External data validation passed." << std::endl;
    
    // Test numeric validation
    btq::NumericValidationOptions numOptions;
    numOptions.minValue = 0.0;
    numOptions.maxValue = 100.0;
    
    btq::ValidationResult numResult = btq::InputValidator::validateNumericInput(50.0, numOptions);
    assert(numResult.isValid == true);
    std::cout << "Numeric input validation passed." << std::endl;
    
    // Test array bounds validation
    assert(btq::InputValidator::validateArrayBounds(5, 10) == true);
    assert(btq::InputValidator::validateArrayBounds(15, 10) == false);
    std::cout << "Array bounds validation passed." << std::endl;
    
    // Test pointer validation
    int testVar = 42;
    assert(btq::InputValidator::validatePointer(&testVar) == true);
    assert(btq::InputValidator::validatePointer(nullptr) == false);
    std::cout << "Pointer validation passed." << std::endl;
    
    // Test finite number validation
    assert(btq::InputValidator::isFiniteNumber(42.0) == true);
    assert(btq::InputValidator::isFiniteNumber(std::numeric_limits<double>::infinity()) == false);
    std::cout << "Finite number validation passed." << std::endl;
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}