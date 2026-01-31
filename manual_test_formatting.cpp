#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

// Test the number formatting logic manually
std::string test_formatNumber(double value, int format, int decimal_places) {
    std::ostringstream oss;
    
    enum NumberFormat {
        Raw,           // 0
        ThousandsK,    // 1
        MillionsM,     // 2
        Scientific,    // 3
        CustomDecimal  // 4
    };
    
    switch (format) {
        case Raw:
            oss << std::fixed << std::setprecision(decimal_places) << value;
            break;

        case ThousandsK:
            if (std::abs(value) >= 1000000000.0) {
                // Billions
                oss << std::fixed << std::setprecision(decimal_places) << (value / 1000000000.0) << "B";
            } else if (std::abs(value) >= 1000000.0) {
                // Millions
                oss << std::fixed << std::setprecision(decimal_places) << (value / 1000000.0) << "M";
            } else if (std::abs(value) >= 1000.0) {
                // Thousands
                oss << std::fixed << std::setprecision(decimal_places) << (value / 1000.0) << "K";
            } else {
                // Raw value
                oss << std::fixed << std::setprecision(decimal_places) << value;
            }
            break;

        case MillionsM:
            if (std::abs(value) >= 1000000000.0) {
                // Billions
                oss << std::fixed << std::setprecision(decimal_places) << (value / 1000000000.0) << "B";
            } else if (std::abs(value) >= 1000000.0) {
                // Millions
                oss << std::fixed << std::setprecision(decimal_places) << (value / 1000000.0) << "M";
            } else if (std::abs(value) >= 1000.0) {
                // Thousands
                oss << std::fixed << std::setprecision(decimal_places) << (value / 1000.0) << "K";
            } else {
                // Raw value
                oss << std::fixed << std::setprecision(decimal_places) << value;
            }
            break;

        case Scientific:
            oss << std::scientific << std::setprecision(decimal_places) << value;
            break;

        case CustomDecimal:
            oss << std::fixed << std::setprecision(decimal_places) << value;
            break;
    }

    return oss.str();
}

int main() {
    std::cout << "Testing number formatting options:\n\n";
    
    // Test values
    double test_values[] = {123.456, 1234.567, 1234567.89, 1234567890.12, -1234.567, -1234567.89};
    
    for (double val : test_values) {
        std::cout << "Value: " << val << "\n";
        
        // Test Raw format
        std::string raw = test_formatNumber(val, 0, 2);
        std::cout << "  Raw: " << raw << "\n";
        
        // Test ThousandsK format
        std::string thousands = test_formatNumber(val, 1, 2);
        std::cout << "  ThousandsK: " << thousands << "\n";
        
        // Test MillionsM format
        std::string millions = test_formatNumber(val, 2, 2);
        std::cout << "  MillionsM: " << millions << "\n";
        
        // Test Scientific format
        std::string scientific = test_formatNumber(val, 3, 2);
        std::cout << "  Scientific: " << scientific << "\n";
        
        // Test CustomDecimal format
        std::string custom = test_formatNumber(val, 4, 3);
        std::cout << "  CustomDecimal (3): " << custom << "\n";
        
        std::cout << "\n";
    }
    
    return 0;
}