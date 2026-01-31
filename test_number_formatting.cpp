#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

// Include the necessary headers
#include "dependencies/BTQ_Render_Engine/include/components/footprint_panel.hpp"

// Define a simple test function to validate the formatNumber function
int main() {
    using NumberFormat = BTQuant::NumberFormat;

    std::cout << "Testing number formatting options:\n\n";

    // Test values
    double test_values[] = {123.456, 1234.567, 1234567.89, 1234567890.12, -1234.567, -1234567.89};

    for (double val : test_values) {
        std::cout << "Value: " << val << "\n";

        // Test Raw format
        std::string raw = BTQuant::FootprintPanel::formatNumber(val, NumberFormat::Raw, 2);
        std::cout << "  Raw: " << raw << "\n";

        // Test ThousandsK format
        std::string thousands = BTQuant::FootprintPanel::formatNumber(val, NumberFormat::ThousandsK, 2);
        std::cout << "  ThousandsK: " << thousands << "\n";

        // Test MillionsM format
        std::string millions = BTQuant::FootprintPanel::formatNumber(val, NumberFormat::MillionsM, 2);
        std::cout << "  MillionsM: " << millions << "\n";

        // Test Scientific format
        std::string scientific = BTQuant::FootprintPanel::formatNumber(val, NumberFormat::Scientific, 2);
        std::cout << "  Scientific: " << scientific << "\n";

        // Test CustomDecimal format
        std::string custom = BTQuant::FootprintPanel::formatNumber(val, NumberFormat::CustomDecimal, 3);
        std::cout << "  CustomDecimal (3): " << custom << "\n";

        std::cout << "\n";
    }

    return 0;
}