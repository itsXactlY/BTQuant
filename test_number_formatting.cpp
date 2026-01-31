#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <string>

// Copy the formatNumber function from the footprint panel to test it
enum class NumberFormat {
  Raw,           // Raw numbers without any formatting
  ThousandsK,    // Format with K suffix for thousands
  MillionsM,     // Format with M suffix for millions
  Scientific,    // Scientific notation
  CustomDecimal  // Custom decimal places
};

std::string formatNumber(double value, NumberFormat format, int decimal_places) {
  std::ostringstream oss;

  // Limit decimal places to reasonable range to prevent overflow issues
  decimal_places = std::max(0, std::min(10, decimal_places));

  switch (format) {
    case NumberFormat::Raw:
      // Raw number formatting with special handling for edge cases
      if (value == 0.0) {
        oss << "0.0";
      } else {
        oss << std::fixed << std::setprecision(decimal_places) << value;
      }
      break;

    case NumberFormat::ThousandsK:
      if (std::abs(value) >= 1e18) {
        // Exa (quintillions)
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1e18) << "E";
      } else if (std::abs(value) >= 1e15) {
        // Peta (quadrillions)
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1e15) << "P";
      } else if (std::abs(value) >= 1e12) {
        // Trillions
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1e12) << "T";
      } else if (std::abs(value) >= 1e9) {
        // Billions
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1e9) << "B";
      } else if (std::abs(value) >= 1e6) {
        // Millions
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1e6) << "M";
      } else if (std::abs(value) >= 1e3) {
        // Thousands
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1e3) << "K";
      } else {
        // Raw value
        oss << std::fixed << std::setprecision(decimal_places) << value;
      }
      break;

    case NumberFormat::MillionsM:
      // Format primarily in millions, with fallback to thousands for smaller values and billions for larger values
      if (value == 0.0) {
        oss << "0.0";
      } else if (std::abs(value) >= 1e12) {
        // Trillions - show as trillions
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1e12) << "T";
      } else if (std::abs(value) >= 1e9) {
        // Billions - show as billions
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1e9) << "B";
      } else if (std::abs(value) >= 1e6) {
        // Millions - show as millions (this is the primary unit for this format)
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1e6) << "M";
      } else if (std::abs(value) >= 1e3) {
        // Thousands - show as thousands
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1e3) << "K";
      } else if (std::abs(value) >= 1.0) {
        // Values between 1 and 1000 - show as raw value
        oss << std::fixed << std::setprecision(decimal_places) << value;
      } else {
        // Small values - use scientific notation for better readability
        oss << std::scientific << std::setprecision(decimal_places) << value;
      }
      break;

    case NumberFormat::Scientific:
      // Scientific notation with special handling for edge cases
      if (value == 0.0) {
        oss << "0.0";
      } else {
        oss << std::scientific << std::setprecision(decimal_places) << value;
      }
      break;

    case NumberFormat::CustomDecimal:
      oss << std::fixed << std::setprecision(decimal_places) << value;
      break;
  }

  return oss.str();
}

int main() {
  std::cout << "Testing Number Formatting Options:\n";
  std::cout << "==================================\n\n";

  // Test values
  double test_values[] = {0.0, 123.456, 1234.567, 12345.67, 123456.78, 1234567.89, 12345678.90, 123456789.01, 1234567890.12};
  
  std::cout << "Raw Format:\n";
  for (double val : test_values) {
    std::cout << "  " << val << " -> " << formatNumber(val, NumberFormat::Raw, 2) << std::endl;
  }
  std::cout << std::endl;

  std::cout << "ThousandsK Format:\n";
  for (double val : test_values) {
    std::cout << "  " << val << " -> " << formatNumber(val, NumberFormat::ThousandsK, 2) << std::endl;
  }
  std::cout << std::endl;

  std::cout << "MillionsM Format:\n";
  for (double val : test_values) {
    std::cout << "  " << val << " -> " << formatNumber(val, NumberFormat::MillionsM, 2) << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Scientific Format:\n";
  for (double val : test_values) {
    std::cout << "  " << val << " -> " << formatNumber(val, NumberFormat::Scientific, 2) << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Custom Decimal Format (2 decimals):\n";
  for (double val : test_values) {
    std::cout << "  " << val << " -> " << formatNumber(val, NumberFormat::CustomDecimal, 2) << std::endl;
  }
  std::cout << std::endl;

  // Test negative values
  std::cout << "Testing Negative Values (MillionsM Format):\n";
  double neg_values[] = {-123.456, -1234.567, -12345.67, -123456.78, -1234567.89, -12345678.90, -123456789.01};
  for (double val : neg_values) {
    std::cout << "  " << val << " -> " << formatNumber(val, NumberFormat::MillionsM, 2) << std::endl;
  }
  std::cout << std::endl;

  // Test very small values
  std::cout << "Testing Very Small Values (Scientific Format):\n";
  double small_values[] = {0.001, 0.0001, 0.00001, 0.000001, 0.0000001};
  for (double val : small_values) {
    std::cout << "  " << val << " -> " << formatNumber(val, NumberFormat::Scientific, 2) << std::endl;
  }
  std::cout << std::endl;

  return 0;
}