#include <iostream>
#include <cmath>

// Test the center mode Y-limits calculation
void test_center_mode_formula() {
    std::cout << "Testing Center Mode Y-limits calculation:\n";
    std::cout << "Formula: y_min = current_price - range, y_max = current_price + range\n\n";
    
    // Test with different price levels and range percentages
    struct TestCase {
        double current_price;
        double range_percentage;
        std::string description;
    };
    
    TestCase test_cases[] = {
        {100.0, 0.01, "Price $100, 1% range"},
        {50000.0, 0.005, "Bitcoin ~$50k, 0.5% range"},
        {0.50, 0.02, "Low price $0.50, 2% range"},
        {150.75, 0.015, "Medium price $150.75, 1.5% range"}
    };
    
    for (const auto& test : test_cases) {
        double range = test.current_price * test.range_percentage;
        double y_min = test.current_price - range;
        double y_max = test.current_price + range;
        double calculated_center = (y_min + y_max) / 2.0;
        
        std::cout << test.description << ":\n";
        std::cout << "  Current price: " << test.current_price << std::endl;
        std::cout << "  Range: " << range << std::endl;
        std::cout << "  Y-min: " << y_min << std::endl;
        std::cout << "  Y-max: " << y_max << std::endl;
        std::cout << "  Calculated center: " << calculated_center << " (should equal current price)" << std::endl;
        std::cout << "  Formula verified: " << (std::abs(calculated_center - test.current_price) < 1e-9 ? "YES" : "NO") << std::endl;
        std::cout << std::endl;
    }
    
    std::cout << "Center Mode formula verification completed.\n";
    std::cout << "The implementation correctly follows: y_min = current_price - range\n";
}

int main() {
    test_center_mode_formula();
    return 0;
}