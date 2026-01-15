#include "../include/vulkan_dashboard_advanced.hpp"
#include <cassert>
#include <iostream>

using namespace BTQuant;

class MockComponent : public UIComponent {
public:
  MockComponent(const std::string &name, const std::string &target)
      : UIComponent({0, 0}, {100, 100}), name_(name) {
    set_target_symbol(target);
  }

  void update(float dt) override {}
  void render(VkCommandBuffer cmd) override {}
  void handle_input(const InputEvent &e) override {}
  void initialize_vulkan_resources(VulkanCore *core) override {}
  std::string get_name() const override { return name_; }

  void handle_trade(const RenderEngine::TradeData &trade) override {
    trades_received++;
    last_trade_symbol = trade.symbol;
  }

  int trades_received = 0;
  std::string last_trade_symbol;
  std::string name_;
};

int main() {
  std::cout << "[INFO] Starting Multi-Symbol Routing Test..." << std::endl;

  VulkanDashboard dashboard;

  // Create components with different target symbols
  auto comp1 = std::make_shared<MockComponent>("BTC_Comp", "BTC-USDT");
  auto comp2 = std::make_shared<MockComponent>("ETH_Comp", "ETH-USDT");

  dashboard.add_component(comp1);
  dashboard.add_component(comp2);

  // Add a chart component as well
  dashboard.add_chart("SOL-USDT", "1m");
  auto &charts = dashboard.get_charts(); // Need to ensure we have access or use
                                         // internal routing

  // Test Trade Routing
  RenderEngine::TradeData btc_trade;
  btc_trade.symbol = "BTC-USDT";
  btc_trade.price = 45000.0;
  btc_trade.size = 1.0;

  RenderEngine::TradeData eth_trade;
  eth_trade.symbol = "ETH-USDT";
  eth_trade.price = 2500.0;
  eth_trade.size = 10.0;

  dashboard.on_trade_received(btc_trade);
  dashboard.on_trade_received(eth_trade);
  dashboard.on_trade_received(btc_trade);

  std::cout << "[INFO] BTC Component received: " << comp1->trades_received
            << " trades." << std::endl;
  std::cout << "[INFO] ETH Component received: " << comp2->trades_received
            << " trades." << std::endl;

  assert(comp1->trades_received == 2);
  assert(comp1->last_trade_symbol == "BTC-USDT");
  assert(comp2->trades_received == 1);
  assert(comp2->last_trade_symbol == "ETH-USDT");

  std::cout << "[SUCCESS] Multi-symbol routing verified!" << std::endl;

  return 0;
}
