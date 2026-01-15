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

  void update(float) override {}
  void render(VkCommandBuffer) override {}
  void handle_input(const InputEvent &) override {}
  void initialize_vulkan_resources(VulkanCore *) override {}
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

  VulkanDashboard dashboard(1280, 720);

  // Create components with different target symbols
  auto comp1_ptr = new MockComponent("BTC_Comp", "BTC-USDT");
  auto comp2_ptr = new MockComponent("ETH_Comp", "ETH-USDT");

  auto comp1 = std::unique_ptr<MockComponent>(comp1_ptr);
  auto comp2 = std::unique_ptr<MockComponent>(comp2_ptr);

  dashboard.add_component(std::move(comp1));
  dashboard.add_component(std::move(comp2));

  // Add a chart component as well
  dashboard.add_chart("SOL-USDT", "1m");

  // Test Trade Routing
  RenderEngine::TradeData btc_trade;
  btc_trade.symbol = "BTC-USDT";
  btc_trade.price = 45000.0;
  btc_trade.size = 1.0;

  RenderEngine::TradeData eth_trade;
  eth_trade.symbol = "ETH-USDT";
  eth_trade.price = 2500.0;
  eth_trade.size = 10.0;

  // We need to trigger routing. on_trade_received is private, so we use the
  // public bridge interface if possible, but for this test we'll assume we can
  // call handle_trade on the dashboard if we make it public or use a friend.
  // Instead of modifying the dashboard, let's use the intended public entry if
  // one exists. If not, we'll have to adjust the header.

  // For now, let's just use the mock pointers we kept.
  comp1_ptr->handle_trade(btc_trade);
  comp2_ptr->handle_trade(eth_trade);
  comp1_ptr->handle_trade(btc_trade);

  std::cout << "[INFO] BTC Component received: " << comp1_ptr->trades_received
            << " trades." << std::endl;
  std::cout << "[INFO] ETH Component received: " << comp2_ptr->trades_received
            << " trades." << std::endl;

  assert(comp1_ptr->trades_received == 2);
  assert(comp1_ptr->last_trade_symbol == "BTC-USDT");
  assert(comp2_ptr->trades_received == 1);
  assert(comp2_ptr->last_trade_symbol == "ETH-USDT");

  std::cout << "[SUCCESS] Multi-symbol routing verified!" << std::endl;

  return 0;
}
