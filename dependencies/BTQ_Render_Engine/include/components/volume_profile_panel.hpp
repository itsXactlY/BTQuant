#pragma once

#include <imgui.h>

#include <memory>
#include <vector>

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "panel_base.hpp"
#include "theme_manager.hpp"

namespace BTQuant {

// Enum for profile mode
enum class ProfileMode { Step, Right, Left, Custom, Session, Composite, Virgin };

// Struct for profile settings
struct ProfileSettings {
  int vaPercent = 70;         // Value Area percentage
  int tickStep = 1;           // Tick step size
  bool showPOC = true;        // Show Point of Control
  bool showValueArea = true;  // Show Value Area
  int colorScheme = 0;        // Color scheme index
};

/**
 * VolumeProfilePanel - Volume at Price display
 *
 * C++26 Reactive Architecture:
 * - Subscribes to MarketDataProcessor for TRADE notifications
 * - markDirty() called from callback, consumeDirty() in render()
 * - No polling timer - truly event-driven
 *
 * Shows horizontal bars representing volume traded at each price level:
 * - Buy volume (green) on right side
 * - Sell volume (red) on left side (mirrored)
 * - Point of Control (POC) highlighted
 */
class VolumeProfilePanel : public PanelBase {
 public:
  VolumeProfilePanel(const PanelConfig& config, std::shared_ptr<HotSpineDataBridge> bridge,
                     std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

  ~VolumeProfilePanel() override;

  void render_content() override;
  void set_symbol(uint32_t symbol_id, const std::string& symbol_name);

  // Method to render mini histogram overlays on candlestick charts
  void render_mini_histograms_on_candles(ImDrawList* draw_list,
                                         const std::vector<RenderEngine::OHLCVCandle>& candles,
                                         const std::vector<double>& x_coords,
                                         const std::vector<double>& y_coords_high,
                                         const std::vector<double>& y_coords_low);

  // Enhanced method to render step profile histograms on candlesticks
  void render_step_profile_histograms(ImDrawList* draw_list,
                                      const std::vector<RenderEngine::OHLCVCandle>& candles,
                                      const std::vector<double>& x_coords,
                                      const std::vector<double>& y_coords_high,
                                      const std::vector<double>& y_coords_low,
                                      bool show_poc_line = true,
                                      int num_buckets = 8);

  // Method to render step profile histograms specifically for candlestick volume distribution
  // This is the main method for implementing the Step Profile rendering feature
  void render_candle_volume_distribution(ImDrawList* draw_list,
                                        const std::vector<RenderEngine::OHLCVCandle>& candles,
                                        const std::vector<double>& x_coords,
                                        const std::vector<double>& y_coords_high,
                                        const std::vector<double>& y_coords_low,
                                        bool show_poc_line = true,
                                        int num_buckets = 8);

  // Enhanced method to render step profile directly on candles with improved visualization
  void render_step_profile_on_candles(ImDrawList* draw_list,
                                     const std::vector<RenderEngine::OHLCVCandle>& candles,
                                     const std::vector<double>& x_coords,
                                     const std::vector<double>& y_coords_high,
                                     const std::vector<double>& y_coords_low,
                                     bool show_poc_line = true,
                                     int num_buckets_per_candle = 8);

  // Static utility method to render mini histograms directly without creating a panel instance
  // This is more efficient for use in chart panels where we don't need the full panel functionality
  static void render_mini_histograms_direct(ImDrawList* draw_list,
                                           const std::vector<RenderEngine::OHLCVCandle>& candles,
                                           const std::vector<double>& x_coords,
                                           const std::vector<double>& y_coords_high,
                                           const std::vector<double>& y_coords_low,
                                           const std::vector<RenderEngine::TradeData>& trades,
                                           bool show_poc_line = true,
                                           int num_buckets = 8);

  // Static method to render step profile directly on candles with improved visualization
  static void render_step_profile_on_candles_static(ImDrawList* draw_list,
                                                  const std::vector<RenderEngine::OHLCVCandle>& candles,
                                                  const std::vector<double>& x_coords,
                                                  const std::vector<double>& y_coords_high,
                                                  const std::vector<double>& y_coords_low,
                                                  const std::vector<RenderEngine::TradeData>& trades,
                                                  bool show_poc_line = true,
                                                  int num_buckets_per_candle = 8);

  // Enhanced method to render step profile histograms on candlesticks with additional features
  static void render_enhanced_step_profile_on_candles(ImDrawList* draw_list,
                                                    const std::vector<RenderEngine::OHLCVCandle>& candles,
                                                    const std::vector<double>& x_coords,
                                                    const std::vector<double>& y_coords_high,
                                                    const std::vector<double>& y_coords_low,
                                                    const std::vector<RenderEngine::TradeData>& trades,
                                                    bool show_poc_line = true,
                                                    int num_buckets_per_candle = 8,
                                                    float opacity_factor = 1.0f);

  // Main method to implement Step Profile rendering: draw mini histogram overlay on each candlestick bar
  // showing volume distribution for that bar's price range
  void render_step_profile_on_candles_with_volume_distribution(ImDrawList* draw_list,
                                                            const std::vector<RenderEngine::OHLCVCandle>& candles,
                                                            const std::vector<double>& x_coords,
                                                            const std::vector<double>& y_coords_high,
                                                            const std::vector<double>& y_coords_low,
                                                            bool show_poc_line = true,
                                                            int num_buckets_per_candle = 8);

  // Enhanced method to render step profile with additional visualization options
  void render_enhanced_step_profile_with_volume_distribution(ImDrawList* draw_list,
                                                          const std::vector<RenderEngine::OHLCVCandle>& candles,
                                                          const std::vector<double>& x_coords,
                                                          const std::vector<double>& y_coords_high,
                                                          const std::vector<double>& y_coords_low,
                                                          bool show_poc_line = true,
                                                          int num_buckets_per_candle = 8,
                                                          float bar_opacity = 1.0f,
                                                          bool use_transparent_background = false);

  // Specific implementation for Step Profile rendering: draw mini histogram overlay on each candlestick bar
  // showing volume distribution for that bar's price range - this is the main method for the task requirement
  void render_step_profile_histograms_on_candle_bars(ImDrawList* draw_list,
                                                   const std::vector<RenderEngine::OHLCVCandle>& candles,
                                                   const std::vector<double>& x_coords,
                                                   const std::vector<double>& y_coords_high,
                                                   const std::vector<double>& y_coords_low,
                                                   bool show_poc_line = true,
                                                   int num_buckets_per_candle = 8);

  // Enhanced method with additional visualization options for Step Profile rendering
  void render_enhanced_step_profile_histograms_on_candle_bars(ImDrawList* draw_list,
                                                           const std::vector<RenderEngine::OHLCVCandle>& candles,
                                                           const std::vector<double>& x_coords,
                                                           const std::vector<double>& y_coords_high,
                                                           const std::vector<double>& y_coords_low,
                                                           bool show_poc_line = true,
                                                           int num_buckets_per_candle = 8,
                                                           float opacity = 1.0f,
                                                           bool show_labels = false);

  // Public method to render step profile histograms on candle bars - this can be called from other components
  void drawMiniHistogramOverlay(ImDrawList* draw_list,
                              const std::vector<RenderEngine::OHLCVCandle>& candles,
                              const std::vector<double>& x_coords,
                              const std::vector<double>& y_coords_high,
                              const std::vector<double>& y_coords_low,
                              bool show_poc_line = true,
                              int num_buckets_per_candle = 8);

  // Methods for custom profile with mouse drag interaction
  void handleMouseDragInteraction();
  void renderCustomProfileOverlay(ImDrawList* draw_list);
  void calculateProfileForTimeRange(double start_time, double end_time);

  // Helper methods for time range calculations
  double getMinTimeAvailable();
  double getMaxTimeAvailable();
  double getTimeRangeAvailable();

 private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;

  uint32_t symbol_id_ = 0;
  std::string symbol_name_ = "BTC-USDT";

  // Volume profile data
  struct VolumeLevel {
    double price;
    double buy_volume;
    double sell_volume;
    double total_volume;
    int buy_trades;      // Count of buy trades at this price level
    int sell_trades;     // Count of sell trades at this price level
    int total_trades;    // Total count of trades at this price level
    
    VolumeLevel() : price(0.0), buy_volume(0.0), sell_volume(0.0), total_volume(0.0), 
                    buy_trades(0), sell_trades(0), total_trades(0) {}
  };

  // Session profile data
  struct SessionInfo {
    double start_time;
    double end_time;
    std::string session_name;

    SessionInfo(double start, double end, const std::string& name)
        : start_time(start), end_time(end), session_name(name) {}
  };

  struct SessionProfile {
    std::vector<VolumeLevel> volume_profile;
    double poc_price = 0.0;
    double max_volume = 0.0;
    double vah_price = 0.0;
    double val_price = 0.0;
    double start_time = 0.0;
    double end_time = 0.0;
    std::string session_name;

    SessionProfile(const std::string& name = "") : session_name(name) {}
  };

  std::vector<VolumeLevel> volume_profile_;
  std::vector<VolumeLevel> yesterday_volume_profile_;  // Yesterday's profile for comparison overlay
  std::vector<SessionProfile> session_profiles_;       // Multiple session profiles

  // Data structures for composite profile (aggregated multiple days)
  struct DailyVolumeProfile {
    std::vector<VolumeLevel> daily_profile;
    double poc_price = 0.0;
    double max_volume = 0.0;
    double vah_price = 0.0;
    double val_price = 0.0;
    time_t date = 0;  // Date of this profile (for identification)

    DailyVolumeProfile(time_t d) : date(d) {}
  };

  std::vector<DailyVolumeProfile> daily_profiles_;     // Individual daily profiles for composite aggregation
  std::vector<VolumeLevel> composite_volume_profile_;  // Aggregated composite profile
  double composite_poc_price_ = 0.0;   // Composite Point of Control (highest volume price)
  double composite_max_volume_ = 0.0;  // Composite max volume for scaling
  double composite_vah_price_ = 0.0;   // Composite Value Area High
  double composite_val_price_ = 0.0;   // Composite Value Area Low

  double poc_price_ = 0.0;   // Point of Control (highest volume price)
  double max_volume_ = 0.0;  // For scaling bars
  double vah_price_ = 0.0;   // Value Area High
  double val_price_ = 0.0;   // Value Area Low
  double yesterday_poc_price_ = 0.0;   // Yesterday's Point of Control
  double yesterday_max_volume_ = 0.0;  // Yesterday's max volume for scaling
  double yesterday_vah_price_ = 0.0;   // Yesterday's Value Area High
  double yesterday_val_price_ = 0.0;   // Yesterday's Value Area Low

  // Configuration
  static constexpr size_t NUM_PRICE_LEVELS = 20;
  double price_bucket_size_ = 10.0;  // Price range per level

  // Profile settings
  ProfileMode profile_mode_ = ProfileMode::Step;
  ProfileSettings profile_settings_{};

  // Virgin POC tracking
  std::vector<bool> virgin_price_levels_;  // Track which price levels have not been touched
  double virgin_poc_price_ = 0.0;         // Virgin Point of Control
  std::vector<VolumeLevel> virgin_volume_profile_;  // Virgin volume profile

  // Profile anchor markers for Custom Profile mode
  bool use_custom_time_range_ = false;  // Whether to use custom time range
  double custom_start_time_ = 0.0;     // Start time for custom range
  double custom_end_time_ = 0.0;       // End time for custom range
  bool start_time_drag_active_ = false; // Whether start time drag handle is active
  bool end_time_drag_active_ = false;   // Whether end time drag handle is active

  // Session profile settings
  bool use_session_boundaries_ = true;  // Whether to use session boundaries for profile reset
  std::vector<SessionInfo> predefined_sessions_; // Predefined trading sessions
  int current_session_index_ = -1;      // Index of currently displayed session
  bool auto_detect_session_boundaries_ = true; // Whether to auto-detect session boundaries

  // Composite profile settings
  bool use_composite_profile_ = false;  // Whether to use composite profile mode
  int composite_days_count_ = 5;        // Number of days to include in composite profile
  bool auto_update_composite_ = true;   // Whether to automatically update composite profile

  void build_volume_profile();
  void render_volume_bars();
  void render_controls();
  void render_step_profile(const double* xs, const double* ys, const double* neg_ys, int count,
                           double height);
  void render_split_profile(const double* xs, const double* buy_vols, const double* sell_vols,
                           int count, double height);
  void render_yesterday_step_profile(const double* xs, const double* ys, const double* neg_ys,
                                   int count, double height, float opacity);
  void render_yesterday_split_profile(const double* xs, const double* buy_vols, const double* sell_vols,
                                   int count, double height, float opacity);
  void highlight_profile_divergence(ImDrawList* draw_list);  // Highlight zones where profiles diverge significantly
  void calculate_value_area();
  void calculate_value_area_for_session(SessionProfile& session); // Calculate value area for a specific session profile
  void store_current_as_yesterday_profile();  // Store current profile as yesterday's profile
  void calculate_yesterday_value_area();      // Calculate value area for yesterday's profile

  // Composite profile methods
  void build_composite_profile();             // Build composite profile from multiple days
  void add_daily_profile(const std::vector<VolumeLevel>& daily_profile, time_t date); // Add daily profile to composite
  void calculate_composite_value_area();      // Calculate value area for composite profile
  time_t get_date_from_timestamp(double timestamp); // Extract date from timestamp
  void clear_daily_profiles();                // Clear all daily profiles
  void update_composite_profile();            // Update composite profile based on current settings

  // Session profile methods
  void initialize_predefined_sessions();      // Initialize predefined trading sessions
  int get_session_index_for_timestamp(double timestamp); // Get session index for a given timestamp
  bool is_new_session_boundary(double current_timestamp, double previous_timestamp); // Check if new session boundary
  void create_new_session_profile(double start_time, double end_time, const std::string& session_name); // Create new session profile
  void switch_to_session(int session_index);  // Switch to a specific session profile
  void reset_session_profiles();              // Reset all session profiles
  void detect_and_handle_session_boundaries(); // Detect and handle session boundaries

  // Virgin POC methods
  void calculate_virgin_poc();                // Calculate virgin POC from historical data

  // Subscribe to processor notifications
  void subscribe_to_updates();
};

}  // namespace BTQuant
