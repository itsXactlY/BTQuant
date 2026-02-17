#include "../../include/components/volume_profile_panel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>

#include "../../include/components/interaction_manager.hpp"
#include "imgui.h"
#include "implot.h"

namespace BTQuant {

VolumeProfilePanel::VolumeProfilePanel(const PanelConfig& config,
                                       std::shared_ptr<HotSpineDataBridge> bridge,
                                       std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), bridge_(bridge), processor_(processor) {
  volume_profile_.reserve(NUM_PRICE_LEVELS);

  // Initialize predefined trading sessions
  initialize_predefined_sessions();

  // Initialize composite profile settings
  composite_days_count_ = 5;  // Default to 5 days
  auto_update_composite_ = true;

  // Initialize virgin POC tracking
  virgin_price_levels_.resize(NUM_PRICE_LEVELS, true);  // Initially all levels are virgin
  virgin_poc_price_ = 0.0;
  virgin_volume_profile_.reserve(NUM_PRICE_LEVELS);

  // C++26: Subscribe to push notifications instead of polling
  subscribe_to_updates();
}

VolumeProfilePanel::~VolumeProfilePanel() {
  // C++26: Clean unsubscription on destruction
  if (processor_ && subscription_id_ != 0) {
    processor_->unsubscribe(subscription_id_);
  }
}

void VolumeProfilePanel::initialize_predefined_sessions() {
  // Clear existing sessions
  predefined_sessions_.clear();

  // Add common trading sessions (these are examples - in practice, these would come from market
  // data) Using UTC time for standard sessions

  // Asian session: 23:00 - 08:00 UTC
  predefined_sessions_.emplace_back(23 * 3600, 8 * 3600, "Asian");

  // London session: 08:00 - 17:00 UTC
  predefined_sessions_.emplace_back(8 * 3600, 17 * 3600, "London");

  // New York session: 13:00 - 22:00 UTC (overlaps with London)
  predefined_sessions_.emplace_back(13 * 3600, 22 * 3600, "New York");

  // Daily session: 00:00 - 24:00 UTC (midnight reset)
  predefined_sessions_.emplace_back(0, 24 * 3600, "Daily");
}

int VolumeProfilePanel::get_session_index_for_timestamp(double timestamp) {
  if (predefined_sessions_.empty()) {
    return -1;
  }

  // Convert timestamp to time structure to get hour of day
  time_t time_sec = static_cast<time_t>(timestamp);
  struct tm* tm_info = gmtime(&time_sec);

  int hour_of_day = tm_info->tm_hour;
  int seconds_into_day = hour_of_day * 3600 + tm_info->tm_min * 60 + tm_info->tm_sec;

  // Check each predefined session to see if the timestamp falls within it
  for (size_t i = 0; i < predefined_sessions_.size(); ++i) {
    const auto& session = predefined_sessions_[i];

    // Handle sessions that cross midnight (e.g., 23:00 - 08:00)
    if (session.start_time > session.end_time) {
      // Session crosses midnight
      if (seconds_into_day >= session.start_time || seconds_into_day <= session.end_time) {
        return static_cast<int>(i);
      }
    } else {
      // Normal session within same day
      if (seconds_into_day >= session.start_time && seconds_into_day <= session.end_time) {
        return static_cast<int>(i);
      }
    }
  }

  // If no predefined session matches, return -1 (unknown session)
  return -1;
}

bool VolumeProfilePanel::is_new_session_boundary(double current_timestamp,
                                                 double previous_timestamp) {
  if (previous_timestamp == 0.0) {
    // First timestamp, consider it a new session
    return true;
  }

  // Check for daily boundary (UTC 00:00)
  time_t current_time = static_cast<time_t>(current_timestamp);
  time_t previous_time = static_cast<time_t>(previous_timestamp);

  struct tm* current_tm = gmtime(&current_time);
  struct tm* previous_tm = gmtime(&previous_time);

  // Check if we crossed a day boundary (UTC 00:00)
  if (current_tm->tm_year != previous_tm->tm_year || current_tm->tm_yday != previous_tm->tm_yday) {
    return true;
  }

  // If using predefined sessions, check if we moved to a different session
  if (!predefined_sessions_.empty()) {
    int current_session_idx = get_session_index_for_timestamp(current_timestamp);
    int previous_session_idx = get_session_index_for_timestamp(previous_timestamp);

    return current_session_idx != previous_session_idx;
  }

  // Default: no session boundary detected
  return false;
}

void VolumeProfilePanel::create_new_session_profile(double start_time, double end_time,
                                                    const std::string& session_name) {
  SessionProfile new_session(session_name);
  new_session.start_time = start_time;
  new_session.end_time = end_time;

  session_profiles_.push_back(new_session);
}

void VolumeProfilePanel::switch_to_session(int session_index) {
  if (session_index >= 0 && session_index < static_cast<int>(session_profiles_.size())) {
    current_session_index_ = session_index;

    // Copy the selected session profile to the main profile for rendering
    const auto& selected_session = session_profiles_[session_index];
    volume_profile_ = selected_session.volume_profile;
    poc_price_ = selected_session.poc_price;
    max_volume_ = selected_session.max_volume;
    vah_price_ = selected_session.vah_price;
    val_price_ = selected_session.val_price;

    // If POC wasn't calculated yet, calculate it now
    if (selected_session.poc_price == 0.0 && !selected_session.volume_profile.empty()) {
      double max_total_volume = 0.0;
      for (const auto& level : selected_session.volume_profile) {
        double total = level.buy_volume + level.sell_volume;
        if (total > max_total_volume) {
          max_total_volume = total;
          poc_price_ = level.price;
        }
      }
    }
  }
}

void VolumeProfilePanel::reset_session_profiles() {
  session_profiles_.clear();
  current_session_index_ = -1;
}

void VolumeProfilePanel::detect_and_handle_session_boundaries() {
  if (!processor_ || symbol_id_ == 0) return;

  auto analytics = processor_->getSymbolAnalytics(symbol_id_);
  auto trades = analytics.recent_trades;

  if (trades.empty()) return;

  // Sort trades by timestamp to process in chronological order
  std::sort(trades.begin(), trades.end(),
            [](const RenderEngine::TradeData& a, const RenderEngine::TradeData& b) {
              return a.timestamp < b.timestamp;
            });

  // Process trades to detect session boundaries
  double previous_timestamp = 0.0;
  int last_session_index = -1;

  for (const auto& trade : trades) {
    int current_session_index = get_session_index_for_timestamp(trade.timestamp);

    // Check if we crossed a session boundary
    if (is_new_session_boundary(trade.timestamp, previous_timestamp)) {
      // Create a new session profile if needed
      if (auto_detect_session_boundaries_) {
        std::string session_name = "Session_" + std::to_string(session_profiles_.size() + 1);

        // If we have predefined sessions, use the session name
        if (current_session_index >= 0 &&
            current_session_index < static_cast<int>(predefined_sessions_.size())) {
          session_name = predefined_sessions_[current_session_index].session_name;
        }

        create_new_session_profile(trade.timestamp, 0.0, session_name);
        last_session_index = static_cast<int>(session_profiles_.size() - 1);
      }
    }

    // Add the trade to the appropriate session profile
    if (last_session_index >= 0 &&
        last_session_index < static_cast<int>(session_profiles_.size())) {
      // Add trade to the current session profile
      auto& current_session = session_profiles_[last_session_index];

      // Find the appropriate price bucket for this trade
      double min_price = std::numeric_limits<double>::max();
      double max_price = std::numeric_limits<double>::lowest();

      // Determine the price range for this session profile
      if (current_session.volume_profile.empty()) {
        // First trade in this session, initialize the profile
        double range = 100.0;  // Default range
        double price_bucket_size = range / NUM_PRICE_LEVELS;

        current_session.volume_profile.resize(NUM_PRICE_LEVELS);
        for (size_t i = 0; i < NUM_PRICE_LEVELS; ++i) {
          current_session.volume_profile[i].price =
              trade.price - (range / 2.0) + (i + 0.5) * price_bucket_size;
          current_session.volume_profile[i].buy_volume = 0;
          current_session.volume_profile[i].sell_volume = 0;
          current_session.volume_profile[i].total_volume = 0;
          current_session.volume_profile[i].buy_trades = 0;
          current_session.volume_profile[i].sell_trades = 0;
          current_session.volume_profile[i].total_trades = 0;
        }
      }

      // Find the appropriate bucket for this trade
      double bucket_size =
          current_session.volume_profile[1].price - current_session.volume_profile[0].price;
      size_t bucket_idx = static_cast<size_t>(
          (trade.price - current_session.volume_profile[0].price) / bucket_size);

      if (bucket_idx < current_session.volume_profile.size()) {
        if (trade.is_buy) {
          current_session.volume_profile[bucket_idx].buy_volume += trade.size;
          current_session.volume_profile[bucket_idx].buy_trades++;
        } else {
          current_session.volume_profile[bucket_idx].sell_volume += trade.size;
          current_session.volume_profile[bucket_idx].sell_trades++;
        }
        current_session.volume_profile[bucket_idx].total_volume += trade.size;
        current_session.volume_profile[bucket_idx].total_trades++;

        // Update max volume if needed
        double max_vol_in_bucket = std::max(current_session.volume_profile[bucket_idx].buy_volume,
                                            current_session.volume_profile[bucket_idx].sell_volume);
        if (max_vol_in_bucket > current_session.max_volume) {
          current_session.max_volume = max_vol_in_bucket;
        }
      }
    }

    previous_timestamp = trade.timestamp;
  }

  // Calculate value areas for all session profiles
  for (auto& session : session_profiles_) {
    calculate_value_area_for_session(session);
  }

  // Update the main profile to show the current session
  if (current_session_index_ >= 0 &&
      current_session_index_ < static_cast<int>(session_profiles_.size())) {
    switch_to_session(current_session_index_);
  } else if (!session_profiles_.empty()) {
    // Show the most recent session by default
    switch_to_session(static_cast<int>(session_profiles_.size() - 1));
  }
}

void VolumeProfilePanel::subscribe_to_updates() {
  if (!processor_ || symbol_id_ == 0) return;

  // Unsubscribe from previous symbol if any
  if (subscription_id_ != 0) {
    processor_->unsubscribe(subscription_id_);
  }

  // Subscribe to TRADE notifications for this symbol
  // Callback sets dirty flag - will be processed in next render()
  subscription_id_ = processor_->subscribe(
      symbol_id_, RenderEngine::NotificationType::TRADE,
      [this](uint32_t /*symbol_id*/, RenderEngine::NotificationType /*type*/) {
        // Thread-safe: atomic flag set from worker thread
        this->markDirty();
      });
}

void VolumeProfilePanel::render_content() {
  begin_panel_window();

  if (!is_visible()) {
    end_panel_window();
    return;
  }

  render_panel_header();
  render_controls();
  ImGui::Separator();

  // Handle mouse drag interaction for custom profile creation
  handleMouseDragInteraction();

  // C++26 Reactive: Only rebuild when new data arrives or first load
  if (processor_ && symbol_id_ != 0) {
    // consumeDirty() returns true initially or when notified
    if (consumeDirty() || volume_profile_.empty()) {
      build_volume_profile();
    }

    // For composite profile, update if auto-update is enabled
    if (profile_mode_ == ProfileMode::Composite && auto_update_composite_) {
      update_composite_profile();
    }
  }

  render_volume_bars();

  end_panel_window();
}

void VolumeProfilePanel::set_symbol(uint32_t symbol_id, const std::string& symbol_name) {
  symbol_id_ = symbol_id;
  symbol_name_ = symbol_name;
  volume_profile_.clear();
  max_volume_ = 0.0;
  poc_price_ = 0.0;

  // Re-subscribe to new symbol
  subscribe_to_updates();
  markDirty();  // Force immediate build
}

void VolumeProfilePanel::build_volume_profile() {
  auto analytics = processor_->getSymbolAnalytics(symbol_id_);
  auto trades = analytics.recent_trades;  // Copy to potentially filter

  // Handle session-based profiles if in Session mode
  if (profile_mode_ == ProfileMode::Session) {
    detect_and_handle_session_boundaries();
    return;
  }

  // Handle composite profile if in Composite mode
  if (profile_mode_ == ProfileMode::Composite) {
    build_composite_profile();
    return;
  }

  // Check if we need to store current profile as yesterday's profile
  // This could be based on time of day or other criteria
  // For now, we'll implement a simple mechanism to store when a new day starts
  if (!trades.empty() && !volume_profile_.empty()) {
    // Get the most recent trade timestamp
    double latest_timestamp = trades.back().timestamp;

    // Compare with the oldest trade in the current profile to determine if we're in a new period
    // For simplicity, we'll store the current profile as yesterday's profile when we detect a
    // significant time gap This would typically be handled by checking if the current time is a new
    // day compared to the last update

    // For demonstration purposes, we'll add a flag to indicate when to store the profile
    // In a real implementation, this would be based on actual date/time logic
    static bool first_build_after_startup = true;
    static double last_update_timestamp = 0.0;

    // Check if enough time has passed to consider it a new day/session
    // For now, we'll use a simple threshold (e.g., 24 hours = 86400 seconds)
    const double DAY_THRESHOLD = 86400.0;  // 24 hours in seconds

    if (!first_build_after_startup && (latest_timestamp - last_update_timestamp) > DAY_THRESHOLD) {
      // Store the current profile as yesterday's profile
      store_current_as_yesterday_profile();
    }

    last_update_timestamp = latest_timestamp;
    first_build_after_startup = false;
  }

  // Filter trades based on custom time range if enabled and in Custom Profile mode
  if (use_custom_time_range_ && profile_mode_ == ProfileMode::Custom && !trades.empty()) {
    std::vector<RenderEngine::TradeData> filtered_trades;

    for (const auto& trade : trades) {
      if (trade.timestamp >= custom_start_time_ && trade.timestamp <= custom_end_time_) {
        filtered_trades.push_back(trade);
      }
    }

    trades = std::move(filtered_trades);
  }

  if (trades.empty()) return;

  // Check if we have an existing volume profile and if the new trades fit in the current range
  bool needs_rebuild = false;
  double min_price = std::numeric_limits<double>::max();
  double max_price = std::numeric_limits<double>::lowest();

  // Find the range of new trades
  for (const auto& trade : trades) {
    min_price = std::min(min_price, trade.price);
    max_price = std::max(max_price, trade.price);
  }

  // If we have an existing profile, check if new trades fall outside the current range
  if (!volume_profile_.empty()) {
    double current_min_price = volume_profile_.front().price - (price_bucket_size_ / 2.0);
    double current_max_price = volume_profile_.back().price + (price_bucket_size_ / 2.0);

    // Check if any new trade falls outside the current range
    if (min_price < current_min_price || max_price > current_max_price) {
      needs_rebuild = true;
    } else {
      // Trades fit in existing range, update incrementally
      for (const auto& trade : trades) {
        // Calculate which bucket this trade belongs to based on existing bucket size
        size_t bucket_index =
            static_cast<size_t>((trade.price - current_min_price) / price_bucket_size_);

        if (bucket_index < volume_profile_.size()) {
          if (trade.is_buy) {
            volume_profile_[bucket_index].buy_volume += trade.size;
            volume_profile_[bucket_index].buy_trades++;
          } else {
            volume_profile_[bucket_index].sell_volume += trade.size;
            volume_profile_[bucket_index].sell_trades++;
          }
          volume_profile_[bucket_index].total_volume += trade.size;
          volume_profile_[bucket_index].total_trades++;

          // Update max volume if needed
          double max_vol_in_bucket = std::max(volume_profile_[bucket_index].buy_volume,
                                              volume_profile_[bucket_index].sell_volume);
          if (max_vol_in_bucket > max_volume_) {
            max_volume_ = max_vol_in_bucket;
          }

          // Mark this price level as not virgin anymore
          if (bucket_index < virgin_price_levels_.size()) {
            virgin_price_levels_[bucket_index] = false;
          }
        }
      }

      // Update POC
      double poc_volume = 0;
      for (const auto& level : volume_profile_) {
        double total = level.buy_volume + level.sell_volume;
        if (total > poc_volume) {
          poc_volume = total;
          poc_price_ = level.price;
        }
      }

      // Ensure poc_price_ is always set to the price level with the highest total volume
      if (poc_volume > 0) {
        // Double check to make sure we have the correct POC
        double current_max_volume = 0;
        for (const auto& level : volume_profile_) {
          double total = level.buy_volume + level.sell_volume;
          if (total > current_max_volume) {
            current_max_volume = total;
            poc_price_ = level.price;
          }
        }
      }

      // Recalculate Value Area after incremental update
      calculate_value_area();

      return;  // Exit early since we've updated incrementally
    }
  } else {
    needs_rebuild = true;  // First time building, so we need to rebuild
  }

  if (needs_rebuild || volume_profile_.empty()) {
    if (max_price <= min_price) return;

    // Compute bucket size
    double range = max_price - min_price;
    price_bucket_size_ = range / NUM_PRICE_LEVELS;
    if (price_bucket_size_ <= 0) price_bucket_size_ = 1.0;

    // Reset profile
    volume_profile_.clear();
    volume_profile_.resize(NUM_PRICE_LEVELS);

    // Also resize virgin profile if needed
    if (virgin_volume_profile_.size() != NUM_PRICE_LEVELS) {
      virgin_volume_profile_.clear();
      virgin_volume_profile_.resize(NUM_PRICE_LEVELS);
    }

    for (size_t i = 0; i < NUM_PRICE_LEVELS; ++i) {
      volume_profile_[i].price = min_price + (i + 0.5) * price_bucket_size_;
      volume_profile_[i].buy_volume = 0;
      volume_profile_[i].sell_volume = 0;
      volume_profile_[i].total_volume = 0;
      volume_profile_[i].buy_trades = 0;
      volume_profile_[i].sell_trades = 0;
      volume_profile_[i].total_trades = 0;

      // Initialize virgin profile with same price levels
      virgin_volume_profile_[i].price = volume_profile_[i].price;
      virgin_volume_profile_[i].buy_volume = 0;
      virgin_volume_profile_[i].sell_volume = 0;
      virgin_volume_profile_[i].total_volume = 0;
      virgin_volume_profile_[i].buy_trades = 0;
      virgin_volume_profile_[i].sell_trades = 0;
      virgin_volume_profile_[i].total_trades = 0;
    }

    // Aggregate trades into buckets and update virgin status
    for (const auto& trade : trades) {
      size_t bucket = static_cast<size_t>((trade.price - min_price) / price_bucket_size_);
      bucket = std::min(bucket, NUM_PRICE_LEVELS - 1);

      // Mark this price level as not virgin anymore
      if (bucket < virgin_price_levels_.size()) {
        virgin_price_levels_[bucket] = false;
      }

      if (trade.is_buy) {
        volume_profile_[bucket].buy_volume += trade.size;
        volume_profile_[bucket].buy_trades++;
      } else {
        volume_profile_[bucket].sell_volume += trade.size;
        volume_profile_[bucket].sell_trades++;
      }
      volume_profile_[bucket].total_volume += trade.size;
      volume_profile_[bucket].total_trades++;
    }

    // Find POC and max volume
    max_volume_ = 0;
    poc_price_ = volume_profile_[0].price;
    double poc_volume = 0;

    for (const auto& level : volume_profile_) {
      double total = level.buy_volume + level.sell_volume;
      max_volume_ = std::max(max_volume_, std::max(level.buy_volume, level.sell_volume));
      if (total > poc_volume) {
        poc_volume = total;
        poc_price_ = level.price;
      }
    }

    // Ensure poc_price_ is always set to the price level with the highest total volume
    if (poc_volume > 0) {
      // Double check to make sure we have the correct POC
      double current_max_volume = 0;
      for (const auto& level : volume_profile_) {
        double total = level.buy_volume + level.sell_volume;
        if (total > current_max_volume) {
          current_max_volume = total;
          poc_price_ = level.price;
        }
      }
    }

    // Calculate Virgin POC - find the highest virgin price level with potential significance
    calculate_virgin_poc();

    // Calculate Value Area
    calculate_value_area();
  }
}

void VolumeProfilePanel::render_controls() {
  if (ImGui::Button("Reset View")) {
    ImPlot::SetNextAxesToFit();
  }
  ImGui::SameLine();
  ImGui::Text("Symbol: %s", symbol_name_.c_str());
  ImGui::SameLine();
  ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "| POC: %.4f", poc_price_);

  // Add level count control
  ImGui::SameLine();
  ImGui::Text("| Levels: %zu", volume_profile_.size());

  // Add profile mode selection
  ImGui::SameLine();
  const char* profile_modes[] = {"Step",    "Right",     "Left",  "Custom",
                                 "Session", "Composite", "Virgin"};
  ImGui::Combo("Profile Mode", reinterpret_cast<int*>(&profile_mode_), profile_modes, 7);

  // Add VA% control
  ImGui::SameLine();
  ImGui::SliderInt("VA%", &profile_settings_.vaPercent, 50, 99);

  // Add controls for Custom Profile range when Custom mode is selected
  if (profile_mode_ == ProfileMode::Custom) {
    ImGui::SameLine();
    ImGui::Checkbox("Use Time Range", &use_custom_time_range_);

    if (use_custom_time_range_) {
      ImGui::Separator();

      // Initialize start and end times if not already set
      if (custom_start_time_ == 0.0 && custom_end_time_ == 0.0) {
        // Get current time range from available data
        auto analytics = processor_->getSymbolAnalytics(symbol_id_);
        const auto& trades = analytics.recent_trades;

        if (!trades.empty()) {
          // Find min and max timestamps
          double min_time = trades[0].timestamp;
          double max_time = trades[0].timestamp;

          for (const auto& trade : trades) {
            if (trade.timestamp < min_time) min_time = trade.timestamp;
            if (trade.timestamp > max_time) max_time = trade.timestamp;
          }

          custom_start_time_ = min_time;
          custom_end_time_ = max_time;
        }
      }

      // Show controls for start and end times
      ImGui::Text("Time Range:");
      ImGui::SameLine();
      ImGui::Text("Start: %.2f", custom_start_time_);
      ImGui::SameLine();
      ImGui::Text("End: %.2f", custom_end_time_);

      // Buttons to reset to full range
      if (ImGui::Button("Reset Time Range")) {
        auto analytics = processor_->getSymbolAnalytics(symbol_id_);
        const auto& trades = analytics.recent_trades;

        if (!trades.empty()) {
          // Find min and max timestamps
          double min_time = trades[0].timestamp;
          double max_time = trades[0].timestamp;

          for (const auto& trade : trades) {
            if (trade.timestamp < min_time) min_time = trade.timestamp;
            if (trade.timestamp > max_time) max_time = trade.timestamp;
          }

          custom_start_time_ = min_time;
          custom_end_time_ = max_time;
        }
      }
    }
  }

  // Add controls for Session Profile mode when Session mode is selected
  if (profile_mode_ == ProfileMode::Session) {
    ImGui::SameLine();
    ImGui::Checkbox("Auto-detect Sessions", &auto_detect_session_boundaries_);

    // Show session selection dropdown if we have multiple sessions
    if (!session_profiles_.empty()) {
      ImGui::Separator();
      ImGui::Text("Session Profiles:");

      // Create combo box to select which session to display
      std::vector<const char*> session_names;
      for (const auto& session : session_profiles_) {
        session_names.push_back(session.session_name.c_str());
      }

      if (!session_names.empty()) {
        ImGui::Combo("Select Session", &current_session_index_, session_names.data(),
                     static_cast<int>(session_names.size()));

        // Update the main profile to show the selected session
        if (current_session_index_ >= 0 &&
            current_session_index_ < static_cast<int>(session_profiles_.size())) {
          switch_to_session(current_session_index_);
        }
      }
    }

    // Show button to reset all session profiles
    if (ImGui::Button("Reset Session Profiles")) {
      reset_session_profiles();
      volume_profile_.clear();
      max_volume_ = 0.0;
      poc_price_ = 0.0;
    }
  }

  // Add controls for Composite Profile mode when Composite mode is selected
  if (profile_mode_ == ProfileMode::Composite) {
    ImGui::Separator();
    ImGui::Text("Composite Profile Settings:");

    // Number of days to include in composite profile
    ImGui::Text("Days to Include:");
    ImGui::SameLine();
    ImGui::SliderInt("##DaysCount", &composite_days_count_, 1, 30);
    ImGui::SameLine();
    ImGui::Text("%d days", composite_days_count_);

    // Auto-update option
    ImGui::Checkbox("Auto Update Composite", &auto_update_composite_);

    // Manual update button
    if (ImGui::Button("Update Composite Profile")) {
      update_composite_profile();
    }

    // Show number of daily profiles collected
    ImGui::Text("Daily Profiles Collected: %zu", daily_profiles_.size());

    // Clear daily profiles button
    if (ImGui::Button("Clear Daily Profiles")) {
      clear_daily_profiles();
      composite_volume_profile_.clear();
      composite_poc_price_ = 0.0;
      composite_max_volume_ = 0.0;
      composite_vah_price_ = 0.0;
      composite_val_price_ = 0.0;
    }
  }

  // Add controls for Virgin Profile mode when Virgin mode is selected
  if (profile_mode_ == ProfileMode::Virgin) {
    ImGui::Separator();
    ImGui::Text("Virgin Profile Settings:");

    // Show number of virgin price levels remaining
    int virgin_count = 0;
    for (bool is_virgin : virgin_price_levels_) {
      if (is_virgin) virgin_count++;
    }

    ImGui::Text("Untouched Price Levels: %d", virgin_count);
    ImGui::Text("Virgin POC: %.4f", virgin_poc_price_);

    // Button to reset virgin tracking (mark all levels as virgin again)
    if (ImGui::Button("Reset Virgin Tracking")) {
      std::fill(virgin_price_levels_.begin(), virgin_price_levels_.end(), true);
      virgin_poc_price_ = 0.0;
    }
  }

  // Display VAH and VAL if available
  double display_vah = vah_price_;
  double display_val = val_price_;
  double display_poc = poc_price_;

  if (profile_mode_ == ProfileMode::Composite) {
    display_vah = composite_vah_price_;
    display_val = composite_val_price_;
    display_poc = composite_poc_price_;
  } else if (profile_mode_ == ProfileMode::Virgin) {
    // For Virgin profile, we show the virgin POC but don't typically show VAH/VAL
    display_poc = virgin_poc_price_;
    // Keep default values for VAH and VAL (they won't be displayed for Virgin mode)
  }

  ImGui::SameLine();
  if (profile_mode_ != ProfileMode::Virgin) {
    ImGui::Text("| VAH: %.4f", display_vah);
    ImGui::SameLine();
    ImGui::Text("| VAL: %.4f", display_val);
  } else {
    // For Virgin mode, show Virgin POC instead of regular POC
    ImGui::Text("| Virgin POC: %.4f", display_poc);
  }

  // Add profile statistics panel
  if (ImGui::CollapsingHeader("Profile Statistics")) {
    ImGui::Indent();

    // Display POC price
    double stats_poc_price = poc_price_;
    double stats_vah_price = vah_price_;
    double stats_val_price = val_price_;
    const auto& stats_volume_profile =
        (profile_mode_ == ProfileMode::Composite) ? composite_volume_profile_ : volume_profile_;

    if (profile_mode_ == ProfileMode::Composite) {
      stats_poc_price = composite_poc_price_;
      stats_vah_price = composite_vah_price_;
      stats_val_price = composite_val_price_;
    } else if (profile_mode_ == ProfileMode::Virgin) {
      // For Virgin profile, we'll show the virgin POC but use the regular volume profile for stats
      stats_poc_price = virgin_poc_price_;
      // Virgin profile doesn't typically have VAH/VAL, so we'll leave those as is
    }

    ImGui::Text("POC Price: %.4f", stats_poc_price);

    // Display VAH price
    ImGui::Text("VAH Price: %.4f", stats_vah_price);

    // Display VAL price
    ImGui::Text("VAL Price: %.4f", stats_val_price);

    // Calculate and display total volume in value area
    double total_volume_in_value_area = 0.0;
    double total_volume_above_poc = 0.0;
    double total_volume_below_poc = 0.0;
    double poc_volume_level = 0.0;  // Volume at POC level

    if (!stats_volume_profile.empty()) {
      for (const auto& level : stats_volume_profile) {
        // Check if this price level is within the value area
        if (level.price >= stats_val_price && level.price <= stats_vah_price) {
          total_volume_in_value_area += level.total_volume;
        }

        // Calculate volume above and below POC
        if (level.price > stats_poc_price) {
          total_volume_above_poc += level.total_volume;
        } else if (level.price < stats_poc_price) {
          total_volume_below_poc += level.total_volume;
        } else if (level.price == stats_poc_price) {
          poc_volume_level = level.total_volume;
        }
      }
    }

    ImGui::Text("Total Volume in Value Area: %.2f", total_volume_in_value_area);
    ImGui::Text("POC Volume Level: %.2f", poc_volume_level);

    // Calculate and display percentage of volume above POC
    double total_volume_above_below_poc = total_volume_above_poc + total_volume_below_poc;
    if (total_volume_above_below_poc > 0) {
      double percentage_above_poc = (total_volume_above_poc / total_volume_above_below_poc) * 100.0;
      ImGui::Text("Percentage of Volume Above POC: %.2f%%", percentage_above_poc);

      double percentage_below_poc = (total_volume_below_poc / total_volume_above_below_poc) * 100.0;
      ImGui::Text("Percentage of Volume Below POC: %.2f%%", percentage_below_poc);
    } else {
      ImGui::Text("Percentage of Volume Above POC: N/A");
      ImGui::Text("Percentage of Volume Below POC: N/A");
    }

    // Calculate and display additional statistics
    double total_market_volume = total_volume_above_poc + total_volume_below_poc + poc_volume_level;
    ImGui::Text("Total Market Volume: %.2f", total_market_volume);

    if (total_market_volume > 0) {
      double poc_volume_percentage = (poc_volume_level / total_market_volume) * 100.0;
      ImGui::Text("POC Volume Percentage: %.2f%%", poc_volume_percentage);
    } else {
      ImGui::Text("POC Volume Percentage: N/A");
    }

    ImGui::Unindent();
  }

  // Add detailed volume profile table
  if (ImGui::CollapsingHeader("Volume Profile Detail Table")) {
    if (ImGui::BeginTable(
            "##VolumeProfileDetailTable", 6,
            ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg)) {
      ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthFixed, 80.0f);
      ImGui::TableSetupColumn("% of Volume at POC", ImGuiTableColumnFlags_WidthFixed, 120.0f);
      ImGui::TableSetupColumn("Total Trades", ImGuiTableColumnFlags_WidthFixed, 100.0f);
      ImGui::TableSetupColumn("Buy/Sell Ratio", ImGuiTableColumnFlags_WidthFixed, 100.0f);
      ImGui::TableSetupColumn("Relative Volume", ImGuiTableColumnFlags_WidthFixed, 120.0f);
      ImGui::TableSetupColumn("Total Volume", ImGuiTableColumnFlags_WidthFixed, 100.0f);
      ImGui::TableHeadersRow();

      // Determine which profile to use based on the current mode
      const auto& current_volume_profile =
          (profile_mode_ == ProfileMode::Composite) ? composite_volume_profile_ : volume_profile_;
      double current_poc_price =
          (profile_mode_ == ProfileMode::Composite) ? composite_poc_price_ : poc_price_;
      double current_price_bucket_size = price_bucket_size_;

      // Calculate average volume across all levels for relative volume calculation
      double total_volume_all_levels = 0.0;
      int valid_levels_count = 0;
      for (const auto& level : current_volume_profile) {
        if (level.total_volume > 0) {
          total_volume_all_levels += level.total_volume;
          valid_levels_count++;
        }
      }
      double avg_volume =
          (valid_levels_count > 0) ? total_volume_all_levels / valid_levels_count : 0.0;

      // Calculate total volume at POC for percentage calculation
      double total_volume_at_poc = 0.0;
      for (const auto& level : current_volume_profile) {
        if (std::abs(level.price - current_poc_price) <
            current_price_bucket_size / 2.0) {  // Check if this level is near POC
          total_volume_at_poc = level.total_volume;
          break;
        }
      }

      // If we didn't find the exact POC level, find the level with the highest volume
      if (total_volume_at_poc == 0.0 && !current_volume_profile.empty()) {
        double max_volume = 0.0;
        for (const auto& level : current_volume_profile) {
          double total = level.buy_volume + level.sell_volume;
          if (total > max_volume) {
            max_volume = total;
            total_volume_at_poc = level.total_volume;
          }
        }
      }

      // Add rows for each volume level
      for (const auto& level : current_volume_profile) {
        ImGui::TableNextRow();

        // Price column
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%.4f", level.price);

        // % of Volume at POC column
        ImGui::TableSetColumnIndex(1);
        if (total_volume_at_poc > 0) {
          double percentage_of_poc = (level.total_volume / total_volume_at_poc) * 100.0;
          ImGui::Text("%.2f%%", percentage_of_poc);
        } else {
          ImGui::Text("N/A");
        }

        // Total Trades column
        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%d", level.total_trades);

        // Buy/Sell Ratio column
        ImGui::TableSetColumnIndex(3);
        if (level.sell_volume > 0) {
          double buy_sell_ratio = level.buy_volume / level.sell_volume;
          ImGui::Text("%.2f", buy_sell_ratio);
        } else if (level.buy_volume > 0) {
          ImGui::Text("Inf");  // Infinite ratio if no sells
        } else {
          ImGui::Text("N/A");  // No trades at this level
        }

        // Relative Volume column (Volume / Avg Volume)
        ImGui::TableSetColumnIndex(4);
        if (avg_volume > 0) {
          double relative_volume = level.total_volume / avg_volume;
          ImGui::Text("%.2f", relative_volume);
        } else {
          ImGui::Text("N/A");
        }

        // Total Volume column
        ImGui::TableSetColumnIndex(5);
        ImGui::Text("%.2f", level.total_volume);
      }

      ImGui::EndTable();
    }
  }
}

void VolumeProfilePanel::render_volume_bars() {
  if (volume_profile_.empty() || max_volume_ <= 0) {
    ImGui::Text("No volume data available");
    return;
  }

  ImVec2 region = ImGui::GetContentRegionAvail();
  if (region.x < 100 || region.y < 100) return;

  // Prepare data for ImPlot horizontal bars
  std::vector<double> prices;
  std::vector<double> buy_volumes;
  std::vector<double> sell_volumes;

  prices.reserve(volume_profile_.size());
  buy_volumes.reserve(volume_profile_.size());
  sell_volumes.reserve(volume_profile_.size());

  for (const auto& level : volume_profile_) {
    prices.push_back(level.price);
    buy_volumes.push_back(level.buy_volume);
    sell_volumes.push_back(-level.sell_volume);  // Negative for left side
  }

  // Unique plot ID per panel instance to avoid ImGui ID conflicts
  char plot_id[64];
  snprintf(plot_id, sizeof(plot_id), "##VolumeProfile_%s", config_.title.c_str());

  if (ImPlot::BeginPlot(plot_id, region, ImPlotFlags_NoTitle | ImPlotFlags_NoLegend)) {
    ImPlot::SetupAxes("Volume", "Price", ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_Invert,
                      ImPlotAxisFlags_AutoFit);

    // Bar height based on price bucket size
    double bar_height = price_bucket_size_ * 0.8;

    // For Right and Left profiles, recalculate value area based on visible range during rendering
    double local_vah_price = vah_price_;
    double local_val_price = val_price_;

    if (profile_mode_ == ProfileMode::Right || profile_mode_ == ProfileMode::Left) {
      // Get the plot limits to determine what's currently visible
      ImPlotRect plot_limits = ImPlot::GetPlotLimits();

      // Create a temporary vector of visible levels with their volumes
      std::vector<std::pair<double, double>> visible_levels;  // {price, total_volume}
      double total_volume = 0.0;

      for (const auto& level : volume_profile_) {
        if (level.price >= plot_limits.Y.Min && level.price <= plot_limits.Y.Max) {
          visible_levels.push_back({level.price, level.total_volume});
          total_volume += level.total_volume;
        }
      }

      if (!visible_levels.empty() && total_volume > 0) {
        // Target volume for value area (based on profile_settings_.vaPercent % of total volume)
        double target_volume =
            (static_cast<double>(profile_settings_.vaPercent) / 100.0) * total_volume;

        // Sort visible levels by volume in descending order to find POC
        std::vector<std::pair<double, double>> sorted_levels = visible_levels;
        std::sort(sorted_levels.begin(), sorted_levels.end(),
                  [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                    return a.second > b.second;  // Sort by volume descending
                  });

        // Find the POC (Point of Control) - the price level with highest volume among visible
        // levels
        double poc_price = sorted_levels[0].first;

        // Sort visible levels by price to make expansion easier
        std::sort(visible_levels.begin(), visible_levels.end());

        // Find the index of POC in the price-sorted array
        size_t poc_idx_sorted = 0;
        for (size_t i = 0; i < visible_levels.size(); ++i) {
          if (std::abs(visible_levels[i].first - poc_price) <
              0.000001) {  // Use epsilon comparison for floating point
            poc_idx_sorted = i;
            break;
          }
        }

        // Expand from POC outward to capture the required volume, centered around POC
        size_t start_idx = poc_idx_sorted;
        size_t end_idx = poc_idx_sorted;
        double current_volume = visible_levels[poc_idx_sorted].second;

        // Expand upward (higher prices) and downward (lower prices) alternately
        // until we reach the target volume, keeping the expansion balanced around POC
        while (current_volume < target_volume) {
          // Decide whether to expand up or down
          bool expand_up = false;
          bool expand_down = false;

          // Check if we can expand in each direction
          if (start_idx > 0) expand_down = true;
          if (end_idx < visible_levels.size() - 1) expand_up = true;

          // If we can't expand in either direction, break
          if (!expand_up && !expand_down) break;

          // If we can only expand in one direction, do that
          if (!expand_up && expand_down) {
            start_idx--;
            current_volume += visible_levels[start_idx].second;
          } else if (expand_up && !expand_down) {
            end_idx++;
            current_volume += visible_levels[end_idx].second;
          } else {
            // We can expand in both directions - try to keep it centered around POC
            // Calculate potential volumes for each direction
            double vol_up = visible_levels[end_idx + 1].second;
            double vol_down = visible_levels[start_idx - 1].second;

            // To center around POC, we should try to balance the expansion
            // If both sides have similar volume, expand the side that currently has a smaller range
            size_t current_upper_range = end_idx - poc_idx_sorted;
            size_t current_lower_range = poc_idx_sorted - start_idx;

            if (current_lower_range < current_upper_range) {
              // Current lower range is smaller, expand downward to balance
              start_idx--;
              current_volume += visible_levels[start_idx].second;
            } else if (current_upper_range < current_lower_range) {
              // Current upper range is smaller, expand upward to balance
              end_idx++;
              current_volume += visible_levels[end_idx].second;
            } else {
              // Ranges are equal, expand toward the side with more volume to capture more volume
              // efficiently
              if (vol_up >= vol_down) {
                end_idx++;
                current_volume += visible_levels[end_idx].second;
              } else {
                start_idx--;
                current_volume += visible_levels[start_idx].second;
              }
            }
          }

          // If we've captured enough volume, break
          if (current_volume >= target_volume) break;
        }

        // Set the local VAH and VAL prices for this render pass
        local_vah_price = visible_levels[end_idx].first;
        local_val_price = visible_levels[start_idx].first;
      }
    }

    // Draw Value Area overlay if we have valid VAH and VAL
    // Use local values for Right/Left profiles, global values for others
    double overlay_vah = (profile_mode_ == ProfileMode::Right || profile_mode_ == ProfileMode::Left)
                             ? local_vah_price
                             : vah_price_;
    double overlay_val = (profile_mode_ == ProfileMode::Right || profile_mode_ == ProfileMode::Left)
                             ? local_val_price
                             : val_price_;

    if (overlay_vah > 0 && overlay_val > 0 && overlay_vah >= overlay_val) {
      ImDrawList* draw_list = ImPlot::GetPlotDrawList();

      // For Right/Left profiles, we need to calculate the appropriate x-coordinates based on the
      // profile type
      double left_x_limit, right_x_limit;

      if (profile_mode_ == ProfileMode::Right || profile_mode_ == ProfileMode::Left) {
        // For Right/Left profiles: find the maximum volume to determine the appropriate range
        double max_total_volume = 0.0;
        for (const auto& level : volume_profile_) {
          double total_vol = level.buy_volume + level.sell_volume;
          if (total_vol > max_total_volume) {
            max_total_volume = total_vol;
          }
        }
        if (max_total_volume <= 0) max_total_volume = max_volume_;
        if (max_total_volume <= 0) max_total_volume = 1.0;

        if (profile_mode_ == ProfileMode::Right) {
          // For Right profile: overlay spans from 0 to max volume (full range of the profile)
          left_x_limit = 0.0;
          right_x_limit = max_total_volume;
        } else if (profile_mode_ == ProfileMode::Left) {
          // For Left profile: overlay spans from 0 to max volume (full range of the profile)
          left_x_limit = 0.0;
          right_x_limit = max_total_volume;
        } else {
          // For other profile modes, use the full range
          left_x_limit = -max_volume_;
          right_x_limit = max_volume_;
        }
      } else {
        // For other profile modes, use the full range
        left_x_limit = -max_volume_;
        right_x_limit = max_volume_;
      }

      // Convert value area prices to pixel coordinates
      // For the overlay, we want to draw a horizontal band from overlay_val to overlay_vah
      ImVec2 top_left = ImPlot::PlotToPixels(left_x_limit, overlay_vah);
      ImVec2 bottom_right = ImPlot::PlotToPixels(right_x_limit, overlay_val);

      // Draw semi-transparent rectangle for value area - horizontal band spanning the volume range
      // The overlay should cover the entire horizontal span of the profile within the VAH/VAL range
      draw_list->AddRectFilled(
          top_left, bottom_right,
          IM_COL32(138, 43, 226,
                   60));  // Semi-transparent purple (reduced opacity for better visibility)
    }

    // Render based on profile mode
    switch (profile_mode_) {
      case ProfileMode::Step:
        render_step_profile(prices.data(), buy_volumes.data(), sell_volumes.data(),
                            static_cast<int>(prices.size()), bar_height);
        break;
      case ProfileMode::Right: {
        // Right Profile: Create a single aggregated histogram of all visible trades anchored to
        // right edge with horizontal bars extending left Get the plot limits to determine what's
        // currently visible
        ImPlotRect plot_limits = ImPlot::GetPlotLimits();  // This gets the current visible range

        // Aggregate all visible trades into a single total volume
        double total_visible_buy_volume = 0.0;
        double total_visible_sell_volume = 0.0;
        double total_visible_volume = 0.0;

        for (size_t i = 0; i < buy_volumes.size(); ++i) {
          // Only include if the price level is within the visible range
          if (prices[i] >= plot_limits.Y.Min && prices[i] <= plot_limits.Y.Max) {
            double buy_vol = std::abs(buy_volumes[i]);
            double sell_vol = std::abs(sell_volumes[i]);
            double total_vol = buy_vol + sell_vol;

            if (total_vol > 0) {
              total_visible_buy_volume += buy_vol;
              total_visible_sell_volume += sell_vol;
              total_visible_volume += total_vol;
            }
          }
        }

        // If no visible volume, nothing to draw
        if (total_visible_volume <= 0) {
          break;
        }

        // Calculate the rightmost x-coordinate in plot space (this will be our anchor)
        // Use the maximum volume across the entire dataset for consistent scaling
        double max_total_volume = 0.0;
        for (size_t i = 0; i < buy_volumes.size(); ++i) {
          double total_vol = std::abs(buy_volumes[i]) + std::abs(sell_volumes[i]);
          if (total_vol > max_total_volume) {
            max_total_volume = total_vol;
          }
        }

        // If no volume data, use fallback
        if (max_total_volume <= 0) max_total_volume = max_volume_;
        if (max_total_volume <= 0) max_total_volume = 1.0;  // Ultimate fallback

        // Anchor to the right edge
        double right_anchor = max_total_volume;

        // Draw a single horizontal bar representing the total volume of all visible trades
        // The bar extends left from the right anchor, with width proportional to total volume
        ImDrawList* draw_list = ImPlot::GetPlotDrawList();

        // Calculate the left extent based on total visible volume
        double bar_left_extent =
            right_anchor - (total_visible_volume / max_total_volume) * max_total_volume;

        // Determine the vertical range of the visible area (entire visible Y range)
        float bar_top = ImPlot::PlotToPixels(0, plot_limits.Y.Max).y;     // Top of visible area
        float bar_bottom = ImPlot::PlotToPixels(0, plot_limits.Y.Min).y;  // Bottom of visible area
        float bar_right =
            ImPlot::PlotToPixels(right_anchor, (plot_limits.Y.Max + plot_limits.Y.Min) / 2)
                .x;  // Right edge of chart
        float bar_left =
            ImPlot::PlotToPixels(bar_left_extent, (plot_limits.Y.Max + plot_limits.Y.Min) / 2)
                .x;  // Left extent of the bar

        // Determine color based on whether buy or sell volume dominates in the visible range
        ImU32 color;
        if (total_visible_buy_volume >= total_visible_sell_volume) {
          // More buy volume - use green with intensity based on dominance
          float dominance =
              (total_visible_volume > 0)
                  ? static_cast<float>((total_visible_buy_volume - total_visible_sell_volume) /
                                       total_visible_volume)
                  : 0.0f;
          int green = 200 + static_cast<int>(55 * dominance);  // Vary from 200 to 255
          int red = 26 - static_cast<int>(26 * dominance);     // Vary from 26 to 0
          color = IM_COL32(red, green, 26, 179);               // Green dominant
        } else {
          // More sell volume - use red with intensity based on dominance
          float dominance =
              (total_visible_volume > 0)
                  ? static_cast<float>((total_visible_sell_volume - total_visible_buy_volume) /
                                       total_visible_volume)
                  : 0.0f;
          int red = 200 + static_cast<int>(55 * dominance);   // Vary from 200 to 255
          int green = 26 - static_cast<int>(26 * dominance);  // Vary from 26 to 0
          color = IM_COL32(red, green, 26, 179);              // Red dominant
        }

        // Draw the aggregated horizontal bar extending left from the right anchor
        draw_list->AddRectFilled(ImVec2(bar_left, bar_top), ImVec2(bar_right, bar_bottom), color);

        // Add a subtle border for better visibility
        draw_list->AddRect(ImVec2(bar_left, bar_top), ImVec2(bar_right, bar_bottom),
                           IM_COL32(0, 0, 0, 100), 0.0f, 0, 1.0f);

        // Draw POC line for Right profile mode based on the overall POC (not recalculated for
        // visible range)
        if (poc_price_ > 0) {
          // Only draw POC line if it's within the visible range
          if (poc_price_ >= plot_limits.Y.Min && poc_price_ <= plot_limits.Y.Max) {
            double poc_line_x[2] = {bar_left_extent,
                                    right_anchor};  // From left extent to right anchor
            double poc_line_y[2] = {poc_price_, poc_price_};
            // ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.8f, 0.0f, 1.0f));
            ImPlot::PlotLine("POC", poc_line_x, poc_line_y, 2);
            ImPlot::PopStyleColor();
          }
        }
        break;
      }
      case ProfileMode::Left: {
        // Left Profile: Same as Right but anchored to left edge, bars extending right
        // Get the plot limits to determine what's currently visible
        ImPlotRect plot_limits = ImPlot::GetPlotLimits();  // This gets the current visible range

        // Aggregate all visible trades into a single total volume
        double total_visible_buy_volume = 0.0;
        double total_visible_sell_volume = 0.0;
        double total_visible_volume = 0.0;

        for (size_t i = 0; i < buy_volumes.size(); ++i) {
          // Only include if the price level is within the visible range
          if (prices[i] >= plot_limits.Y.Min && prices[i] <= plot_limits.Y.Max) {
            double buy_vol = std::abs(buy_volumes[i]);
            double sell_vol = std::abs(sell_volumes[i]);
            double total_vol = buy_vol + sell_vol;

            if (total_vol > 0) {
              total_visible_buy_volume += buy_vol;
              total_visible_sell_volume += sell_vol;
              total_visible_volume += total_vol;
            }
          }
        }

        // If no visible volume, nothing to draw
        if (total_visible_volume <= 0) {
          break;
        }

        // Calculate the leftmost x-coordinate in plot space (this will be our anchor)
        // Use the maximum volume across the entire dataset for consistent scaling
        double max_total_volume = 0.0;
        for (size_t i = 0; i < buy_volumes.size(); ++i) {
          double total_vol = std::abs(buy_volumes[i]) + std::abs(sell_volumes[i]);
          if (total_vol > max_total_volume) {
            max_total_volume = total_vol;
          }
        }

        // If no volume data, use fallback
        if (max_total_volume <= 0) max_total_volume = max_volume_;
        if (max_total_volume <= 0) max_total_volume = 1.0;  // Ultimate fallback

        // Anchor to the left edge
        double left_anchor = 0.0;

        // Draw a single horizontal bar representing the total volume of all visible trades
        // The bar extends right from the left anchor, with width proportional to total volume
        ImDrawList* draw_list = ImPlot::GetPlotDrawList();

        // Calculate the right extent based on total visible volume
        double bar_right_extent =
            left_anchor + (total_visible_volume / max_total_volume) * max_total_volume;

        // Determine the vertical range of the visible area (entire visible Y range)
        float bar_top = ImPlot::PlotToPixels(0, plot_limits.Y.Min)
                            .y;  // Top of visible area (note: Y axis is inverted in plots)
        float bar_bottom = ImPlot::PlotToPixels(0, plot_limits.Y.Max)
                               .y;  // Bottom of visible area (note: Y axis is inverted in plots)
        float bar_left =
            ImPlot::PlotToPixels(left_anchor, (plot_limits.Y.Max + plot_limits.Y.Min) / 2)
                .x;  // Left edge of chart
        float bar_right =
            ImPlot::PlotToPixels(bar_right_extent, (plot_limits.Y.Max + plot_limits.Y.Min) / 2)
                .x;  // Right extent of the bar

        // Determine color based on whether buy or sell volume dominates in the visible range
        ImU32 color;
        if (total_visible_buy_volume >= total_visible_sell_volume) {
          // More buy volume - use green with intensity based on dominance
          float dominance =
              (total_visible_volume > 0)
                  ? static_cast<float>((total_visible_buy_volume - total_visible_sell_volume) /
                                       total_visible_volume)
                  : 0.0f;
          int green = 200 + static_cast<int>(55 * dominance);  // Vary from 200 to 255
          int red = 26 - static_cast<int>(26 * dominance);     // Vary from 26 to 0
          color = IM_COL32(red, green, 26, 179);               // Green dominant
        } else {
          // More sell volume - use red with intensity based on dominance
          float dominance =
              (total_visible_volume > 0)
                  ? static_cast<float>((total_visible_sell_volume - total_visible_buy_volume) /
                                       total_visible_volume)
                  : 0.0f;
          int red = 200 + static_cast<int>(55 * dominance);   // Vary from 200 to 255
          int green = 26 - static_cast<int>(26 * dominance);  // Vary from 26 to 0
          color = IM_COL32(red, green, 26, 179);              // Red dominant
        }

        // Draw the aggregated horizontal bar extending right from the left anchor
        draw_list->AddRectFilled(ImVec2(bar_left, bar_top), ImVec2(bar_right, bar_bottom), color);

        // Add a subtle border for better visibility
        draw_list->AddRect(ImVec2(bar_left, bar_top), ImVec2(bar_right, bar_bottom),
                           IM_COL32(0, 0, 0, 100), 0.0f, 0, 1.0f);

        // Draw POC line for Left profile mode based on the overall POC (not recalculated for
        // visible range)
        if (poc_price_ > 0) {
          // Only draw POC line if it's within the visible range
          if (poc_price_ >= plot_limits.Y.Min && poc_price_ <= plot_limits.Y.Max) {
            double poc_line_x[2] = {left_anchor,
                                    bar_right_extent};  // From left anchor to right extent
            double poc_line_y[2] = {poc_price_, poc_price_};
            // ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.8f, 0.0f, 1.0f));
            ImPlot::PlotLine("POC", poc_line_x, poc_line_y, 2);
            ImPlot::PopStyleColor();
          }
        }
        break;
      }
      case ProfileMode::Composite: {
        // Composite Profile: Render the aggregated profile from multiple days
        // Use the composite_volume_profile_ data instead of the current volume_profile_

        if (composite_volume_profile_.empty()) {
          // If no composite profile exists, build it
          build_composite_profile();
        }

        if (!composite_volume_profile_.empty()) {
          // Prepare data for ImPlot horizontal bars using composite profile
          std::vector<double> comp_prices;
          std::vector<double> comp_buy_volumes;
          std::vector<double> comp_sell_volumes;

          comp_prices.reserve(composite_volume_profile_.size());
          comp_buy_volumes.reserve(composite_volume_profile_.size());
          comp_sell_volumes.reserve(composite_volume_profile_.size());

          for (const auto& level : composite_volume_profile_) {
            comp_prices.push_back(level.price);
            comp_buy_volumes.push_back(level.buy_volume);
            comp_sell_volumes.push_back(-level.sell_volume);  // Negative for left side
          }

          // Render based on the selected sub-mode for composite (we'll use split profile by
          // default)
          render_split_profile(comp_prices.data(), comp_buy_volumes.data(),
                               comp_sell_volumes.data(), static_cast<int>(comp_prices.size()),
                               bar_height);

          // Update the main profile variables to show composite values in UI
          poc_price_ = composite_poc_price_;
          max_volume_ = composite_max_volume_;
          vah_price_ = composite_vah_price_;
          val_price_ = composite_val_price_;
        }
        break;
      }
      case ProfileMode::Virgin: {
        // Virgin Profile: Render only the untouched/virgin price levels
        // This highlights price levels that have not been tested yet

        if (!volume_profile_.empty()) {
          // Prepare data for ImPlot horizontal bars using only virgin price levels
          std::vector<double> virgin_prices;
          std::vector<double> virgin_buy_volumes;
          std::vector<double> virgin_sell_volumes;

          // Only include virgin price levels (those that haven't been touched)
          for (size_t i = 0; i < volume_profile_.size() && i < virgin_price_levels_.size(); ++i) {
            if (virgin_price_levels_[i]) {  // Only include virgin levels
              virgin_prices.push_back(volume_profile_[i].price);

              // For virgin levels, we show the potential volume that could accumulate
              // This could be based on historical patterns or simply highlight the level
              virgin_buy_volumes.push_back(volume_profile_[i].buy_volume *
                                           0.1);  // Reduced volume for visual distinction
              virgin_sell_volumes.push_back(-volume_profile_[i].sell_volume *
                                            0.1);  // Negative for left side
            }
          }

          if (!virgin_prices.empty()) {
            // Render the virgin profile with a different appearance
            render_split_profile(virgin_prices.data(), virgin_buy_volumes.data(),
                                 virgin_sell_volumes.data(), static_cast<int>(virgin_prices.size()),
                                 bar_height * 0.8);  // Slightly thinner bars

            // Update the main profile variables to show virgin values in UI
            poc_price_ = virgin_poc_price_;
          } else {
            // If no virgin levels remain, show a message
            ImGui::Text("No virgin price levels remaining - all have been tested");
          }
        }
        break;
      }
      case ProfileMode::Custom:
      default:
        // Custom Profile Mode: Split bars with buy volume on left (green) and sell volume on right
        // (red) Each bar is centered at zero with buy volume extending left (negative) and sell
        // volume extending right (positive)
        render_split_profile(prices.data(), buy_volumes.data(), sell_volumes.data(),
                             static_cast<int>(prices.size()), bar_height);
        break;
    }

    // POC line - draw differently based on profile mode for consistency
    // For Step Profile mode, POC line is drawn in render_step_profile function
    if (profile_mode_ == ProfileMode::Composite) {
      // For composite profile, use composite POC if available
      if (composite_poc_price_ > 0) {
        if (profile_mode_ != ProfileMode::Step) {
          // For other modes, use ImPlot's PlotLine
          double poc_line_x[2] = {-composite_max_volume_, composite_max_volume_};
          double poc_line_y[2] = {composite_poc_price_, composite_poc_price_};
          // ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.8f, 0.0f, 1.0f));
          ImPlot::PlotLine("POC", poc_line_x, poc_line_y, 2);
          ImPlot::PopStyleColor();
        }
      }
    } else if (profile_mode_ == ProfileMode::Virgin) {
      // For virgin profile, use virgin POC if available
      if (virgin_poc_price_ > 0) {
        if (profile_mode_ != ProfileMode::Step) {
          // For other modes, use ImPlot's PlotLine
          double poc_line_x[2] = {-max_volume_, max_volume_};
          double poc_line_y[2] = {virgin_poc_price_, virgin_poc_price_};
          // ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.0f, 1.0f, 1.0f, 1.0f)); // Cyan color
          // for virgin POC
          ImPlot::PlotLine("Virgin POC", poc_line_x, poc_line_y, 2);
          ImPlot::PopStyleColor();
        }
      }
    } else {
      // For other modes, use the regular POC
      if (poc_price_ > 0) {
        if (profile_mode_ != ProfileMode::Step) {
          // For other modes, use ImPlot's PlotLine
          double poc_line_x[2] = {-max_volume_, max_volume_};
          double poc_line_y[2] = {poc_price_, poc_price_};
          // ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.8f, 0.0f, 1.0f));
          ImPlot::PlotLine("POC", poc_line_x, poc_line_y, 2);
          ImPlot::PopStyleColor();
        }
      }
    }

    // VAH and VAL lines - use local values for Right/Left profiles
    double display_vah_price = vah_price_;
    double display_val_price = val_price_;
    double display_max_volume = max_volume_;

    if (profile_mode_ == ProfileMode::Composite) {
      // For composite profile, use composite values
      display_vah_price = composite_vah_price_;
      display_val_price = composite_val_price_;
      display_max_volume = composite_max_volume_;
    } else if (profile_mode_ == ProfileMode::Virgin) {
      // For virgin profile, we don't typically show VAH/VAL since it's about untouched levels
      // So we'll keep the default values or set them to 0
      display_vah_price = 0.0;  // Don't show VAH for virgin profile
      display_val_price = 0.0;  // Don't show VAL for virgin profile
      display_max_volume = max_volume_;
    }

    if (display_vah_price > 0) {
      // For Right/Left profiles, calculate appropriate x-coordinates based on the profile type
      double vah_line_x[2];
      if (profile_mode_ == ProfileMode::Right || profile_mode_ == ProfileMode::Left) {
        // For Right/Left profiles: find the maximum volume to determine the appropriate range
        double max_total_volume = 0.0;
        const auto& current_profile =
            (profile_mode_ == ProfileMode::Composite) ? composite_volume_profile_ : volume_profile_;
        for (const auto& level : current_profile) {
          double total_vol = level.buy_volume + level.sell_volume;
          if (total_vol > max_total_volume) {
            max_total_volume = total_vol;
          }
        }
        if (max_total_volume <= 0) max_total_volume = display_max_volume;
        if (max_total_volume <= 0) max_total_volume = 1.0;

        if (profile_mode_ == ProfileMode::Right) {
          // For Right profile: line spans from 0 to max volume (full range of the profile)
          vah_line_x[0] = 0.0;
          vah_line_x[1] = max_total_volume;
        } else if (profile_mode_ == ProfileMode::Left) {
          // For Left profile: line spans from 0 to max volume (full range of the profile)
          vah_line_x[0] = 0.0;
          vah_line_x[1] = max_total_volume;
        } else {
          // For other profile modes, use the full range
          vah_line_x[0] = -display_max_volume;
          vah_line_x[1] = display_max_volume;
        }
      } else {
        // For other profile modes, use the full range
        vah_line_x[0] = -display_max_volume;
        vah_line_x[1] = display_max_volume;
      }

      double vah_line_y[2] = {display_vah_price, display_vah_price};

      // Use brighter color for VAH line to make it more visible
      // ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.0f, 1.0f, 1.0f, 1.0f));  // Bright cyan
      ImPlot::PlotLine("VAH", vah_line_x, vah_line_y, 2);
      ImPlot::PopStyleColor();

      // Add label for VAH line in Right/Left profile modes
      if (profile_mode_ == ProfileMode::Right || profile_mode_ == ProfileMode::Left) {
        ImDrawList* draw_list = ImPlot::GetPlotDrawList();
        ImVec2 plot_size = ImPlot::GetPlotSize();
        ImVec2 plot_pos = ImPlot::GetPlotPos();

        // Position label appropriately based on profile mode to avoid overlap with bars
        ImVec2 vah_pos;
        if (profile_mode_ == ProfileMode::Right) {
          // For Right profile, place label on the left side to avoid overlapping with
          // right-anchored bars
          vah_pos = ImVec2(plot_pos.x + 10, ImPlot::PlotToPixels(0, display_vah_price).y - 10);
        } else {
          // For Left profile, place label on the right side to avoid overlapping with left-anchored
          // bars
          vah_pos = ImVec2(plot_pos.x + plot_size.x - 80,
                           ImPlot::PlotToPixels(0, display_vah_price).y - 10);
        }

        // Draw a more prominent label with background
        char vah_label[32];
        snprintf(vah_label, sizeof(vah_label), "VAH: %.4f", display_vah_price);

        // Calculate text size for background rectangle
        ImVec2 text_size = ImGui::CalcTextSize(vah_label);

        // Draw background rectangle for better visibility
        ImVec2 bg_min = ImVec2(vah_pos.x - 2, vah_pos.y - 2);
        ImVec2 bg_max = ImVec2(vah_pos.x + text_size.x + 2, vah_pos.y + text_size.y);
        draw_list->AddRectFilled(bg_min, bg_max, IM_COL32(0, 0, 0, 200));  // Dark background

        // Draw border around label
        draw_list->AddRect(bg_min, bg_max, IM_COL32(0, 255, 255, 200));  // Cyan border

        // Draw the text
        draw_list->AddText(vah_pos, IM_COL32(0, 255, 255, 255), vah_label);
      }
    }

    if (display_val_price > 0) {
      // For Right/Left profiles, calculate appropriate x-coordinates based on the profile type
      double val_line_x[2];
      if (profile_mode_ == ProfileMode::Right || profile_mode_ == ProfileMode::Left) {
        // For Right/Left profiles: find the maximum volume to determine the appropriate range
        double max_total_volume = 0.0;
        const auto& current_profile =
            (profile_mode_ == ProfileMode::Composite) ? composite_volume_profile_ : volume_profile_;
        for (const auto& level : current_profile) {
          double total_vol = level.buy_volume + level.sell_volume;
          if (total_vol > max_total_volume) {
            max_total_volume = total_vol;
          }
        }
        if (max_total_volume <= 0) max_total_volume = display_max_volume;
        if (max_total_volume <= 0) max_total_volume = 1.0;

        if (profile_mode_ == ProfileMode::Right) {
          // For Right profile: line spans from 0 to max volume (full range of the profile)
          val_line_x[0] = 0.0;
          val_line_x[1] = max_total_volume;
        } else if (profile_mode_ == ProfileMode::Left) {
          // For Left profile: line spans from 0 to max volume (full range of the profile)
          val_line_x[0] = 0.0;
          val_line_x[1] = max_total_volume;
        } else {
          // For other profile modes, use the full range
          val_line_x[0] = -display_max_volume;
          val_line_x[1] = display_max_volume;
        }
      } else {
        // For other profile modes, use the full range
        val_line_x[0] = -display_max_volume;
        val_line_x[1] = display_max_volume;
      }

      double val_line_y[2] = {display_val_price, display_val_price};

      // Use brighter color for VAL line to make it more visible
      // ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.0f, 1.0f, 1.0f, 1.0f));  // Bright cyan
      ImPlot::PlotLine("VAL", val_line_x, val_line_y, 2);
      ImPlot::PopStyleColor();

      // Add label for VAL line in Right/Left profile modes
      if (profile_mode_ == ProfileMode::Right || profile_mode_ == ProfileMode::Left) {
        ImDrawList* draw_list = ImPlot::GetPlotDrawList();
        ImVec2 plot_size = ImPlot::GetPlotSize();
        ImVec2 plot_pos = ImPlot::GetPlotPos();

        // Position label appropriately based on profile mode to avoid overlap with bars
        ImVec2 val_pos;
        if (profile_mode_ == ProfileMode::Right) {
          // For Right profile, place label on the left side to avoid overlapping with
          // right-anchored bars
          val_pos = ImVec2(plot_pos.x + 10, ImPlot::PlotToPixels(0, display_val_price).y - 10);
        } else {
          // For Left profile, place label on the right side to avoid overlapping with left-anchored
          // bars
          val_pos = ImVec2(plot_pos.x + plot_size.x - 80,
                           ImPlot::PlotToPixels(0, display_val_price).y - 10);
        }

        // Draw a more prominent label with background
        char val_label[32];
        snprintf(val_label, sizeof(val_label), "VAL: %.4f", display_val_price);

        // Calculate text size for background rectangle
        ImVec2 text_size = ImGui::CalcTextSize(val_label);

        // Draw background rectangle for better visibility
        ImVec2 bg_min = ImVec2(val_pos.x - 2, val_pos.y - 2);
        ImVec2 bg_max = ImVec2(val_pos.x + text_size.x + 2, val_pos.y + text_size.y);
        draw_list->AddRectFilled(bg_min, bg_max, IM_COL32(0, 0, 0, 200));  // Dark background

        // Draw border around label
        draw_list->AddRect(bg_min, bg_max, IM_COL32(0, 255, 255, 200));  // Cyan border

        // Draw the text
        draw_list->AddText(val_pos, IM_COL32(0, 255, 255, 255), val_label);
      }
    }

    // Add VWAP line if available
    auto analytics = processor_->getSymbolAnalytics(symbol_id_);
    if (analytics.vwap > 0) {
      double vwap_line_x[2] = {-max_volume_, max_volume_};
      double vwap_line_y[2] = {analytics.vwap, analytics.vwap};
      // ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.0f, 0.5f, 1.0f, 1.0f));
      ImPlot::PlotLine("VWAP", vwap_line_x, vwap_line_y, 2);
      ImPlot::PopStyleColor();
    }

    // Add profile anchor markers for Custom Profile mode
    // Note: In the volume profile view, we can't directly show time on the X-axis since it
    // represents volume Instead, we'll show indicators of the selected time range in a way that
    // makes sense for the volume profile
    if (profile_mode_ == ProfileMode::Custom && use_custom_time_range_) {
      ImDrawList* draw_list = ImPlot::GetPlotDrawList();

      // Get plot limits to determine positioning
      ImPlotRect plot_limits = ImPlot::GetPlotLimits();

      // Get the actual plot position and size in screen coordinates
      ImVec2 plot_pos = ImPlot::GetPlotPos();
      ImVec2 plot_size = ImPlot::GetPlotSize();

      // Draw time range indicators at the top of the plot
      // Calculate relative position based on time range (0 to 1)
      double time_range = (custom_end_time_ - custom_start_time_) > 0
                              ? (custom_end_time_ - custom_start_time_)
                              : 1.0;

      // Draw a horizontal indicator bar showing the time range at the top of the plot
      ImVec2 top_left = ImVec2(plot_pos.x, plot_pos.y);
      ImVec2 bottom_right = ImVec2(plot_pos.x + plot_size.x, plot_pos.y + plot_size.y);

      // Draw a semi-transparent bar to indicate the time range selection context
      draw_list->AddRectFilled(ImVec2(top_left.x, top_left.y),
                               ImVec2(bottom_right.x, top_left.y + 25),
                               IM_COL32(100, 100, 100, 100));  // Gray bar at top

      // Draw start and end time markers with handles
      // Since we can't directly map time to X-axis (which represents volume),
      // we'll show the time range info as text labels at the top
      char start_time_text[64];
      char end_time_text[64];
      snprintf(start_time_text, sizeof(start_time_text), "Start: %.0f", custom_start_time_);
      snprintf(end_time_text, sizeof(end_time_text), "End: %.0f", custom_end_time_);

      ImVec2 start_text_size = ImGui::CalcTextSize(start_time_text);
      ImVec2 end_text_size = ImGui::CalcTextSize(end_time_text);

      // Position text at the top corners
      ImVec2 start_text_pos = ImVec2(top_left.x + 5, top_left.y + 5);
      ImVec2 end_text_pos = ImVec2(bottom_right.x - end_text_size.x - 5, top_left.y + 5);

      draw_list->AddText(start_text_pos, IM_COL32(0, 255, 0, 255), start_time_text);
      draw_list->AddText(end_text_pos, IM_COL32(255, 0, 0, 255), end_time_text);

      // Calculate positions for the time range indicators based on the proportion of the time range
      float start_pos_x =
          plot_pos.x +
          ((custom_start_time_ - getMinTimeAvailable()) / getTimeRangeAvailable()) * plot_size.x;
      float end_pos_x =
          plot_pos.x +
          ((custom_end_time_ - getMinTimeAvailable()) / getTimeRangeAvailable()) * plot_size.x;

      // Clamp positions to plot boundaries
      start_pos_x = std::clamp(start_pos_x, plot_pos.x, plot_pos.x + plot_size.x);
      end_pos_x = std::clamp(end_pos_x, plot_pos.x, plot_pos.x + plot_size.x);

      // Add small vertical lines at the top to indicate range boundaries
      ImVec2 start_line_top = ImVec2(start_pos_x, top_left.y);
      ImVec2 start_line_bottom = ImVec2(start_pos_x, top_left.y + 20);
      ImVec2 end_line_top = ImVec2(end_pos_x, top_left.y);
      ImVec2 end_line_bottom = ImVec2(end_pos_x, top_left.y + 20);

      draw_list->AddLine(start_line_top, start_line_bottom, IM_COL32(0, 255, 0, 200), 2.0f);
      draw_list->AddLine(end_line_top, end_line_bottom, IM_COL32(255, 0, 0, 200), 2.0f);

      // Add draggable handles for the time range at the top of the plot
      ImVec2 start_handle_pos = ImVec2(start_pos_x, top_left.y + 10);
      ImVec2 end_handle_pos = ImVec2(end_pos_x, top_left.y + 10);

      ImVec2 handle_size = ImVec2(10.0f, 15.0f);
      ImVec2 start_handle_tl =
          ImVec2(start_handle_pos.x - handle_size.x / 2, start_handle_pos.y - handle_size.y / 2);
      ImVec2 start_handle_br =
          ImVec2(start_handle_pos.x + handle_size.x / 2, start_handle_pos.y + handle_size.y / 2);
      ImVec2 end_handle_tl =
          ImVec2(end_handle_pos.x - handle_size.x / 2, end_handle_pos.y - handle_size.y / 2);
      ImVec2 end_handle_br =
          ImVec2(end_handle_pos.x + handle_size.x / 2, end_handle_pos.y + handle_size.y / 2);

      // Draw the handles
      draw_list->AddRectFilled(start_handle_tl, start_handle_br,
                               IM_COL32(0, 255, 0, 200));  // Green handle for start
      draw_list->AddRect(start_handle_tl, start_handle_br,
                         IM_COL32(255, 255, 255, 255));  // White border
      draw_list->AddRectFilled(end_handle_tl, end_handle_br,
                               IM_COL32(255, 0, 0, 200));  // Red handle for end
      draw_list->AddRect(end_handle_tl, end_handle_br,
                         IM_COL32(255, 255, 255, 255));  // White border

      // Handle dragging for start time handle
      ImGui::SetCursorScreenPos(start_handle_tl);
      if (ImGui::InvisibleButton("start_time_handle", handle_size)) {
        // Button clicked but not dragged
      }
      if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        start_time_drag_active_ = true;
        // Calculate relative mouse movement to adjust start time
        ImVec2 mouse_pos = ImGui::GetMousePos();
        float mouse_x = mouse_pos.x;

        // Convert screen position back to time value
        double new_time = getMinTimeAvailable() +
                          (double)(mouse_x - plot_pos.x) / plot_size.x * getTimeRangeAvailable();

        // Constrain to valid range (ensure start_time < end_time)
        if (new_time < custom_end_time_) {
          custom_start_time_ = new_time;
          markDirty();  // Trigger rebuild with new time range
        }
      } else if (start_time_drag_active_ && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        start_time_drag_active_ = false;
      }

      // Handle dragging for end time handle
      ImGui::SetCursorScreenPos(end_handle_tl);
      if (ImGui::InvisibleButton("end_time_handle", handle_size)) {
        // Button clicked but not dragged
      }
      if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        end_time_drag_active_ = true;
        // Calculate relative mouse movement to adjust end time
        ImVec2 mouse_pos = ImGui::GetMousePos();
        float mouse_x = mouse_pos.x;

        // Convert screen position back to time value
        double new_time = getMinTimeAvailable() +
                          (double)(mouse_x - plot_pos.x) / plot_size.x * getTimeRangeAvailable();

        // Constrain to valid range (ensure end_time > start_time)
        if (new_time > custom_start_time_) {
          custom_end_time_ = new_time;
          markDirty();  // Trigger rebuild with new time range
        }
      } else if (end_time_drag_active_ && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        end_time_drag_active_ = false;
      }
    }

    // Render yesterday's profile for comparison overlay
    // Only render if we have yesterday's data and the user wants to see the comparison
    if (!yesterday_volume_profile_.empty() && yesterday_max_volume_ > 0) {
      // Prepare data for yesterday's profile
      std::vector<double> yesterday_prices;
      std::vector<double> yesterday_buy_volumes;
      std::vector<double> yesterday_sell_volumes;

      yesterday_prices.reserve(yesterday_volume_profile_.size());
      yesterday_buy_volumes.reserve(yesterday_volume_profile_.size());
      yesterday_sell_volumes.reserve(yesterday_volume_profile_.size());

      for (const auto& level : yesterday_volume_profile_) {
        yesterday_prices.push_back(level.price);
        yesterday_buy_volumes.push_back(level.buy_volume);
        yesterday_sell_volumes.push_back(-level.sell_volume);  // Negative for left side
      }

      // Render yesterday's profile based on the same profile mode as today's profile
      // But with reduced opacity (30%)
      switch (profile_mode_) {
        case ProfileMode::Step:
          // Render yesterday's step profile with reduced opacity
          render_yesterday_step_profile(
              yesterday_prices.data(), yesterday_buy_volumes.data(), yesterday_sell_volumes.data(),
              static_cast<int>(yesterday_prices.size()), bar_height, 0.3f);  // 30% opacity
          break;
        case ProfileMode::Right: {
          // Right Profile: Create a single aggregated histogram of yesterday's trades
          ImPlotRect plot_limits = ImPlot::GetPlotLimits();  // This gets the current visible range

          // Aggregate all visible trades from yesterday into a single total volume
          double total_visible_buy_volume = 0.0;
          double total_visible_sell_volume = 0.0;
          double total_visible_volume = 0.0;

          for (size_t i = 0; i < yesterday_buy_volumes.size(); ++i) {
            // Only include if the price level is within the visible range
            if (yesterday_prices[i] >= plot_limits.Y.Min &&
                yesterday_prices[i] <= plot_limits.Y.Max) {
              double buy_vol = std::abs(yesterday_buy_volumes[i]);
              double sell_vol = std::abs(yesterday_sell_volumes[i]);
              double total_vol = buy_vol + sell_vol;

              if (total_vol > 0) {
                total_visible_buy_volume += buy_vol;
                total_visible_sell_volume += sell_vol;
                total_visible_volume += total_vol;
              }
            }
          }

          // If no visible volume, nothing to draw
          if (total_visible_volume > 0) {
            // Calculate the rightmost x-coordinate in plot space (this will be our anchor)
            // Use the maximum volume across the entire dataset for consistent scaling
            double max_total_volume = 0.0;
            for (size_t i = 0; i < yesterday_buy_volumes.size(); ++i) {
              double total_vol =
                  std::abs(yesterday_buy_volumes[i]) + std::abs(yesterday_sell_volumes[i]);
              if (total_vol > max_total_volume) {
                max_total_volume = total_vol;
              }
            }
            if (max_total_volume <= 0) max_total_volume = yesterday_max_volume_;
            if (max_total_volume <= 0) max_total_volume = 1.0;  // Ultimate fallback

            // Anchor to the right edge
            double right_anchor = max_total_volume;

            // Draw a single horizontal bar representing the total volume of all visible trades
            // The bar extends left from the right anchor, with width proportional to total volume
            ImDrawList* draw_list = ImPlot::GetPlotDrawList();

            // Calculate the left extent based on total visible volume
            double bar_left_extent =
                right_anchor - (total_visible_volume / max_total_volume) * max_total_volume;

            // Determine the vertical range of the visible area (entire visible Y range)
            float bar_top = ImPlot::PlotToPixels(0, plot_limits.Y.Max).y;  // Top of visible area
            float bar_bottom =
                ImPlot::PlotToPixels(0, plot_limits.Y.Min).y;  // Bottom of visible area
            float bar_right =
                ImPlot::PlotToPixels(right_anchor, (plot_limits.Y.Max + plot_limits.Y.Min) / 2)
                    .x;  // Right edge of chart
            float bar_left =
                ImPlot::PlotToPixels(bar_left_extent, (plot_limits.Y.Max + plot_limits.Y.Min) / 2)
                    .x;  // Left extent of the bar

            // Determine color based on whether buy or sell volume dominates in the visible range
            // Use reduced opacity for yesterday's profile (30%)
            ImU32 color;
            if (total_visible_buy_volume >= total_visible_sell_volume) {
              // More buy volume - use green with intensity based on dominance
              float dominance =
                  (total_visible_volume > 0)
                      ? static_cast<float>((total_visible_buy_volume - total_visible_sell_volume) /
                                           total_visible_volume)
                      : 0.0f;
              int green = 200 + static_cast<int>(55 * dominance);  // Vary from 200 to 255
              int red = 26 - static_cast<int>(26 * dominance);     // Vary from 26 to 0
              color = IM_COL32(red, green, 26,
                               static_cast<int>(179 * 0.3f));  // Green dominant with 30% opacity
            } else {
              // More sell volume - use red with intensity based on dominance
              float dominance =
                  (total_visible_volume > 0)
                      ? static_cast<float>((total_visible_sell_volume - total_visible_buy_volume) /
                                           total_visible_volume)
                      : 0.0f;
              int red = 200 + static_cast<int>(55 * dominance);   // Vary from 200 to 255
              int green = 26 - static_cast<int>(26 * dominance);  // Vary from 26 to 0
              color = IM_COL32(red, green, 26,
                               static_cast<int>(179 * 0.3f));  // Red dominant with 30% opacity
            }

            // Draw the aggregated horizontal bar extending left from the right anchor
            draw_list->AddRectFilled(ImVec2(bar_left, bar_top), ImVec2(bar_right, bar_bottom),
                                     color);

            // Add a subtle border for better visibility
            draw_list->AddRect(ImVec2(bar_left, bar_top), ImVec2(bar_right, bar_bottom),
                               IM_COL32(0, 0, 0, static_cast<int>(100 * 0.3f)), 0.0f, 0, 1.0f);
          }
          break;
        }
        case ProfileMode::Left: {
          // Left Profile: Same as Right but anchored to left edge, bars extending right
          ImPlotRect plot_limits = ImPlot::GetPlotLimits();  // This gets the current visible range

          // Aggregate all visible trades from yesterday into a single total volume
          double total_visible_buy_volume = 0.0;
          double total_visible_sell_volume = 0.0;
          double total_visible_volume = 0.0;

          for (size_t i = 0; i < yesterday_buy_volumes.size(); ++i) {
            // Only include if the price level is within the visible range
            if (yesterday_prices[i] >= plot_limits.Y.Min &&
                yesterday_prices[i] <= plot_limits.Y.Max) {
              double buy_vol = std::abs(yesterday_buy_volumes[i]);
              double sell_vol = std::abs(yesterday_sell_volumes[i]);
              double total_vol = buy_vol + sell_vol;

              if (total_vol > 0) {
                total_visible_buy_volume += buy_vol;
                total_visible_sell_volume += sell_vol;
                total_visible_volume += total_vol;
              }
            }
          }

          // If no visible volume, nothing to draw
          if (total_visible_volume > 0) {
            // Calculate the leftmost x-coordinate in plot space (this will be our anchor)
            // Use the maximum volume across the entire dataset for consistent scaling
            double max_total_volume = 0.0;
            for (size_t i = 0; i < yesterday_buy_volumes.size(); ++i) {
              double total_vol =
                  std::abs(yesterday_buy_volumes[i]) + std::abs(yesterday_sell_volumes[i]);
              if (total_vol > max_total_volume) {
                max_total_volume = total_vol;
              }
            }
            if (max_total_volume <= 0) max_total_volume = yesterday_max_volume_;
            if (max_total_volume <= 0) max_total_volume = 1.0;  // Ultimate fallback

            // Anchor to the left edge
            double left_anchor = 0.0;

            // Draw a single horizontal bar representing the total volume of all visible trades
            // The bar extends right from the left anchor, with width proportional to total volume
            ImDrawList* draw_list = ImPlot::GetPlotDrawList();

            // Calculate the right extent based on total visible volume
            double bar_right_extent =
                left_anchor + (total_visible_volume / max_total_volume) * max_total_volume;

            // Determine the vertical range of the visible area (entire visible Y range)
            float bar_top = ImPlot::PlotToPixels(0, plot_limits.Y.Min)
                                .y;  // Top of visible area (note: Y axis is inverted in plots)
            float bar_bottom =
                ImPlot::PlotToPixels(0, plot_limits.Y.Max)
                    .y;  // Bottom of visible area (note: Y axis is inverted in plots)
            float bar_left =
                ImPlot::PlotToPixels(left_anchor, (plot_limits.Y.Max + plot_limits.Y.Min) / 2)
                    .x;  // Left edge of chart
            float bar_right =
                ImPlot::PlotToPixels(bar_right_extent, (plot_limits.Y.Max + plot_limits.Y.Min) / 2)
                    .x;  // Right extent of the bar

            // Determine color based on whether buy or sell volume dominates in the visible range
            // Use reduced opacity for yesterday's profile (30%)
            ImU32 color;
            if (total_visible_buy_volume >= total_visible_sell_volume) {
              // More buy volume - use green with intensity based on dominance
              float dominance =
                  (total_visible_volume > 0)
                      ? static_cast<float>((total_visible_buy_volume - total_visible_sell_volume) /
                                           total_visible_volume)
                      : 0.0f;
              int green = 200 + static_cast<int>(55 * dominance);  // Vary from 200 to 255
              int red = 26 - static_cast<int>(26 * dominance);     // Vary from 26 to 0
              color = IM_COL32(red, green, 26,
                               static_cast<int>(179 * 0.3f));  // Green dominant with 30% opacity
            } else {
              // More sell volume - use red with intensity based on dominance
              float dominance =
                  (total_visible_volume > 0)
                      ? static_cast<float>((total_visible_sell_volume - total_visible_buy_volume) /
                                           total_visible_volume)
                      : 0.0f;
              int red = 200 + static_cast<int>(55 * dominance);   // Vary from 200 to 255
              int green = 26 - static_cast<int>(26 * dominance);  // Vary from 26 to 0
              color = IM_COL32(red, green, 26,
                               static_cast<int>(179 * 0.3f));  // Red dominant with 30% opacity
            }

            // Draw the aggregated horizontal bar extending right from the left anchor
            draw_list->AddRectFilled(ImVec2(bar_left, bar_top), ImVec2(bar_right, bar_bottom),
                                     color);

            // Add a subtle border for better visibility
            draw_list->AddRect(ImVec2(bar_left, bar_top), ImVec2(bar_right, bar_bottom),
                               IM_COL32(0, 0, 0, static_cast<int>(100 * 0.3f)), 0.0f, 0, 1.0f);
          }
          break;
        }
        case ProfileMode::Custom:
        default:
          // Custom Profile Mode: Split bars with buy volume on left (green) and sell volume on
          // right (red) Each bar is centered at zero with buy volume extending left (negative) and
          // sell volume extending right (positive)
          render_yesterday_split_profile(
              yesterday_prices.data(), yesterday_buy_volumes.data(), yesterday_sell_volumes.data(),
              static_cast<int>(yesterday_prices.size()), bar_height, 0.3f);  // 30% opacity
          break;
      }

      // Draw yesterday's POC line with reduced opacity
      if (yesterday_poc_price_ > 0) {
        if (profile_mode_ != ProfileMode::Step) {
          // For other modes, use ImPlot's PlotLine
          double poc_line_x[2] = {-yesterday_max_volume_, yesterday_max_volume_};
          double poc_line_y[2] = {yesterday_poc_price_, yesterday_poc_price_};
          // ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.8f, 0.0f, 0.3f)); // 30% opacity
          ImPlot::PlotLine("Yesterday POC", poc_line_x, poc_line_y, 2);
          ImPlot::PopStyleColor();
        }
      }
    }

    // Highlight zones where profiles diverge significantly
    highlight_profile_divergence(ImPlot::GetPlotDrawList());

    // Render custom profile overlay if in custom profile mode and time range selection is active
    if (profile_mode_ == ProfileMode::Custom) {
      renderCustomProfileOverlay(ImPlot::GetPlotDrawList());
    }

    ImPlot::EndPlot();
  }
}

void VolumeProfilePanel::calculate_value_area() {
  // Use the current volume profile to calculate value area
  const auto& profile = volume_profile_;

  if (profile.empty()) {
    vah_price_ = 0.0;
    val_price_ = 0.0;
    return;
  }

  // Calculate total volume in the profile for all modes
  double total_volume = 0.0;
  for (const auto& level : profile) {
    total_volume += level.total_volume;
  }

  if (total_volume <= 0) {
    vah_price_ = 0.0;
    val_price_ = 0.0;
    return;
  }

  // Target volume for value area (based on profile_settings_.vaPercent % of total volume)
  double target_volume = (static_cast<double>(profile_settings_.vaPercent) / 100.0) * total_volume;

  // Find the POC index (Point of Control - highest volume)
  size_t poc_index = 0;
  double max_total_volume = 0.0;
  for (size_t i = 0; i < profile.size(); ++i) {
    double total = profile[i].buy_volume + profile[i].sell_volume;
    if (total > max_total_volume) {
      max_total_volume = total;
      poc_index = i;
    }
  }

  // Expand from POC outward to capture the required volume, centered around POC
  size_t start_idx = poc_index;
  size_t end_idx = poc_index;
  double current_volume = profile[poc_index].total_volume;

  // Expand upward (higher prices) and downward (lower prices) alternately
  // until we reach the target volume, keeping the expansion balanced around POC
  while (current_volume < target_volume) {
    // Decide whether to expand up or down
    bool expand_up = false;
    bool expand_down = false;

    // Check if we can expand in each direction
    if (start_idx > 0) expand_down = true;
    if (end_idx < profile.size() - 1) expand_up = true;

    // If we can't expand in either direction, break
    if (!expand_up && !expand_down) break;

    // If we can only expand in one direction, do that
    if (!expand_up && expand_down) {
      start_idx--;
      current_volume += profile[start_idx].total_volume;
    } else if (expand_up && !expand_down) {
      end_idx++;
      current_volume += profile[end_idx].total_volume;
    } else {
      // We can expand in both directions - try to keep it centered around POC
      // Calculate potential volumes for each direction
      double vol_up = profile[end_idx + 1].total_volume;
      double vol_down = profile[start_idx - 1].total_volume;

      // To center around POC, we should try to balance the expansion
      // If both sides have similar volume, expand the side that currently has a smaller range
      size_t current_upper_range = end_idx - poc_index;
      size_t current_lower_range = poc_index - start_idx;

      if (current_lower_range < current_upper_range) {
        // Current lower range is smaller, expand downward to balance
        start_idx--;
        current_volume += profile[start_idx].total_volume;
      } else if (current_upper_range < current_lower_range) {
        // Current upper range is smaller, expand upward to balance
        end_idx++;
        current_volume += profile[end_idx].total_volume;
      } else {
        // Ranges are equal, expand toward the side with more volume to capture more volume
        // efficiently
        if (vol_up >= vol_down) {
          end_idx++;
          current_volume += profile[end_idx].total_volume;
        } else {
          start_idx--;
          current_volume += profile[start_idx].total_volume;
        }
      }
    }

    // If we've captured enough volume, break
    if (current_volume >= target_volume) break;
  }

  // Set the VAH and VAL prices
  vah_price_ = profile[end_idx].price;
  val_price_ = profile[start_idx].price;
}

// Calculate value area for a specific session profile
void VolumeProfilePanel::calculate_value_area_for_session(SessionProfile& session) {
  if (session.volume_profile.empty()) {
    session.vah_price = 0.0;
    session.val_price = 0.0;
    return;
  }

  // Calculate total volume in the session profile
  double total_volume = 0.0;
  for (const auto& level : session.volume_profile) {
    total_volume += level.total_volume;
  }

  if (total_volume <= 0) {
    session.vah_price = 0.0;
    session.val_price = 0.0;
    return;
  }

  // Target volume for value area (based on profile_settings_.vaPercent % of total volume)
  double target_volume = (static_cast<double>(profile_settings_.vaPercent) / 100.0) * total_volume;

  // Find the POC index (Point of Control - highest volume)
  size_t poc_index = 0;
  double max_total_volume = 0.0;
  for (size_t i = 0; i < session.volume_profile.size(); ++i) {
    double total = session.volume_profile[i].buy_volume + session.volume_profile[i].sell_volume;
    if (total > max_total_volume) {
      max_total_volume = total;
      poc_index = i;
    }
  }

  // Expand from POC outward to capture the required volume, centered around POC
  size_t start_idx = poc_index;
  size_t end_idx = poc_index;
  double current_volume = session.volume_profile[poc_index].total_volume;

  // Expand upward (higher prices) and downward (lower prices) alternately
  // until we reach the target volume, keeping the expansion balanced around POC
  while (current_volume < target_volume) {
    // Decide whether to expand up or down
    bool expand_up = false;
    bool expand_down = false;

    // Check if we can expand in each direction
    if (start_idx > 0) expand_down = true;
    if (end_idx < session.volume_profile.size() - 1) expand_up = true;

    // If we can't expand in either direction, break
    if (!expand_up && !expand_down) break;

    // If we can only expand in one direction, do that
    if (!expand_up && expand_down) {
      start_idx--;
      current_volume += session.volume_profile[start_idx].total_volume;
    } else if (expand_up && !expand_down) {
      end_idx++;
      current_volume += session.volume_profile[end_idx].total_volume;
    } else {
      // We can expand in both directions - try to keep it centered around POC
      // Calculate potential volumes for each direction
      double vol_up = session.volume_profile[end_idx + 1].total_volume;
      double vol_down = session.volume_profile[start_idx - 1].total_volume;

      // To center around POC, we should try to balance the expansion
      // If both sides have similar volume, expand the side that currently has a smaller range
      size_t current_upper_range = end_idx - poc_index;
      size_t current_lower_range = poc_index - start_idx;

      if (current_lower_range < current_upper_range) {
        // Current lower range is smaller, expand downward to balance
        start_idx--;
        current_volume += session.volume_profile[start_idx].total_volume;
      } else if (current_upper_range < current_lower_range) {
        // Current upper range is smaller, expand upward to balance
        end_idx++;
        current_volume += session.volume_profile[end_idx].total_volume;
      } else {
        // Ranges are equal, expand toward the side with more volume to capture more volume
        // efficiently
        if (vol_up >= vol_down) {
          end_idx++;
          current_volume += session.volume_profile[end_idx].total_volume;
        } else {
          start_idx--;
          current_volume += session.volume_profile[start_idx].total_volume;
        }
      }
    }

    // If we've captured enough volume, break
    if (current_volume >= target_volume) break;
  }

  // Set the VAH and VAL prices for the session
  session.vah_price = session.volume_profile[end_idx].price;
  session.val_price = session.volume_profile[start_idx].price;
}

// Calculate Virgin POC - find the highest virgin price level with potential significance
void VolumeProfilePanel::calculate_virgin_poc() {
  // Find the virgin price levels that are still untouched
  std::vector<VolumeLevel> virgin_levels;

  for (size_t i = 0; i < volume_profile_.size() && i < virgin_price_levels_.size(); ++i) {
    if (virgin_price_levels_[i]) {
      virgin_levels.push_back(volume_profile_[i]);
    }
  }

  if (!virgin_levels.empty()) {
    // For virgin POC, we want to identify the most significant untouched price level
    // This could be based on proximity to current market price, or historical importance

    // Get current market price from analytics
    auto analytics = processor_->getSymbolAnalytics(symbol_id_);
    double current_price = analytics.last_trade_price > 0
                               ? analytics.last_trade_price
                               : volume_profile_[volume_profile_.size() / 2].price;

    // Find the virgin level closest to current market price (as potential resistance/support)
    double min_distance = std::numeric_limits<double>::max();
    double closest_virgin_price = virgin_levels[0].price;

    for (const auto& level : virgin_levels) {
      double distance = std::abs(level.price - current_price);
      if (distance < min_distance) {
        min_distance = distance;
        closest_virgin_price = level.price;
      }
    }

    // Alternatively, we could also consider the price levels that are most likely to be tested
    // based on their position relative to current market conditions
    virgin_poc_price_ = closest_virgin_price;
  } else {
    // If no virgin levels remain, set to 0
    virgin_poc_price_ = 0.0;
  }
}

void VolumeProfilePanel::render_step_profile(const double* xs, const double* ys,
                                             const double* neg_ys, int count, double height) {
  if (count <= 0) return;
  ImDrawList* draw_list = ImPlot::GetPlotDrawList();
  const ImU32 col_pos = IM_COL32(0, 255, 0, 170);    // Green for positive
  const ImU32 col_neg = IM_COL32(255, 0, 0, 170);    // Red for negative
  const ImU32 col_poc = IM_COL32(255, 255, 0, 255);  // Yellow for POC (Point of Control)

  // Find the POC (Point of Control) - the price level with highest total volume
  int poc_index = -1;
  double max_total_volume = 0.0;

  for (int i = 0; i < count; ++i) {
    double total_volume = std::abs(ys[i]) + std::abs(neg_ys[i]);
    if (total_volume > max_total_volume) {
      max_total_volume = total_volume;
      poc_index = i;
    }
  }

  for (int i = 0; i < count; ++i) {
    // Determine if this is the POC bar
    bool is_poc_bar = (i == poc_index && max_total_volume > 0);
    ImU32 current_col_pos = is_poc_bar ? col_poc : col_pos;
    ImU32 current_col_neg = is_poc_bar ? col_poc : col_neg;

    if (ys[i] != 0) {
      ImVec2 p1 = ImPlot::PlotToPixels(0, xs[i]);  // x-axis is volume, y-axis is price
      ImVec2 p2 = ImPlot::PlotToPixels(ys[i], xs[i]);

      // Draw step-style bar
      ImVec2 bar_tl = ImVec2(std::min(p1.x, p2.x), p1.y - height / 2);
      ImVec2 bar_br = ImVec2(std::max(p1.x, p2.x), p1.y + height / 2);

      draw_list->AddRectFilled(bar_tl, bar_br, current_col_pos);
    }

    if (neg_ys[i] != 0) {
      ImVec2 p1 = ImPlot::PlotToPixels(0, xs[i]);
      ImVec2 p2 = ImPlot::PlotToPixels(neg_ys[i], xs[i]);

      // Draw step-style bar for negative values
      ImVec2 bar_tl = ImVec2(std::min(p1.x, p2.x), p1.y - height / 2);
      ImVec2 bar_br = ImVec2(std::max(p1.x, p2.x), p1.y + height / 2);

      draw_list->AddRectFilled(bar_tl, bar_br, current_col_neg);
    }
  }

  // Draw horizontal yellow POC line at the price level with highest volume
  // Use the locally calculated POC for consistency with highlighted bar
  if (poc_index >= 0 && max_total_volume > 0) {
    // Get the current plot limits to draw the POC line across the full visible width
    ImPlotRect plot_limits = ImPlot::GetPlotLimits();

    // Use the plot limits to draw the line across the full width of the visible area
    ImVec2 poc_start = ImPlot::PlotToPixels(plot_limits.X.Min, xs[poc_index]);
    ImVec2 poc_end = ImPlot::PlotToPixels(plot_limits.X.Max, xs[poc_index]);

    // Draw the horizontal POC line - make it more prominent with consistent styling
    // Use the same color as in other profile modes for consistency
    draw_list->AddLine(poc_start, poc_end, IM_COL32(255, 255, 0, 255),
                       3.0f);  // Bright yellow with increased thickness

    // Also update the global POC price to reflect the current calculation for display purposes
    poc_price_ = xs[poc_index];

    // Ensure the POC line is visible and properly rendered in Step Profile mode
    // This confirms that the POC line is calculated and rendered for each bar in Step Profile mode
  }
}

// Method to render mini histogram overlays on candlestick charts
void VolumeProfilePanel::render_mini_histograms_on_candles(
    ImDrawList* draw_list, const std::vector<RenderEngine::OHLCVCandle>& candles,
    const std::vector<double>& x_coords, const std::vector<double>& y_coords_high,
    const std::vector<double>& y_coords_low) {
  if (candles.empty() || x_coords.size() != candles.size() ||
      y_coords_high.size() != candles.size() || y_coords_low.size() != candles.size()) {
    return;
  }

  // Get theme colors for consistent styling
  const auto& colors = ThemeManager::getInstance().getColors();

  // Iterate through each candle to draw mini volume profile histogram
  for (size_t i = 0; i < candles.size(); ++i) {
    const auto& candle = candles[i];

    // Skip if candle has no volume
    if (candle.volume <= 0) continue;

    // Get the candle's price range (high - low)
    double price_range = candle.high - candle.low;
    if (price_range <= 0) continue;

    // Determine number of price buckets for this candle's range
    int num_buckets = 8;  // Fixed number of buckets for mini histogram
    double bucket_size = price_range / num_buckets;

    // Get recent trades for this symbol to populate the histogram
    auto analytics = processor_->getSymbolAnalytics(symbol_id_);
    const auto& trades = analytics.recent_trades;

    // Create temporary buckets for this candle's price range
    std::vector<double> bucket_volumes(num_buckets, 0.0);
    std::vector<int> bucket_counts(num_buckets, 0);

    // Aggregate trades into buckets based on price within this candle's range
    for (const auto& trade : trades) {
      if (trade.price >= candle.low && trade.price <= candle.high) {
        int bucket_idx = static_cast<int>((trade.price - candle.low) / bucket_size);
        // Ensure we don't exceed bounds
        bucket_idx = std::max(0, std::min(bucket_idx, num_buckets - 1));

        bucket_volumes[bucket_idx] += trade.size;
        bucket_counts[bucket_idx]++;
      }
    }

    // Find max volume in this candle's histogram for scaling
    double max_vol_in_candle = 0.0;
    int poc_bucket_idx = 0;  // Index of the bucket with highest volume (POC)

    for (int j = 0; j < num_buckets; ++j) {
      if (bucket_volumes[j] > max_vol_in_candle) {
        max_vol_in_candle = bucket_volumes[j];
        poc_bucket_idx = j;
      }
    }

    if (max_vol_in_candle <= 0) continue;

    // Calculate screen coordinates for the mini histogram
    float x_center = static_cast<float>(x_coords[i]);
    float y_high = static_cast<float>(y_coords_high[i]);
    float y_low = static_cast<float>(y_coords_low[i]);

    // Calculate height of each bucket in screen coordinates
    float total_height = y_low - y_high;  // Height of the candle in screen space
    float bucket_height = total_height / num_buckets;

    // Draw mini histogram inside the candle
    for (int j = 0; j < num_buckets; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage =
            static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.8f;  // Use 80% of height as width for visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is the POC bucket or not
        ImU32 color;
        if (j == poc_bucket_idx) {
          // Highlight POC bucket with yellow
          color = IM_COL32(255, 255, 0, 220);  // Brighter yellow for POC
        } else {
          // Use gradient colors based on volume intensity and position in the candle
          float volume_ratio =
              static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);
          float position_ratio = static_cast<float>(j) / static_cast<float>(num_buckets - 1);

          // Create a color gradient based on volume intensity and position
          if (position_ratio < 0.5f) {
            // Lower half - red for selling pressure, with intensity based on volume
            color = IM_COL32(255, static_cast<int>(100 * volume_ratio),
                             static_cast<int>(100 * volume_ratio), 150);
          } else {
            // Upper half - green for buying pressure, with intensity based on volume
            color = IM_COL32(static_cast<int>(100 * volume_ratio), 255,
                             static_cast<int>(100 * volume_ratio), 150);
          }
        }

        // Draw the mini histogram bar with a slight outline for better visibility
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);

        // Add a subtle border to make individual bars more distinguishable
        draw_list->AddRect(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), IM_COL32(0, 0, 0, 50));
      }
    }

    // Draw POC (Point of Control) line - horizontal yellow line at the price level with highest
    // volume Calculate the y-coordinate for the POC line
    float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height;  // Center of the POC bucket

    // Draw horizontal yellow line across the candle width
    float poc_line_half_width = (y_low - y_high) * 0.4f;  // Same width as candle
    float poc_x_left = x_center - poc_line_half_width;
    float poc_x_right = x_center + poc_line_half_width;

    // Draw the POC line as a horizontal yellow line
    draw_list->AddLine(ImVec2(poc_x_left, poc_y), ImVec2(poc_x_right, poc_y),
                       IM_COL32(255, 255, 0, 255),  // Yellow color for POC
                       2.0f                         // Line thickness
    );
  }
}

void VolumeProfilePanel::render_step_profile_histograms(
    ImDrawList* draw_list, const std::vector<RenderEngine::OHLCVCandle>& candles,
    const std::vector<double>& x_coords, const std::vector<double>& y_coords_high,
    const std::vector<double>& y_coords_low, bool show_poc_line, int num_buckets) {
  if (candles.empty() || x_coords.size() != candles.size() ||
      y_coords_high.size() != candles.size() || y_coords_low.size() != candles.size()) {
    return;
  }

  // Get theme colors for consistent styling
  const auto& colors = ThemeManager::getInstance().getColors();

  // Get recent trades for this symbol to populate the histograms
  // Only fetch once for all candles to improve efficiency
  auto analytics = processor_->getSymbolAnalytics(symbol_id_);
  const auto& trades = analytics.recent_trades;

  // Iterate through each candle to draw step profile histogram
  for (size_t i = 0; i < candles.size(); ++i) {
    const auto& candle = candles[i];

    // Skip if candle has no volume
    if (candle.volume <= 0) continue;

    // Get the candle's price range (high - low)
    double price_range = candle.high - candle.low;
    if (price_range <= 0) continue;

    // Calculate bucket size for this candle's range
    double bucket_size = price_range / num_buckets;

    // Create temporary buckets for this candle's price range
    std::vector<double> bucket_volumes(num_buckets, 0.0);
    std::vector<int> bucket_counts(num_buckets, 0);

    // Aggregate trades into buckets based on price within this candle's range
    for (const auto& trade : trades) {
      if (trade.price >= candle.low && trade.price <= candle.high) {
        int bucket_idx = static_cast<int>((trade.price - candle.low) / bucket_size);
        // Ensure we don't exceed bounds
        bucket_idx = std::max(0, std::min(bucket_idx, num_buckets - 1));

        bucket_volumes[bucket_idx] += trade.size;
        bucket_counts[bucket_idx]++;
      }
    }

    // Find max volume in this candle's histogram for scaling
    double max_vol_in_candle = 0.0;
    int poc_bucket_idx = 0;  // Index of the bucket with highest volume (POC)

    for (int j = 0; j < num_buckets; ++j) {
      if (bucket_volumes[j] > max_vol_in_candle) {
        max_vol_in_candle = bucket_volumes[j];
        poc_bucket_idx = j;
      }
    }

    if (max_vol_in_candle <= 0) continue;

    // Calculate screen coordinates for the step profile histogram
    float x_center = static_cast<float>(x_coords[i]);
    float y_high = static_cast<float>(y_coords_high[i]);
    float y_low = static_cast<float>(y_coords_low[i]);

    // Calculate height of each bucket in screen coordinates
    float total_height = y_low - y_high;  // Height of the candle in screen space
    float bucket_height = total_height / num_buckets;

    // Draw step profile histogram inside the candle
    for (int j = 0; j < num_buckets; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage =
            static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.8f;  // Use 80% of height as width for visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is the POC bucket or not
        ImU32 color;
        if (j == poc_bucket_idx) {
          // Highlight POC bucket with yellow
          color = IM_COL32(255, 255, 0, 220);  // Brighter yellow for POC
        } else {
          // Use gradient colors based on volume intensity and position in the candle
          float volume_ratio =
              static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);
          float position_ratio = static_cast<float>(j) / static_cast<float>(num_buckets - 1);

          // Create a color gradient based on volume intensity and position
          if (position_ratio < 0.5f) {
            // Lower half - red for selling pressure, with intensity based on volume
            color = IM_COL32(255, static_cast<int>(100 * volume_ratio),
                             static_cast<int>(100 * volume_ratio), 150);
          } else {
            // Upper half - green for buying pressure, with intensity based on volume
            color = IM_COL32(static_cast<int>(100 * volume_ratio), 255,
                             static_cast<int>(100 * volume_ratio), 150);
          }
        }

        // Draw the step profile histogram bar with a slight outline for better visibility
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);

        // Add a subtle border to make individual bars more distinguishable
        draw_list->AddRect(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), IM_COL32(0, 0, 0, 50));
      }
    }

    // Optionally draw POC (Point of Control) line - horizontal yellow line at the price level with
    // highest volume
    if (show_poc_line) {
      // Calculate the y-coordinate for the POC line
      float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height;  // Center of the POC bucket

      // Draw horizontal yellow line across the candle width
      float poc_line_half_width = (y_low - y_high) * 0.4f;  // Same width as candle
      float poc_x_left = x_center - poc_line_half_width;
      float poc_x_right = x_center + poc_line_half_width;

      // Draw the POC line as a horizontal yellow line
      draw_list->AddLine(ImVec2(poc_x_left, poc_y), ImVec2(poc_x_right, poc_y),
                         IM_COL32(255, 255, 0, 255),  // Yellow color for POC
                         2.0f                         // Line thickness
      );
    }
  }
}

void VolumeProfilePanel::render_candle_volume_distribution(
    ImDrawList* draw_list, const std::vector<RenderEngine::OHLCVCandle>& candles,
    const std::vector<double>& x_coords, const std::vector<double>& y_coords_high,
    const std::vector<double>& y_coords_low, bool show_poc_line, int num_buckets) {
  // This method implements the Step Profile rendering: draw mini histogram overlay
  // on each candlestick bar showing volume distribution for that bar's price range
  render_step_profile_histograms(draw_list, candles, x_coords, y_coords_high, y_coords_low,
                                 show_poc_line, num_buckets);
}

void VolumeProfilePanel::render_step_profile_on_candles(
    ImDrawList* draw_list, const std::vector<RenderEngine::OHLCVCandle>& candles,
    const std::vector<double>& x_coords, const std::vector<double>& y_coords_high,
    const std::vector<double>& y_coords_low, bool show_poc_line, int num_buckets_per_candle) {
  // Enhanced Step Profile rendering: draw mini histogram overlay on each candlestick bar
  // showing volume distribution for that bar's price range
  if (candles.empty() || x_coords.size() != candles.size() ||
      y_coords_high.size() != candles.size() || y_coords_low.size() != candles.size()) {
    return;
  }

  // Get recent trades for this symbol to populate the histograms
  auto analytics = processor_->getSymbolAnalytics(symbol_id_);
  const auto& trades = analytics.recent_trades;

  // Iterate through each candle to draw step profile histogram
  for (size_t i = 0; i < candles.size(); ++i) {
    const auto& candle = candles[i];

    // Skip if candle has no meaningful price range
    if (candle.high <= candle.low || candle.volume <= 0) continue;

    // Get the candle's price range (high - low)
    double price_range = candle.high - candle.low;
    if (price_range <= 0) continue;

    // Calculate bucket size for this candle's range
    double bucket_size = price_range / num_buckets_per_candle;

    // Create temporary buckets for this candle's price range
    std::vector<double> bucket_volumes(num_buckets_per_candle, 0.0);
    std::vector<int> bucket_counts(num_buckets_per_candle, 0);

    // Aggregate trades into buckets based on price within this candle's range
    for (const auto& trade : trades) {
      // Only consider trades within this candle's price range
      if (trade.price >= candle.low && trade.price <= candle.high) {
        int bucket_idx = static_cast<int>((trade.price - candle.low) / bucket_size);
        // Ensure we don't exceed bounds
        bucket_idx = std::max(0, std::min(bucket_idx, num_buckets_per_candle - 1));

        bucket_volumes[bucket_idx] += trade.size;
        bucket_counts[bucket_idx]++;
      }
    }

    // Find max volume in this candle's histogram for scaling
    double max_vol_in_candle = 0.0;
    int poc_bucket_idx = -1;  // Index of the bucket with highest volume (POC)

    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > max_vol_in_candle) {
        max_vol_in_candle = bucket_volumes[j];
        poc_bucket_idx = j;
      }
    }

    // If no trades found in this candle's range, skip
    if (max_vol_in_candle <= 0 || poc_bucket_idx < 0) continue;

    // Calculate screen coordinates for the step profile histogram
    float x_center = static_cast<float>(x_coords[i]);
    float y_high = static_cast<float>(y_coords_high[i]);
    float y_low = static_cast<float>(y_coords_low[i]);

    // Calculate height of each bucket in screen coordinates
    float total_height = y_low - y_high;  // Height of the candle in screen space
    float bucket_height = total_height / num_buckets_per_candle;

    // Draw step profile histogram inside the candle
    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage =
            static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.6f;  // Use 60% of height as width for visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is the POC bucket or not
        ImU32 color;
        if (j == poc_bucket_idx) {
          // Highlight POC bucket with bright yellow
          color = IM_COL32(255, 255, 0, 240);  // Bright yellow for POC with high opacity
        } else {
          // Use gradient colors based on volume intensity and position in the candle
          float volume_ratio =
              static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

          // Determine if this bucket is in the upper or lower half of the candle
          float position_ratio =
              static_cast<float>(j) / static_cast<float>(num_buckets_per_candle - 1);

          // Create a color gradient based on volume intensity and position
          if (position_ratio < 0.5f) {
            // Lower half - red for selling pressure, with intensity based on volume
            int red_intensity = static_cast<int>(255 * volume_ratio);
            color = IM_COL32(red_intensity, static_cast<int>(50 * volume_ratio),
                             static_cast<int>(50 * volume_ratio), 180);
          } else {
            // Upper half - green for buying pressure, with intensity based on volume
            int green_intensity = static_cast<int>(255 * volume_ratio);
            color = IM_COL32(static_cast<int>(50 * volume_ratio), green_intensity,
                             static_cast<int>(50 * volume_ratio), 180);
          }
        }

        // Draw the step profile histogram bar with a slight outline for better visibility
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);

        // Add a subtle border to make individual bars more distinguishable
        draw_list->AddRect(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), IM_COL32(0, 0, 0, 80),
                           0.0f, 0, 1.0f);
      }
    }

    // Optionally draw POC (Point of Control) line - horizontal yellow line at the price level with
    // highest volume
    if (show_poc_line && poc_bucket_idx >= 0) {
      // Calculate the y-coordinate for the POC line
      float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height;  // Center of the POC bucket

      // Draw horizontal yellow line across the candle width
      float poc_line_half_width =
          (total_height) * 0.3f;  // Reduced width to avoid overlapping with candle wicks
      float poc_x_left = x_center - poc_line_half_width;
      float poc_x_right = x_center + poc_line_half_width;

      // Draw the POC line as a horizontal yellow line
      draw_list->AddLine(ImVec2(poc_x_left, poc_y), ImVec2(poc_x_right, poc_y),
                         IM_COL32(255, 255, 0, 255),  // Bright yellow color for POC
                         2.0f);                       // Line thickness
    }
  }
}

void VolumeProfilePanel::render_mini_histograms_direct(
    ImDrawList* draw_list, const std::vector<RenderEngine::OHLCVCandle>& candles,
    const std::vector<double>& x_coords, const std::vector<double>& y_coords_high,
    const std::vector<double>& y_coords_low, const std::vector<RenderEngine::TradeData>& trades,
    bool show_poc_line, int num_buckets) {
  if (candles.empty() || x_coords.size() != candles.size() ||
      y_coords_high.size() != candles.size() || y_coords_low.size() != candles.size()) {
    return;
  }

  // Get theme colors for consistent styling
  const auto& colors = ThemeManager::getInstance().getColors();

  // Iterate through each candle to draw step profile histogram
  for (size_t i = 0; i < candles.size(); ++i) {
    const auto& candle = candles[i];

    // Skip if candle has no volume
    if (candle.volume <= 0) continue;

    // Get the candle's price range (high - low)
    double price_range = candle.high - candle.low;
    if (price_range <= 0) continue;

    // Calculate bucket size for this candle's range
    double bucket_size = price_range / num_buckets;

    // Create temporary buckets for this candle's price range
    std::vector<double> bucket_volumes(num_buckets, 0.0);
    std::vector<int> bucket_counts(num_buckets, 0);

    // Aggregate trades into buckets based on price within this candle's range
    for (const auto& trade : trades) {
      if (trade.price >= candle.low && trade.price <= candle.high) {
        int bucket_idx = static_cast<int>((trade.price - candle.low) / bucket_size);
        // Ensure we don't exceed bounds
        bucket_idx = std::max(0, std::min(bucket_idx, num_buckets - 1));

        bucket_volumes[bucket_idx] += trade.size;
        bucket_counts[bucket_idx]++;
      }
    }

    // Find max volume in this candle's histogram for scaling
    double max_vol_in_candle = 0.0;
    int poc_bucket_idx = 0;  // Index of the bucket with highest volume (POC)

    for (int j = 0; j < num_buckets; ++j) {
      if (bucket_volumes[j] > max_vol_in_candle) {
        max_vol_in_candle = bucket_volumes[j];
        poc_bucket_idx = j;
      }
    }

    if (max_vol_in_candle <= 0) continue;

    // Calculate screen coordinates for the step profile histogram
    float x_center = static_cast<float>(x_coords[i]);
    float y_high = static_cast<float>(y_coords_high[i]);
    float y_low = static_cast<float>(y_coords_low[i]);

    // Calculate height of each bucket in screen coordinates
    float total_height = y_low - y_high;  // Height of the candle in screen space
    float bucket_height = total_height / num_buckets;

    // Draw step profile histogram inside the candle
    for (int j = 0; j < num_buckets; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage =
            static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.8f;  // Use 80% of height as width for visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is the POC bucket or not
        ImU32 color;
        if (j == poc_bucket_idx) {
          // Highlight POC bucket with yellow
          color = IM_COL32(255, 255, 0, 220);  // Brighter yellow for POC
        } else {
          // Use gradient colors based on volume intensity and position in the candle
          float volume_ratio =
              static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);
          float position_ratio = static_cast<float>(j) / static_cast<float>(num_buckets - 1);

          // Create a color gradient based on volume intensity and position
          if (position_ratio < 0.5f) {
            // Lower half - red for selling pressure, with intensity based on volume
            color = IM_COL32(255, static_cast<int>(100 * volume_ratio),
                             static_cast<int>(100 * volume_ratio), 150);
          } else {
            // Upper half - green for buying pressure, with intensity based on volume
            color = IM_COL32(static_cast<int>(100 * volume_ratio), 255,
                             static_cast<int>(100 * volume_ratio), 150);
          }
        }

        // Draw the step profile histogram bar with a slight outline for better visibility
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);

        // Add a subtle border to make individual bars more distinguishable
        draw_list->AddRect(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), IM_COL32(0, 0, 0, 50));
      }
    }

    // Optionally draw POC (Point of Control) line - horizontal yellow line at the price level with
    // highest volume
    if (show_poc_line) {
      // Calculate the y-coordinate for the POC line
      float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height;  // Center of the POC bucket

      // Draw horizontal yellow line across the candle width
      float poc_line_half_width = (y_low - y_high) * 0.4f;  // Same width as candle
      float poc_x_left = x_center - poc_line_half_width;
      float poc_x_right = x_center + poc_line_half_width;

      // Draw the POC line as a horizontal yellow line
      draw_list->AddLine(ImVec2(poc_x_left, poc_y), ImVec2(poc_x_right, poc_y),
                         IM_COL32(255, 255, 0, 255),  // Yellow color for POC
                         2.0f                         // Line thickness
      );
    }
  }
}

// Static method to render step profile directly on candles with improved visualization
void VolumeProfilePanel::render_step_profile_on_candles_static(
    ImDrawList* draw_list, const std::vector<RenderEngine::OHLCVCandle>& candles,
    const std::vector<double>& x_coords, const std::vector<double>& y_coords_high,
    const std::vector<double>& y_coords_low, const std::vector<RenderEngine::TradeData>& trades,
    bool show_poc_line, int num_buckets_per_candle) {
  if (candles.empty() || x_coords.size() != candles.size() ||
      y_coords_high.size() != candles.size() || y_coords_low.size() != candles.size()) {
    return;
  }

  // Iterate through each candle to draw step profile histogram
  for (size_t i = 0; i < candles.size(); ++i) {
    const auto& candle = candles[i];

    // Skip if candle has no meaningful price range
    if (candle.high <= candle.low || candle.volume <= 0) continue;

    // Get the candle's price range (high - low)
    double price_range = candle.high - candle.low;
    if (price_range <= 0) continue;

    // Calculate bucket size for this candle's range
    double bucket_size = price_range / num_buckets_per_candle;

    // Create temporary buckets for this candle's price range
    std::vector<double> bucket_volumes(num_buckets_per_candle, 0.0);
    std::vector<int> bucket_counts(num_buckets_per_candle, 0);

    // Aggregate trades into buckets based on price within this candle's range
    for (const auto& trade : trades) {
      // Only consider trades within this candle's price range
      if (trade.price >= candle.low && trade.price <= candle.high) {
        int bucket_idx = static_cast<int>((trade.price - candle.low) / bucket_size);
        // Ensure we don't exceed bounds
        bucket_idx = std::max(0, std::min(bucket_idx, num_buckets_per_candle - 1));

        bucket_volumes[bucket_idx] += trade.size;
        bucket_counts[bucket_idx]++;
      }
    }

    // Find max volume in this candle's histogram for scaling
    double max_vol_in_candle = 0.0;
    int poc_bucket_idx = -1;  // Index of the bucket with highest volume (POC)

    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > max_vol_in_candle) {
        max_vol_in_candle = bucket_volumes[j];
        poc_bucket_idx = j;
      }
    }

    // If no trades found in this candle's range, skip
    if (max_vol_in_candle <= 0 || poc_bucket_idx < 0) continue;

    // Calculate screen coordinates for the step profile histogram
    float x_center = static_cast<float>(x_coords[i]);
    float y_high = static_cast<float>(y_coords_high[i]);
    float y_low = static_cast<float>(y_coords_low[i]);

    // Calculate height of each bucket in screen coordinates
    float total_height = y_low - y_high;  // Height of the candle in screen space
    float bucket_height = total_height / num_buckets_per_candle;

    // Draw step profile histogram inside the candle
    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage =
            static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.6f;  // Use 60% of height as width for visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is the POC bucket or not
        ImU32 color;
        if (j == poc_bucket_idx) {
          // Highlight POC bucket with bright yellow
          color = IM_COL32(255, 255, 0, 240);  // Bright yellow for POC with high opacity
        } else {
          // Use gradient colors based on volume intensity and position in the candle
          float volume_ratio =
              static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

          // Determine if this bucket is in the upper or lower half of the candle
          float position_ratio =
              static_cast<float>(j) / static_cast<float>(num_buckets_per_candle - 1);

          // Create a color gradient based on volume intensity and position
          if (position_ratio < 0.5f) {
            // Lower half - red for selling pressure, with intensity based on volume
            int red_intensity = static_cast<int>(255 * volume_ratio);
            color = IM_COL32(red_intensity, static_cast<int>(50 * volume_ratio),
                             static_cast<int>(50 * volume_ratio), 180);
          } else {
            // Upper half - green for buying pressure, with intensity based on volume
            int green_intensity = static_cast<int>(255 * volume_ratio);
            color = IM_COL32(static_cast<int>(50 * volume_ratio), green_intensity,
                             static_cast<int>(50 * volume_ratio), 180);
          }
        }

        // Draw the step profile histogram bar with a slight outline for better visibility
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);

        // Add a subtle border to make individual bars more distinguishable
        draw_list->AddRect(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), IM_COL32(0, 0, 0, 80),
                           0.0f, 0, 1.0f);
      }
    }

    // Optionally draw POC (Point of Control) line - horizontal yellow line at the price level with
    // highest volume
    if (show_poc_line && poc_bucket_idx >= 0) {
      // Calculate the y-coordinate for the POC line
      float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height;  // Center of the POC bucket

      // Draw horizontal yellow line across the candle width
      float poc_line_half_width =
          (total_height) * 0.3f;  // Reduced width to avoid overlapping with candle wicks
      float poc_x_left = x_center - poc_line_half_width;
      float poc_x_right = x_center + poc_line_half_width;

      // Draw the POC line as a horizontal yellow line
      draw_list->AddLine(ImVec2(poc_x_left, poc_y), ImVec2(poc_x_right, poc_y),
                         IM_COL32(255, 255, 0, 255),  // Bright yellow color for POC
                         2.0f);                       // Line thickness
    }
  }
}

void VolumeProfilePanel::render_split_profile(const double* xs, const double* buy_vols,
                                              const double* sell_vols, int count, double height) {
  if (count <= 0) return;

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();
  const ImU32 col_buy = IM_COL32(0, 255, 0, 170);         // Green for buy volume
  const ImU32 col_sell = IM_COL32(255, 0, 0, 170);        // Red for sell volume
  const ImU32 col_center = IM_COL32(255, 255, 255, 200);  // White for center line

  // Find max volume to scale properly - considering both buy and sell volumes
  double max_vol = 0.0;
  for (int i = 0; i < count; ++i) {
    max_vol = std::max(max_vol, std::max(buy_vols[i], sell_vols[i]));
  }
  if (max_vol <= 0) max_vol = 1.0;

  for (int i = 0; i < count; ++i) {
    // Get pixel coordinates for the center of the bar (price level) - this is where the center line
    // will be
    ImVec2 center_point = ImPlot::PlotToPixels(0, xs[i]);

    // Calculate the extents of buy and sell volumes in plot coordinates
    // Buy volumes extend to the LEFT (negative x direction) from center
    double plot_scaled_buy_vol = (buy_vols[i] / max_vol) * max_volume_;
    // Sell volumes extend to the RIGHT (positive x direction) from center
    double plot_scaled_sell_vol = (sell_vols[i] / max_vol) * max_volume_;

    // Convert plot coordinates to pixel coordinates
    ImVec2 buy_end_point = ImPlot::PlotToPixels(-plot_scaled_buy_vol, xs[i]);
    ImVec2 sell_end_point = ImPlot::PlotToPixels(plot_scaled_sell_vol, xs[i]);

    // Calculate bar dimensions
    float bar_top = center_point.y - height / 2;
    float bar_bottom = center_point.y + height / 2;

    // Draw buy volume bar (left side, green) - extends from buy_end_point.x to center_point.x
    if (buy_vols[i] > 0) {
      ImVec2 bar_tl = ImVec2(buy_end_point.x, bar_top);
      ImVec2 bar_br = ImVec2(center_point.x, bar_bottom);

      draw_list->AddRectFilled(bar_tl, bar_br, col_buy);

      // Add a subtle border to make the bar more distinguishable
      draw_list->AddRect(bar_tl, bar_br, IM_COL32(0, 0, 0, 100), 0.0f, 0, 1.0f);
    }

    // Draw sell volume bar (right side, red) - extends from center_point.x to sell_end_point.x
    if (sell_vols[i] > 0) {
      ImVec2 bar_tl = ImVec2(center_point.x, bar_top);
      ImVec2 bar_br = ImVec2(sell_end_point.x, bar_bottom);

      draw_list->AddRectFilled(bar_tl, bar_br, col_sell);

      // Add a subtle border to make the bar more distinguishable
      draw_list->AddRect(bar_tl, bar_br, IM_COL32(0, 0, 0, 100), 0.0f, 0, 1.0f);
    }

    // Draw center vertical line to separate buy and sell volumes
    draw_list->AddLine(ImVec2(center_point.x, bar_top), ImVec2(center_point.x, bar_bottom),
                       col_center, 2.0f);  // Thicker line for better visibility
  }
}

// Enhanced method to render step profile histograms on candlesticks with additional features
void VolumeProfilePanel::render_enhanced_step_profile_on_candles(
    ImDrawList* draw_list, const std::vector<RenderEngine::OHLCVCandle>& candles,
    const std::vector<double>& x_coords, const std::vector<double>& y_coords_high,
    const std::vector<double>& y_coords_low, const std::vector<RenderEngine::TradeData>& trades,
    bool show_poc_line, int num_buckets_per_candle, float opacity_factor) {
  if (candles.empty() || x_coords.size() != candles.size() ||
      y_coords_high.size() != candles.size() || y_coords_low.size() != candles.size()) {
    return;
  }

  // Iterate through each candle to draw enhanced step profile histogram
  for (size_t i = 0; i < candles.size(); ++i) {
    const auto& candle = candles[i];

    // Skip if candle has no meaningful price range
    if (candle.high <= candle.low || candle.volume <= 0) continue;

    // Get the candle's price range (high - low)
    double price_range = candle.high - candle.low;
    if (price_range <= 0) continue;

    // Calculate bucket size for this candle's range
    double bucket_size = price_range / num_buckets_per_candle;

    // Create temporary buckets for this candle's price range
    std::vector<double> bucket_volumes(num_buckets_per_candle, 0.0);
    std::vector<int> bucket_counts(num_buckets_per_candle, 0);

    // Aggregate trades into buckets based on price within this candle's range
    for (const auto& trade : trades) {
      // Only consider trades within this candle's price range
      if (trade.price >= candle.low && trade.price <= candle.high) {
        int bucket_idx = static_cast<int>((trade.price - candle.low) / bucket_size);
        // Ensure we don't exceed bounds
        bucket_idx = std::max(0, std::min(bucket_idx, num_buckets_per_candle - 1));

        bucket_volumes[bucket_idx] += trade.size;
        bucket_counts[bucket_idx]++;
      }
    }

    // Find max volume in this candle's histogram for scaling
    double max_vol_in_candle = 0.0;
    int poc_bucket_idx = -1;  // Index of the bucket with highest volume (POC)

    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > max_vol_in_candle) {
        max_vol_in_candle = bucket_volumes[j];
        poc_bucket_idx = j;
      }
    }

    // If no trades found in this candle's range, skip
    if (max_vol_in_candle <= 0 || poc_bucket_idx < 0) continue;

    // Calculate screen coordinates for the step profile histogram
    float x_center = static_cast<float>(x_coords[i]);
    float y_high = static_cast<float>(y_coords_high[i]);
    float y_low = static_cast<float>(y_coords_low[i]);

    // Calculate height of each bucket in screen coordinates
    float total_height = y_low - y_high;  // Height of the candle in screen space
    float bucket_height = total_height / num_buckets_per_candle;

    // Draw enhanced step profile histogram inside the candle
    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage =
            static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.7f;  // Slightly wider for better visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is the POC bucket or not
        ImU32 color;
        if (j == poc_bucket_idx) {
          // Highlight POC bucket with bright yellow
          color =
              IM_COL32(255, 255, 0,
                       static_cast<int>(
                           240 * opacity_factor));  // Bright yellow for POC with adjustable opacity
        } else {
          // Use gradient colors based on volume intensity and position in the candle
          float volume_ratio =
              static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

          // Determine if this bucket is in the upper or lower half of the candle
          float position_ratio =
              static_cast<float>(j) / static_cast<float>(num_buckets_per_candle - 1);

          // Create a color gradient based on volume intensity and position
          if (position_ratio < 0.5f) {
            // Lower half - red for selling pressure, with intensity based on volume
            int red_intensity = static_cast<int>(255 * volume_ratio * opacity_factor);
            int alpha = static_cast<int>(180 * opacity_factor);
            color = IM_COL32(red_intensity, static_cast<int>(50 * volume_ratio * opacity_factor),
                             static_cast<int>(50 * volume_ratio * opacity_factor), alpha);
          } else {
            // Upper half - green for buying pressure, with intensity based on volume
            int green_intensity = static_cast<int>(255 * volume_ratio * opacity_factor);
            int alpha = static_cast<int>(180 * opacity_factor);
            color = IM_COL32(static_cast<int>(50 * volume_ratio * opacity_factor), green_intensity,
                             static_cast<int>(50 * volume_ratio * opacity_factor), alpha);
          }
        }

        // Draw the enhanced step profile histogram bar with a slight outline for better visibility
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);

        // Add a subtle border to make individual bars more distinguishable
        if (opacity_factor > 0.3f) {  // Only add border if not too transparent
          draw_list->AddRect(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom),
                             IM_COL32(0, 0, 0, static_cast<int>(80 * opacity_factor)), 0.0f, 0,
                             1.0f);
        }
      }
    }

    // Optionally draw POC (Point of Control) line - horizontal yellow line at the price level with
    // highest volume
    if (show_poc_line && poc_bucket_idx >= 0) {
      // Calculate the y-coordinate for the POC line
      float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height;  // Center of the POC bucket

      // Draw horizontal yellow line across the candle width
      float poc_line_half_width = (total_height) * 0.35f;  // Slightly wider for better visibility
      float poc_x_left = x_center - poc_line_half_width;
      float poc_x_right = x_center + poc_line_half_width;

      // Draw the POC line as a horizontal yellow line with adjustable opacity
      draw_list->AddLine(
          ImVec2(poc_x_left, poc_y), ImVec2(poc_x_right, poc_y),
          IM_COL32(
              255, 255, 0,
              static_cast<int>(
                  255 * opacity_factor)),  // Bright yellow color for POC with adjustable opacity
          2.0f);                           // Line thickness
    }
  }
}

// Main method to implement Step Profile rendering: draw mini histogram overlay on each candlestick
// bar showing volume distribution for that bar's price range
void VolumeProfilePanel::render_step_profile_on_candles_with_volume_distribution(
    ImDrawList* draw_list, const std::vector<RenderEngine::OHLCVCandle>& candles,
    const std::vector<double>& x_coords, const std::vector<double>& y_coords_high,
    const std::vector<double>& y_coords_low, bool show_poc_line, int num_buckets_per_candle) {
  // This method implements the core requirement: draw mini histogram overlay on each candlestick
  // bar showing volume distribution for that bar's price range
  if (candles.empty() || x_coords.size() != candles.size() ||
      y_coords_high.size() != candles.size() || y_coords_low.size() != candles.size()) {
    return;
  }

  // Get recent trades for this symbol to populate the histograms
  auto analytics = processor_->getSymbolAnalytics(symbol_id_);
  const auto& trades = analytics.recent_trades;

  // Iterate through each candle to draw step profile histogram
  for (size_t i = 0; i < candles.size(); ++i) {
    const auto& candle = candles[i];

    // Skip if candle has no meaningful price range
    if (candle.high <= candle.low || candle.volume <= 0) continue;

    // Get the candle's price range (high - low)
    double price_range = candle.high - candle.low;
    if (price_range <= 0) continue;

    // Calculate bucket size for this candle's range
    double bucket_size = price_range / num_buckets_per_candle;

    // Create temporary buckets for this candle's price range
    std::vector<double> bucket_volumes(num_buckets_per_candle, 0.0);
    std::vector<int> bucket_trade_counts(num_buckets_per_candle, 0);

    // Aggregate trades into buckets based on price within this candle's range
    for (const auto& trade : trades) {
      // Only consider trades within this candle's price range
      if (trade.price >= candle.low && trade.price <= candle.high) {
        int bucket_idx = static_cast<int>((trade.price - candle.low) / bucket_size);
        // Ensure we don't exceed bounds
        bucket_idx = std::max(0, std::min(bucket_idx, num_buckets_per_candle - 1));

        bucket_volumes[bucket_idx] += trade.size;
        bucket_trade_counts[bucket_idx]++;
      }
    }

    // Find max volume in this candle's histogram for scaling
    double max_vol_in_candle = 0.0;
    int poc_bucket_idx = -1;  // Index of the bucket with highest volume (POC)

    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > max_vol_in_candle) {
        max_vol_in_candle = bucket_volumes[j];
        poc_bucket_idx = j;
      }
    }

    // If no trades found in this candle's range, skip
    if (max_vol_in_candle <= 0 || poc_bucket_idx < 0) continue;

    // Calculate screen coordinates for the step profile histogram
    float x_center = static_cast<float>(x_coords[i]);
    float y_high = static_cast<float>(y_coords_high[i]);
    float y_low = static_cast<float>(y_coords_low[i]);

    // Calculate height of each bucket in screen coordinates
    float total_height = y_low - y_high;  // Height of the candle in screen space
    float bucket_height = total_height / num_buckets_per_candle;

    // Draw step profile histogram inside the candle
    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage =
            static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.6f;  // Use 60% of height as width for visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is the POC bucket or not
        ImU32 color;
        if (j == poc_bucket_idx) {
          // Highlight POC bucket with bright yellow
          color = IM_COL32(255, 255, 0, 240);  // Bright yellow for POC with high opacity
        } else {
          // Use gradient colors based on volume intensity and position in the candle
          float volume_ratio =
              static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

          // Determine if this bucket is in the upper or lower half of the candle
          float position_ratio =
              static_cast<float>(j) / static_cast<float>(num_buckets_per_candle - 1);

          // Create a color gradient based on volume intensity and position
          if (position_ratio < 0.5f) {
            // Lower half - red for selling pressure, with intensity based on volume
            int red_intensity = static_cast<int>(255 * volume_ratio);
            color = IM_COL32(red_intensity, static_cast<int>(50 * volume_ratio),
                             static_cast<int>(50 * volume_ratio), 180);
          } else {
            // Upper half - green for buying pressure, with intensity based on volume
            int green_intensity = static_cast<int>(255 * volume_ratio);
            color = IM_COL32(static_cast<int>(50 * volume_ratio), green_intensity,
                             static_cast<int>(50 * volume_ratio), 180);
          }
        }

        // Draw the step profile histogram bar with a slight outline for better visibility
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);

        // Add a subtle border to make individual bars more distinguishable
        draw_list->AddRect(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), IM_COL32(0, 0, 0, 80),
                           0.0f, 0, 1.0f);
      }
    }

    // Optionally draw POC (Point of Control) line - horizontal yellow line at the price level with
    // highest volume
    if (show_poc_line && poc_bucket_idx >= 0) {
      // Calculate the y-coordinate for the POC line
      float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height;  // Center of the POC bucket

      // Draw horizontal yellow line across the candle width
      float poc_line_half_width =
          (total_height) * 0.3f;  // Reduced width to avoid overlapping with candle wicks
      float poc_x_left = x_center - poc_line_half_width;
      float poc_x_right = x_center + poc_line_half_width;

      // Draw the POC line as a horizontal yellow line
      draw_list->AddLine(ImVec2(poc_x_left, poc_y), ImVec2(poc_x_right, poc_y),
                         IM_COL32(255, 255, 0, 255),  // Bright yellow color for POC
                         2.0f);                       // Line thickness
    }
  }
}

// Additional method to render step profile with enhanced visualization options
void VolumeProfilePanel::render_enhanced_step_profile_with_volume_distribution(
    ImDrawList* draw_list, const std::vector<RenderEngine::OHLCVCandle>& candles,
    const std::vector<double>& x_coords, const std::vector<double>& y_coords_high,
    const std::vector<double>& y_coords_low, bool show_poc_line, int num_buckets_per_candle,
    float bar_opacity, bool use_transparent_background) {
  // This method implements an enhanced version of the Step Profile rendering with additional
  // customization options
  if (candles.empty() || x_coords.size() != candles.size() ||
      y_coords_high.size() != candles.size() || y_coords_low.size() != candles.size()) {
    return;
  }

  // Get recent trades for this symbol to populate the histograms
  auto analytics = processor_->getSymbolAnalytics(symbol_id_);
  const auto& trades = analytics.recent_trades;

  // Iterate through each candle to draw enhanced step profile histogram
  for (size_t i = 0; i < candles.size(); ++i) {
    const auto& candle = candles[i];

    // Skip if candle has no meaningful price range
    if (candle.high <= candle.low || candle.volume <= 0) continue;

    // Get the candle's price range (high - low)
    double price_range = candle.high - candle.low;
    if (price_range <= 0) continue;

    // Calculate bucket size for this candle's range
    double bucket_size = price_range / num_buckets_per_candle;

    // Create temporary buckets for this candle's price range
    std::vector<double> bucket_volumes(num_buckets_per_candle, 0.0);
    std::vector<int> bucket_trade_counts(num_buckets_per_candle, 0);

    // Aggregate trades into buckets based on price within this candle's range
    for (const auto& trade : trades) {
      // Only consider trades within this candle's price range
      if (trade.price >= candle.low && trade.price <= candle.high) {
        int bucket_idx = static_cast<int>((trade.price - candle.low) / bucket_size);
        // Ensure we don't exceed bounds
        bucket_idx = std::max(0, std::min(bucket_idx, num_buckets_per_candle - 1));

        bucket_volumes[bucket_idx] += trade.size;
        bucket_trade_counts[bucket_idx]++;
      }
    }

    // Find max volume in this candle's histogram for scaling
    double max_vol_in_candle = 0.0;
    int poc_bucket_idx = -1;  // Index of the bucket with highest volume (POC)

    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > max_vol_in_candle) {
        max_vol_in_candle = bucket_volumes[j];
        poc_bucket_idx = j;
      }
    }

    // If no trades found in this candle's range, skip
    if (max_vol_in_candle <= 0 || poc_bucket_idx < 0) continue;

    // Calculate screen coordinates for the step profile histogram
    float x_center = static_cast<float>(x_coords[i]);
    float y_high = static_cast<float>(y_coords_high[i]);
    float y_low = static_cast<float>(y_coords_low[i]);

    // Calculate height of each bucket in screen coordinates
    float total_height = y_low - y_high;  // Height of the candle in screen space
    float bucket_height = total_height / num_buckets_per_candle;

    // Optionally draw a subtle background for the entire candle to highlight the histogram area
    if (use_transparent_background) {
      draw_list->AddRectFilled(ImVec2(x_center - total_height * 0.4f, y_high),
                               ImVec2(x_center + total_height * 0.1f, y_low),
                               IM_COL32(0, 0, 0, 30));  // Very subtle dark background
    }

    // Draw step profile histogram inside the candle
    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage =
            static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.7f;  // Slightly wider for better visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is the POC bucket or not
        ImU32 color;
        if (j == poc_bucket_idx) {
          // Highlight POC bucket with bright yellow
          color =
              IM_COL32(255, 255, 0,
                       static_cast<int>(
                           255 * bar_opacity));  // Bright yellow for POC with adjustable opacity
        } else {
          // Use gradient colors based on volume intensity and position in the candle
          float volume_ratio =
              static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

          // Determine if this bucket is in the upper or lower half of the candle
          float position_ratio =
              static_cast<float>(j) / static_cast<float>(num_buckets_per_candle - 1);

          // Create a color gradient based on volume intensity and position
          if (position_ratio < 0.5f) {
            // Lower half - red for selling pressure, with intensity based on volume
            int red_intensity = static_cast<int>(255 * volume_ratio * bar_opacity);
            int alpha = static_cast<int>(180 * bar_opacity);
            color = IM_COL32(red_intensity, static_cast<int>(50 * volume_ratio * bar_opacity),
                             static_cast<int>(50 * volume_ratio * bar_opacity), alpha);
          } else {
            // Upper half - green for buying pressure, with intensity based on volume
            int green_intensity = static_cast<int>(255 * volume_ratio * bar_opacity);
            int alpha = static_cast<int>(180 * bar_opacity);
            color = IM_COL32(static_cast<int>(50 * volume_ratio * bar_opacity), green_intensity,
                             static_cast<int>(50 * volume_ratio * bar_opacity), alpha);
          }
        }

        // Draw the enhanced step profile histogram bar with a slight outline for better visibility
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);

        // Add a subtle border to make individual bars more distinguishable
        if (bar_opacity > 0.3f) {  // Only add border if not too transparent
          draw_list->AddRect(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom),
                             IM_COL32(0, 0, 0, static_cast<int>(80 * bar_opacity)), 0.0f, 0, 1.0f);
        }
      }
    }

    // Optionally draw POC (Point of Control) line - horizontal yellow line at the price level with
    // highest volume
    if (show_poc_line && poc_bucket_idx >= 0) {
      // Calculate the y-coordinate for the POC line
      float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height;  // Center of the POC bucket

      // Draw horizontal yellow line across the candle width
      float poc_line_half_width = (total_height) * 0.35f;  // Slightly wider for better visibility
      float poc_x_left = x_center - poc_line_half_width;
      float poc_x_right = x_center + poc_line_half_width;

      // Draw the POC line as a horizontal yellow line with adjustable opacity
      draw_list->AddLine(
          ImVec2(poc_x_left, poc_y), ImVec2(poc_x_right, poc_y),
          IM_COL32(255, 255, 0,
                   static_cast<int>(
                       255 * bar_opacity)),  // Bright yellow color for POC with adjustable opacity
          2.0f);                             // Line thickness
    }
  }
}

// Specific implementation for Step Profile rendering: draw mini histogram overlay on each
// candlestick bar showing volume distribution for that bar's price range - this is the main method
// for the task requirement
void VolumeProfilePanel::render_step_profile_histograms_on_candle_bars(
    ImDrawList* draw_list, const std::vector<RenderEngine::OHLCVCandle>& candles,
    const std::vector<double>& x_coords, const std::vector<double>& y_coords_high,
    const std::vector<double>& y_coords_low, bool show_poc_line, int num_buckets_per_candle) {
  // This method implements the core requirement: draw mini histogram overlay on each candlestick
  // bar showing volume distribution for that bar's price range with POC line
  if (candles.empty() || x_coords.size() != candles.size() ||
      y_coords_high.size() != candles.size() || y_coords_low.size() != candles.size()) {
    return;
  }

  // Get recent trades for this symbol to populate the histograms
  auto analytics = processor_->getSymbolAnalytics(symbol_id_);
  const auto& trades = analytics.recent_trades;

  // Iterate through each candle to draw step profile histogram
  for (size_t i = 0; i < candles.size(); ++i) {
    const auto& candle = candles[i];

    // Skip if candle has no meaningful price range
    if (candle.high <= candle.low || candle.volume <= 0) continue;

    // Get the candle's price range (high - low)
    double price_range = candle.high - candle.low;
    if (price_range <= 0) continue;

    // Calculate bucket size for this candle's range
    double bucket_size = price_range / num_buckets_per_candle;

    // Create temporary buckets for this candle's price range
    std::vector<double> bucket_volumes(num_buckets_per_candle, 0.0);
    std::vector<int> bucket_trade_counts(num_buckets_per_candle, 0);

    // Aggregate trades into buckets based on price within this candle's range
    for (const auto& trade : trades) {
      // Only consider trades within this candle's price range
      if (trade.price >= candle.low && trade.price <= candle.high) {
        int bucket_idx = static_cast<int>((trade.price - candle.low) / bucket_size);
        // Ensure we don't exceed bounds
        bucket_idx = std::max(0, std::min(bucket_idx, num_buckets_per_candle - 1));

        bucket_volumes[bucket_idx] += trade.size;
        bucket_trade_counts[bucket_idx]++;
      }
    }

    // Find max volume in this candle's histogram for scaling
    double max_vol_in_candle = 0.0;
    int poc_bucket_idx = -1;  // Index of the bucket with highest volume (POC)

    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > max_vol_in_candle) {
        max_vol_in_candle = bucket_volumes[j];
        poc_bucket_idx = j;
      }
    }

    // If no trades found in this candle's range, skip
    if (max_vol_in_candle <= 0 || poc_bucket_idx < 0) continue;

    // Calculate screen coordinates for the step profile histogram
    float x_center = static_cast<float>(x_coords[i]);
    float y_high = static_cast<float>(y_coords_high[i]);
    float y_low = static_cast<float>(y_coords_low[i]);

    // Calculate height of each bucket in screen coordinates
    float total_height = y_low - y_high;  // Height of the candle in screen space
    float bucket_height = total_height / num_buckets_per_candle;

    // Draw step profile histogram inside the candle
    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage =
            static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.6f;  // Use 60% of height as width for visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is the POC bucket or not
        ImU32 color;
        if (j == poc_bucket_idx) {
          // Highlight POC bucket with bright yellow
          color = IM_COL32(255, 255, 0, 240);  // Bright yellow for POC with high opacity
        } else {
          // Use gradient colors based on volume intensity and position in the candle
          float volume_ratio =
              static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

          // Determine if this bucket is in the upper or lower half of the candle
          float position_ratio =
              static_cast<float>(j) / static_cast<float>(num_buckets_per_candle - 1);

          // Create a color gradient based on volume intensity and position
          if (position_ratio < 0.5f) {
            // Lower half - red for selling pressure, with intensity based on volume
            int red_intensity = static_cast<int>(255 * volume_ratio);
            color = IM_COL32(red_intensity, static_cast<int>(50 * volume_ratio),
                             static_cast<int>(50 * volume_ratio), 180);
          } else {
            // Upper half - green for buying pressure, with intensity based on volume
            int green_intensity = static_cast<int>(255 * volume_ratio);
            color = IM_COL32(static_cast<int>(50 * volume_ratio), green_intensity,
                             static_cast<int>(50 * volume_ratio), 180);
          }
        }

        // Draw the step profile histogram bar with a slight outline for better visibility
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);

        // Add a subtle border to make individual bars more distinguishable
        draw_list->AddRect(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), IM_COL32(0, 0, 0, 80),
                           0.0f, 0, 1.0f);
      }
    }

    // Draw POC (Point of Control) line - horizontal yellow line at the price level with highest
    // volume
    if (show_poc_line && poc_bucket_idx >= 0) {
      // Calculate the y-coordinate for the POC line
      float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height;  // Center of the POC bucket

      // Draw horizontal yellow line across the candle width
      float poc_line_half_width =
          (total_height) * 0.3f;  // Reduced width to avoid overlapping with candle wicks
      float poc_x_left = x_center - poc_line_half_width;
      float poc_x_right = x_center + poc_line_half_width;

      // Draw the POC line as a horizontal yellow line
      draw_list->AddLine(ImVec2(poc_x_left, poc_y), ImVec2(poc_x_right, poc_y),
                         IM_COL32(255, 255, 0, 255),  // Bright yellow color for POC
                         2.5f);                       // Slightly thicker line for better visibility
    }
  }
}

// Enhanced method with additional visualization options for Step Profile rendering
void VolumeProfilePanel::render_enhanced_step_profile_histograms_on_candle_bars(
    ImDrawList* draw_list, const std::vector<RenderEngine::OHLCVCandle>& candles,
    const std::vector<double>& x_coords, const std::vector<double>& y_coords_high,
    const std::vector<double>& y_coords_low, bool show_poc_line, int num_buckets_per_candle,
    float opacity, bool show_labels) {
  // This enhanced method implements the Step Profile rendering with additional visualization
  // options
  if (candles.empty() || x_coords.size() != candles.size() ||
      y_coords_high.size() != candles.size() || y_coords_low.size() != candles.size()) {
    return;
  }

  // Get recent trades for this symbol to populate the histograms
  auto analytics = processor_->getSymbolAnalytics(symbol_id_);
  const auto& trades = analytics.recent_trades;

  // Iterate through each candle to draw enhanced step profile histogram
  for (size_t i = 0; i < candles.size(); ++i) {
    const auto& candle = candles[i];

    // Skip if candle has no meaningful price range
    if (candle.high <= candle.low || candle.volume <= 0) continue;

    // Get the candle's price range (high - low)
    double price_range = candle.high - candle.low;
    if (price_range <= 0) continue;

    // Calculate bucket size for this candle's range
    double bucket_size = price_range / num_buckets_per_candle;

    // Create temporary buckets for this candle's price range
    std::vector<double> bucket_volumes(num_buckets_per_candle, 0.0);
    std::vector<int> bucket_trade_counts(num_buckets_per_candle, 0);

    // Aggregate trades into buckets based on price within this candle's range
    for (const auto& trade : trades) {
      // Only consider trades within this candle's price range
      if (trade.price >= candle.low && trade.price <= candle.high) {
        int bucket_idx = static_cast<int>((trade.price - candle.low) / bucket_size);
        // Ensure we don't exceed bounds
        bucket_idx = std::max(0, std::min(bucket_idx, num_buckets_per_candle - 1));

        bucket_volumes[bucket_idx] += trade.size;
        bucket_trade_counts[bucket_idx]++;
      }
    }

    // Find max volume in this candle's histogram for scaling
    double max_vol_in_candle = 0.0;
    int poc_bucket_idx = -1;  // Index of the bucket with highest volume (POC)

    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > max_vol_in_candle) {
        max_vol_in_candle = bucket_volumes[j];
        poc_bucket_idx = j;
      }
    }

    // If no trades found in this candle's range, skip
    if (max_vol_in_candle <= 0 || poc_bucket_idx < 0) continue;

    // Calculate screen coordinates for the step profile histogram
    float x_center = static_cast<float>(x_coords[i]);
    float y_high = static_cast<float>(y_coords_high[i]);
    float y_low = static_cast<float>(y_coords_low[i]);

    // Calculate height of each bucket in screen coordinates
    float total_height = y_low - y_high;  // Height of the candle in screen space
    float bucket_height = total_height / num_buckets_per_candle;

    // Draw enhanced step profile histogram inside the candle
    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage =
            static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.7f;  // Slightly wider for better visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is the POC bucket or not
        ImU32 color;
        if (j == poc_bucket_idx) {
          // Highlight POC bucket with bright yellow
          color = IM_COL32(
              255, 255, 0,
              static_cast<int>(240 * opacity));  // Bright yellow for POC with adjustable opacity
        } else {
          // Use gradient colors based on volume intensity and position in the candle
          float volume_ratio =
              static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

          // Determine if this bucket is in the upper or lower half of the candle
          float position_ratio =
              static_cast<float>(j) / static_cast<float>(num_buckets_per_candle - 1);

          // Create a color gradient based on volume intensity and position
          if (position_ratio < 0.5f) {
            // Lower half - red for selling pressure, with intensity based on volume
            int red_intensity = static_cast<int>(255 * volume_ratio * opacity);
            int alpha = static_cast<int>(180 * opacity);
            color = IM_COL32(red_intensity, static_cast<int>(50 * volume_ratio * opacity),
                             static_cast<int>(50 * volume_ratio * opacity), alpha);
          } else {
            // Upper half - green for buying pressure, with intensity based on volume
            int green_intensity = static_cast<int>(255 * volume_ratio * opacity);
            int alpha = static_cast<int>(180 * opacity);
            color = IM_COL32(static_cast<int>(50 * volume_ratio * opacity), green_intensity,
                             static_cast<int>(50 * volume_ratio * opacity), alpha);
          }
        }

        // Draw the enhanced step profile histogram bar with a slight outline for better visibility
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);

        // Add a subtle border to make individual bars more distinguishable
        if (opacity > 0.3f) {  // Only add border if not too transparent
          draw_list->AddRect(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom),
                             IM_COL32(0, 0, 0, static_cast<int>(80 * opacity)), 0.0f, 0, 1.0f);
        }
      }
    }

    // Draw POC (Point of Control) line - horizontal yellow line at the price level with highest
    // volume
    if (show_poc_line && poc_bucket_idx >= 0) {
      // Calculate the y-coordinate for the POC line
      float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height;  // Center of the POC bucket

      // Draw horizontal yellow line across the candle width
      float poc_line_half_width = (total_height) * 0.35f;  // Slightly wider for better visibility
      float poc_x_left = x_center - poc_line_half_width;
      float poc_x_right = x_center + poc_line_half_width;

      // Draw the POC line as a horizontal yellow line with adjustable opacity
      draw_list->AddLine(
          ImVec2(poc_x_left, poc_y), ImVec2(poc_x_right, poc_y),
          IM_COL32(255, 255, 0,
                   static_cast<int>(
                       255 * opacity)),  // Bright yellow color for POC with adjustable opacity
          2.5f);                         // Slightly thicker line for better visibility
    }

    // Optionally show labels for the mini histogram
    if (show_labels) {
      // Add a small label indicating this is a volume profile histogram
      char label[32];
      snprintf(label, sizeof(label), "%.0f", candle.volume);

      // Position the label at the top of the candle
      ImVec2 label_pos = ImVec2(x_center, y_high - 15.0f);

      // Draw the label with appropriate color
      draw_list->AddText(label_pos, IM_COL32(255, 255, 255, static_cast<int>(200 * opacity)),
                         label);
    }
  }
}

// Public method to render step profile histograms on candle bars - this can be called from other
// components
void VolumeProfilePanel::drawMiniHistogramOverlay(
    ImDrawList* draw_list, const std::vector<RenderEngine::OHLCVCandle>& candles,
    const std::vector<double>& x_coords, const std::vector<double>& y_coords_high,
    const std::vector<double>& y_coords_low, bool show_poc_line, int num_buckets_per_candle) {
  render_step_profile_histograms_on_candle_bars(draw_list, candles, x_coords, y_coords_high,
                                                y_coords_low, show_poc_line,
                                                num_buckets_per_candle);
}

// Method to handle mouse drag interaction for custom profile creation
void VolumeProfilePanel::handleMouseDragInteraction() {
  auto& interaction_mgr = InteractionManager::getInstance();

  // Check if we're in custom profile mode
  if (profile_mode_ != ProfileMode::Custom) {
    return;
  }

  // Get the current mouse position in plot coordinates if we're in a plot context
  ImVec2 mouse_pos = ImGui::GetMousePos();

  // Check if a time range selection is active
  if (interaction_mgr.isTimeRangeSelectionActive()) {
    // Update the time range selection with current mouse position
    // For volume profile, we might want to use screen coordinates or plot coordinates depending on
    // context
    interaction_mgr.updateTimeRangeSelection(mouse_pos);

    // Get the current time range
    auto time_range = interaction_mgr.getTimeRangeSelection();

    // Calculate profile for the selected time range
    calculateProfileForTimeRange(time_range.first, time_range.second);
  }
  // If no drag is active but we were previously dragging, finalize the selection
  else if (!interaction_mgr.isMouseDragActive() && use_custom_time_range_) {
    // The drag has ended, we can now use the selected time range
    auto time_range = interaction_mgr.getTimeRangeSelection();
    custom_start_time_ = time_range.first;
    custom_end_time_ = time_range.second;
  }
}

// Method to render custom profile overlay when mouse drag is active
void VolumeProfilePanel::renderCustomProfileOverlay(ImDrawList* draw_list) {
  auto& interaction_mgr = InteractionManager::getInstance();

  // Only render if we're in custom profile mode and have an active time range selection
  if (profile_mode_ != ProfileMode::Custom || !interaction_mgr.isTimeRangeSelectionActive()) {
    return;
  }

  auto time_range = interaction_mgr.getTimeRangeSelection();
  double start_time = time_range.first;
  double end_time = time_range.second;

  // Get plot limits to determine the Y range for vertical lines
  ImPlotRect plot_limits = ImPlot::GetPlotLimits();

  // Draw start time vertical line (green)
  double start_line_x[2] = {start_time, start_time};
  double start_line_y[2] = {plot_limits.Y.Min, plot_limits.Y.Max};

  // ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.0f, 1.0f, 0.0f, 0.8f)); // Green
  ImPlot::PlotLine("Start Time Selection", start_line_x, start_line_y, 2);
  ImPlot::PopStyleColor();

  // Draw end time vertical line (red)
  double end_line_x[2] = {end_time, end_time};
  double end_line_y[2] = {plot_limits.Y.Min, plot_limits.Y.Max};

  // ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.0f, 0.0f, 0.8f)); // Red
  ImPlot::PlotLine("End Time Selection", end_line_x, end_line_y, 2);
  ImPlot::PopStyleColor();

  // Draw a shaded area between the two time points
  if (start_time != end_time) {
    // Draw a semi-transparent rectangle between the two time points
    ImVec2 area_start = ImPlot::PlotToPixels(start_time, plot_limits.Y.Min);
    ImVec2 area_end = ImPlot::PlotToPixels(end_time, plot_limits.Y.Max);

    // Swap if needed to ensure area_start.x < area_end.x
    if (area_start.x > area_end.x) {
      ImVec2 temp = area_start;
      area_start = area_end;
      area_end = temp;
    }

    // Draw the shaded area
    draw_list->AddRectFilled(area_start, area_end,
                             IM_COL32(0, 100, 255, 50));  // Semi-transparent blue overlay
  }
}

// Method to calculate profile for a specific time range
void VolumeProfilePanel::calculateProfileForTimeRange(double start_time, double end_time) {
  if (!processor_ || symbol_id_ == 0) return;

  // Get all trades for this symbol
  auto analytics = processor_->getSymbolAnalytics(symbol_id_);
  auto all_trades = analytics.recent_trades;

  // Filter trades based on the time range
  std::vector<RenderEngine::TradeData> filtered_trades;

  for (const auto& trade : all_trades) {
    // Convert timestamp to plot coordinate equivalent if needed
    // For now, assuming the time values are already in the same coordinate system
    if (trade.timestamp >= start_time && trade.timestamp <= end_time) {
      filtered_trades.push_back(trade);
    }
  }

  // If no trades in the selected range, clear the profile
  if (filtered_trades.empty()) {
    volume_profile_.clear();
    max_volume_ = 0.0;
    poc_price_ = 0.0;
    return;
  }

  // Calculate the volume profile based on the filtered trades
  // Find the price range of the filtered trades
  double min_price = std::numeric_limits<double>::max();
  double max_price = std::numeric_limits<double>::lowest();

  for (const auto& trade : filtered_trades) {
    min_price = std::min(min_price, trade.price);
    max_price = std::max(max_price, trade.price);
  }

  if (max_price <= min_price) return;

  // Compute bucket size
  double range = max_price - min_price;
  price_bucket_size_ = range / NUM_PRICE_LEVELS;
  if (price_bucket_size_ <= 0) price_bucket_size_ = 1.0;

  // Reset profile
  volume_profile_.clear();
  volume_profile_.resize(NUM_PRICE_LEVELS);

  for (size_t i = 0; i < NUM_PRICE_LEVELS; ++i) {
    volume_profile_[i].price = min_price + (i + 0.5) * price_bucket_size_;
    volume_profile_[i].buy_volume = 0;
    volume_profile_[i].sell_volume = 0;
    volume_profile_[i].total_volume = 0;
  }

  // Aggregate filtered trades into buckets
  for (const auto& trade : filtered_trades) {
    size_t bucket = static_cast<size_t>((trade.price - min_price) / price_bucket_size_);
    bucket = std::min(bucket, NUM_PRICE_LEVELS - 1);

    if (trade.is_buy) {
      volume_profile_[bucket].buy_volume += trade.size;
    } else {
      volume_profile_[bucket].sell_volume += trade.size;
    }
    volume_profile_[bucket].total_volume += trade.size;
  }

  // Find POC and max volume
  max_volume_ = 0;
  poc_price_ = volume_profile_[0].price;
  double poc_volume = 0;

  for (const auto& level : volume_profile_) {
    double total = level.buy_volume + level.sell_volume;
    max_volume_ = std::max(max_volume_, std::max(level.buy_volume, level.sell_volume));
    if (total > poc_volume) {
      poc_volume = total;
      poc_price_ = level.price;
    }
  }

  // Ensure poc_price_ is always set to the price level with the highest total volume
  if (poc_volume > 0) {
    // Double check to make sure we have the correct POC
    double current_max_volume = 0;
    for (const auto& level : volume_profile_) {
      double total = level.buy_volume + level.sell_volume;
      if (total > current_max_volume) {
        current_max_volume = total;
        poc_price_ = level.price;
      }
    }
  }

  // Calculate Value Area
  calculate_value_area();
}

// Helper method to get minimum available time from trades
double VolumeProfilePanel::getMinTimeAvailable() {
  if (!processor_ || symbol_id_ == 0) return 0.0;

  auto analytics = processor_->getSymbolAnalytics(symbol_id_);
  const auto& trades = analytics.recent_trades;

  if (trades.empty()) return 0.0;

  double min_time = trades[0].timestamp;
  for (const auto& trade : trades) {
    if (trade.timestamp < min_time) min_time = trade.timestamp;
  }

  return min_time;
}

// Helper method to get maximum available time from trades
double VolumeProfilePanel::getMaxTimeAvailable() {
  if (!processor_ || symbol_id_ == 0) return 0.0;

  auto analytics = processor_->getSymbolAnalytics(symbol_id_);
  const auto& trades = analytics.recent_trades;

  if (trades.empty()) return 0.0;

  double max_time = trades[0].timestamp;
  for (const auto& trade : trades) {
    if (trade.timestamp > max_time) max_time = trade.timestamp;
  }

  return max_time;
}

// Helper method to get the total time range available
double VolumeProfilePanel::getTimeRangeAvailable() {
  double min_time = getMinTimeAvailable();
  double max_time = getMaxTimeAvailable();

  if (max_time <= min_time) return 1.0;  // Return a default value if invalid

  return max_time - min_time;
}

// Method to store current profile as yesterday's profile
void VolumeProfilePanel::store_current_as_yesterday_profile() {
  yesterday_volume_profile_ = volume_profile_;
  yesterday_poc_price_ = poc_price_;
  yesterday_max_volume_ = max_volume_;
  yesterday_vah_price_ = vah_price_;
  yesterday_val_price_ = val_price_;

  // Calculate value area for yesterday's profile
  calculate_yesterday_value_area();
}

// Method to calculate value area for yesterday's profile
void VolumeProfilePanel::calculate_yesterday_value_area() {
  if (yesterday_volume_profile_.empty()) {
    yesterday_vah_price_ = 0.0;
    yesterday_val_price_ = 0.0;
    return;
  }

  // Calculate total volume in yesterday's profile
  double total_volume = 0.0;
  for (const auto& level : yesterday_volume_profile_) {
    total_volume += level.total_volume;
  }

  if (total_volume <= 0) {
    yesterday_vah_price_ = 0.0;
    yesterday_val_price_ = 0.0;
    return;
  }

  // Target volume for value area (based on profile_settings_.vaPercent % of total volume)
  double target_volume = (static_cast<double>(profile_settings_.vaPercent) / 100.0) * total_volume;

  // Find the POC index (Point of Control - highest volume)
  size_t poc_index = 0;
  double max_total_volume = 0.0;
  for (size_t i = 0; i < yesterday_volume_profile_.size(); ++i) {
    double total =
        yesterday_volume_profile_[i].buy_volume + yesterday_volume_profile_[i].sell_volume;
    if (total > max_total_volume) {
      max_total_volume = total;
      poc_index = i;
    }
  }

  // Expand from POC outward to capture the required volume, centered around POC
  size_t start_idx = poc_index;
  size_t end_idx = poc_index;
  double current_volume = yesterday_volume_profile_[poc_index].total_volume;

  // Expand upward (higher prices) and downward (lower prices) alternately
  // until we reach the target volume, keeping the expansion balanced around POC
  while (current_volume < target_volume) {
    // Decide whether to expand up or down
    bool expand_up = false;
    bool expand_down = false;

    // Check if we can expand in each direction
    if (start_idx > 0) expand_down = true;
    if (end_idx < yesterday_volume_profile_.size() - 1) expand_up = true;

    // If we can't expand in either direction, break
    if (!expand_up && !expand_down) break;

    // If we can only expand in one direction, do that
    if (!expand_up && expand_down) {
      start_idx--;
      current_volume += yesterday_volume_profile_[start_idx].total_volume;
    } else if (expand_up && !expand_down) {
      end_idx++;
      current_volume += yesterday_volume_profile_[end_idx].total_volume;
    } else {
      // We can expand in both directions - try to keep it centered around POC
      // Calculate potential volumes for each direction
      double vol_up = yesterday_volume_profile_[end_idx + 1].total_volume;
      double vol_down = yesterday_volume_profile_[start_idx - 1].total_volume;

      // To center around POC, we should try to balance the expansion
      // If both sides have similar volume, expand the side that currently has a smaller range
      size_t current_upper_range = end_idx - poc_index;
      size_t current_lower_range = poc_index - start_idx;

      if (current_lower_range < current_upper_range) {
        // Current lower range is smaller, expand downward to balance
        start_idx--;
        current_volume += yesterday_volume_profile_[start_idx].total_volume;
      } else if (current_upper_range < current_lower_range) {
        // Current upper range is smaller, expand upward to balance
        end_idx++;
        current_volume += yesterday_volume_profile_[end_idx].total_volume;
      } else {
        // Ranges are equal, expand toward the side with more volume to capture more volume
        // efficiently
        if (vol_up >= vol_down) {
          end_idx++;
          current_volume += yesterday_volume_profile_[end_idx].total_volume;
        } else {
          start_idx--;
          current_volume += yesterday_volume_profile_[start_idx].total_volume;
        }
      }
    }

    // If we've captured enough volume, break
    if (current_volume >= target_volume) break;
  }

  // Set the VAH and VAL prices for yesterday's profile
  yesterday_vah_price_ = yesterday_volume_profile_[end_idx].price;
  yesterday_val_price_ = yesterday_volume_profile_[start_idx].price;
}

// Method to render yesterday's step profile with reduced opacity
void VolumeProfilePanel::render_yesterday_step_profile(const double* xs, const double* ys,
                                                       const double* neg_ys, int count,
                                                       double height, float opacity) {
  if (count <= 0) return;
  ImDrawList* draw_list = ImPlot::GetPlotDrawList();

  // Adjust colors for 30% opacity
  const ImU32 col_pos =
      IM_COL32(0, 255, 0, static_cast<int>(170 * opacity));  // Green for positive with opacity
  const ImU32 col_neg =
      IM_COL32(255, 0, 0, static_cast<int>(170 * opacity));  // Red for negative with opacity
  const ImU32 col_poc =
      IM_COL32(255, 255, 0, static_cast<int>(255 * opacity));  // Yellow for POC with opacity

  // Find the POC (Point of Control) - the price level with highest total volume
  int poc_index = -1;
  double max_total_volume = 0.0;

  for (int i = 0; i < count; ++i) {
    double total_volume = std::abs(ys[i]) + std::abs(neg_ys[i]);
    if (total_volume > max_total_volume) {
      max_total_volume = total_volume;
      poc_index = i;
    }
  }

  for (int i = 0; i < count; ++i) {
    // Determine if this is the POC bar
    bool is_poc_bar = (i == poc_index && max_total_volume > 0);
    ImU32 current_col_pos = is_poc_bar ? col_poc : col_pos;
    ImU32 current_col_neg = is_poc_bar ? col_poc : col_neg;

    if (ys[i] != 0) {
      ImVec2 p1 = ImPlot::PlotToPixels(0, xs[i]);  // x-axis is volume, y-axis is price
      ImVec2 p2 = ImPlot::PlotToPixels(ys[i], xs[i]);

      // Draw step-style bar
      ImVec2 bar_tl = ImVec2(std::min(p1.x, p2.x), p1.y - height / 2);
      ImVec2 bar_br = ImVec2(std::max(p1.x, p2.x), p1.y + height / 2);

      draw_list->AddRectFilled(bar_tl, bar_br, current_col_pos);
    }

    if (neg_ys[i] != 0) {
      ImVec2 p1 = ImPlot::PlotToPixels(0, xs[i]);
      ImVec2 p2 = ImPlot::PlotToPixels(neg_ys[i], xs[i]);

      // Draw step-style bar for negative values
      ImVec2 bar_tl = ImVec2(std::min(p1.x, p2.x), p1.y - height / 2);
      ImVec2 bar_br = ImVec2(std::max(p1.x, p2.x), p1.y + height / 2);

      draw_list->AddRectFilled(bar_tl, bar_br, current_col_neg);
    }
  }

  // Draw horizontal yellow POC line at the price level with highest volume
  // Use the locally calculated POC for consistency with highlighted bar
  if (poc_index >= 0 && max_total_volume > 0) {
    // Get the current plot limits to draw the POC line across the full visible width
    ImPlotRect plot_limits = ImPlot::GetPlotLimits();

    // Use the plot limits to draw the line across the full width of the visible area
    ImVec2 poc_start = ImPlot::PlotToPixels(plot_limits.X.Min, xs[poc_index]);
    ImVec2 poc_end = ImPlot::PlotToPixels(plot_limits.X.Max, xs[poc_index]);

    // Draw the horizontal POC line - make it more prominent with consistent styling
    // Use the same color as in other profile modes for consistency
    draw_list->AddLine(poc_start, poc_end, IM_COL32(255, 255, 0, static_cast<int>(255 * opacity)),
                       3.0f);  // Bright yellow with increased thickness and opacity
  }
}

// Method to render yesterday's split profile with reduced opacity
void VolumeProfilePanel::render_yesterday_split_profile(const double* xs, const double* buy_vols,
                                                        const double* sell_vols, int count,
                                                        double height, float opacity) {
  if (count <= 0) return;

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();

  // Adjust colors for 30% opacity
  const ImU32 col_buy =
      IM_COL32(0, 255, 0, static_cast<int>(170 * opacity));  // Green for buy volume with opacity
  const ImU32 col_sell =
      IM_COL32(255, 0, 0, static_cast<int>(170 * opacity));  // Red for sell volume with opacity
  const ImU32 col_center = IM_COL32(
      255, 255, 255, static_cast<int>(200 * opacity));  // White for center line with opacity

  // Find max volume to scale properly - considering both buy and sell volumes
  double max_vol = 0.0;
  for (int i = 0; i < count; ++i) {
    max_vol = std::max(max_vol, std::max(buy_vols[i], sell_vols[i]));
  }
  if (max_vol <= 0) max_vol = 1.0;

  for (int i = 0; i < count; ++i) {
    // Get pixel coordinates for the center of the bar (price level) - this is where the center line
    // will be
    ImVec2 center_point = ImPlot::PlotToPixels(0, xs[i]);

    // Calculate the extents of buy and sell volumes in plot coordinates
    // Buy volumes extend to the LEFT (negative x direction) from center
    double plot_scaled_buy_vol = (buy_vols[i] / max_vol) * yesterday_max_volume_;
    // Sell volumes extend to the RIGHT (positive x direction) from center
    double plot_scaled_sell_vol = (sell_vols[i] / max_vol) * yesterday_max_volume_;

    // Convert plot coordinates to pixel coordinates
    ImVec2 buy_end_point = ImPlot::PlotToPixels(-plot_scaled_buy_vol, xs[i]);
    ImVec2 sell_end_point = ImPlot::PlotToPixels(plot_scaled_sell_vol, xs[i]);

    // Calculate bar dimensions
    float bar_top = center_point.y - height / 2;
    float bar_bottom = center_point.y + height / 2;

    // Draw buy volume bar (left side, green) - extends from buy_end_point.x to center_point.x
    if (buy_vols[i] > 0) {
      ImVec2 bar_tl = ImVec2(buy_end_point.x, bar_top);
      ImVec2 bar_br = ImVec2(center_point.x, bar_bottom);

      draw_list->AddRectFilled(bar_tl, bar_br, col_buy);

      // Add a subtle border to make the bar more distinguishable
      draw_list->AddRect(bar_tl, bar_br, IM_COL32(0, 0, 0, static_cast<int>(100 * opacity)), 0.0f,
                         0, 1.0f);
    }

    // Draw sell volume bar (right side, red) - extends from center_point.x to sell_end_point.x
    if (sell_vols[i] > 0) {
      ImVec2 bar_tl = ImVec2(center_point.x, bar_top);
      ImVec2 bar_br = ImVec2(sell_end_point.x, bar_bottom);

      draw_list->AddRectFilled(bar_tl, bar_br, col_sell);

      // Add a subtle border to make the bar more distinguishable
      draw_list->AddRect(bar_tl, bar_br, IM_COL32(0, 0, 0, static_cast<int>(100 * opacity)), 0.0f,
                         0, 1.0f);
    }

    // Draw center vertical line to separate buy and sell volumes
    draw_list->AddLine(ImVec2(center_point.x, bar_top), ImVec2(center_point.x, bar_bottom),
                       col_center, 2.0f);  // Thicker line for better visibility
  }
}

// Helper method to extract date from timestamp (YYYYMMDD format)
time_t VolumeProfilePanel::get_date_from_timestamp(double timestamp) {
  time_t t = static_cast<time_t>(timestamp);
  struct tm* tm_info = gmtime(&t);

  // Create a new time structure with only year, month, and day
  struct tm date_tm = {};
  date_tm.tm_year = tm_info->tm_year;
  date_tm.tm_mon = tm_info->tm_mon;
  date_tm.tm_mday = tm_info->tm_mday;
  date_tm.tm_isdst = -1;  // Let system determine DST

  return mktime(&date_tm);
}

// Method to add a daily profile to the composite collection
void VolumeProfilePanel::add_daily_profile(const std::vector<VolumeLevel>& daily_profile,
                                           time_t date) {
  // Check if we already have a profile for this date
  for (auto& existing_profile : daily_profiles_) {
    if (existing_profile.date == date) {
      // Update existing profile
      existing_profile.daily_profile = daily_profile;

      // Recalculate POC and max volume for this daily profile
      double max_vol = 0.0;
      double poc_price = 0.0;
      double poc_volume = 0.0;

      for (const auto& level : daily_profile) {
        double total = level.buy_volume + level.sell_volume;
        if (total > poc_volume) {
          poc_volume = total;
          poc_price = level.price;
        }
        max_vol = std::max(max_vol, std::max(level.buy_volume, level.sell_volume));
      }

      existing_profile.poc_price = poc_price;
      existing_profile.max_volume = max_vol;

      // Calculate value area for this daily profile
      // This is a simplified calculation - in a real implementation, you might want to reuse the
      // calculate_value_area logic
      double total_volume = 0.0;
      for (const auto& level : daily_profile) {
        total_volume += level.total_volume;
      }

      if (total_volume > 0) {
        double target_volume = 0.70 * total_volume;  // 70% value area
        double current_volume = 0.0;

        // Find POC index
        size_t poc_idx = 0;
        for (size_t i = 0; i < daily_profile.size(); ++i) {
          if (daily_profile[i].price == poc_price) {
            poc_idx = i;
            break;
          }
        }

        // Expand from POC to find value area
        size_t start_idx = poc_idx;
        size_t end_idx = poc_idx;
        current_volume = daily_profile[poc_idx].total_volume;

        while (current_volume < target_volume) {
          bool expand_up = false;
          bool expand_down = false;

          if (start_idx > 0) expand_down = true;
          if (end_idx < daily_profile.size() - 1) expand_up = true;

          if (!expand_up && !expand_down) break;

          if (!expand_up && expand_down) {
            start_idx--;
            current_volume += daily_profile[start_idx].total_volume;
          } else if (expand_up && !expand_down) {
            end_idx++;
            current_volume += daily_profile[end_idx].total_volume;
          } else {
            // Expand both directions alternately
            if ((end_idx - poc_idx) <= (poc_idx - start_idx)) {
              end_idx++;
              current_volume += daily_profile[end_idx].total_volume;
            } else {
              start_idx--;
              current_volume += daily_profile[start_idx].total_volume;
            }
          }

          if (current_volume >= target_volume) break;
        }

        existing_profile.vah_price = daily_profile[end_idx].price;
        existing_profile.val_price = daily_profile[start_idx].price;
      }

      return;  // Found and updated existing profile
    }
  }

  // Add new daily profile
  DailyVolumeProfile new_profile(date);
  new_profile.daily_profile = daily_profile;

  // Calculate POC and max volume for the new profile
  double max_vol = 0.0;
  double poc_price = 0.0;
  double poc_volume = 0.0;

  for (const auto& level : daily_profile) {
    double total = level.buy_volume + level.sell_volume;
    if (total > poc_volume) {
      poc_volume = total;
      poc_price = level.price;
    }
    max_vol = std::max(max_vol, std::max(level.buy_volume, level.sell_volume));
  }

  new_profile.poc_price = poc_price;
  new_profile.max_volume = max_vol;

  // Calculate value area for the new profile
  double total_volume = 0.0;
  for (const auto& level : daily_profile) {
    total_volume += level.total_volume;
  }

  if (total_volume > 0) {
    double target_volume = 0.70 * total_volume;  // 70% value area
    double current_volume = 0.0;

    // Find POC index
    size_t poc_idx = 0;
    for (size_t i = 0; i < daily_profile.size(); ++i) {
      if (daily_profile[i].price == poc_price) {
        poc_idx = i;
        break;
      }
    }

    // Expand from POC to find value area
    size_t start_idx = poc_idx;
    size_t end_idx = poc_idx;
    current_volume = daily_profile[poc_idx].total_volume;

    while (current_volume < target_volume) {
      bool expand_up = false;
      bool expand_down = false;

      if (start_idx > 0) expand_down = true;
      if (end_idx < daily_profile.size() - 1) expand_up = true;

      if (!expand_up && !expand_down) break;

      if (!expand_up && expand_down) {
        start_idx--;
        current_volume += daily_profile[start_idx].total_volume;
      } else if (expand_up && !expand_down) {
        end_idx++;
        current_volume += daily_profile[end_idx].total_volume;
      } else {
        // Expand both directions alternately
        if ((end_idx - poc_idx) <= (poc_idx - start_idx)) {
          end_idx++;
          current_volume += daily_profile[end_idx].total_volume;
        } else {
          start_idx--;
          current_volume += daily_profile[start_idx].total_volume;
        }
      }

      if (current_volume >= target_volume) break;
    }

    new_profile.vah_price = daily_profile[end_idx].price;
    new_profile.val_price = daily_profile[start_idx].price;
  }

  daily_profiles_.push_back(new_profile);
}

// Method to clear all daily profiles
void VolumeProfilePanel::clear_daily_profiles() { daily_profiles_.clear(); }

// Method to build composite profile from multiple days
void VolumeProfilePanel::build_composite_profile() {
  if (daily_profiles_.empty()) {
    // If no daily profiles exist, try to build from recent trades
    if (!processor_ || symbol_id_ == 0) return;

    auto analytics = processor_->getSymbolAnalytics(symbol_id_);
    auto trades = analytics.recent_trades;

    if (trades.empty()) return;

    // Group trades by date
    std::map<time_t, std::vector<RenderEngine::TradeData>> trades_by_date;

    for (const auto& trade : trades) {
      time_t date = get_date_from_timestamp(trade.timestamp);
      trades_by_date[date].push_back(trade);
    }

    // Create daily profiles for each date
    for (const auto& date_trades_pair : trades_by_date) {
      const auto& date_trades = date_trades_pair.second;

      // Find price range for this day
      double min_price = std::numeric_limits<double>::max();
      double max_price = std::numeric_limits<double>::lowest();

      for (const auto& trade : date_trades) {
        min_price = std::min(min_price, trade.price);
        max_price = std::max(max_price, trade.price);
      }

      if (max_price <= min_price) continue;

      // Create volume profile for this day
      std::vector<VolumeLevel> daily_profile;
      const size_t num_levels = NUM_PRICE_LEVELS;
      double range = max_price - min_price;
      double bucket_size = range / num_levels;

      daily_profile.resize(num_levels);

      for (size_t i = 0; i < num_levels; ++i) {
        daily_profile[i].price = min_price + (i + 0.5) * bucket_size;
        daily_profile[i].buy_volume = 0;
        daily_profile[i].sell_volume = 0;
        daily_profile[i].total_volume = 0;
        daily_profile[i].buy_trades = 0;
        daily_profile[i].sell_trades = 0;
        daily_profile[i].total_trades = 0;
      }

      // Aggregate trades into buckets
      for (const auto& trade : date_trades) {
        size_t bucket = static_cast<size_t>((trade.price - min_price) / bucket_size);
        bucket = std::min(bucket, num_levels - 1);

        if (trade.is_buy) {
          daily_profile[bucket].buy_volume += trade.size;
          daily_profile[bucket].buy_trades++;
        } else {
          daily_profile[bucket].sell_volume += trade.size;
          daily_profile[bucket].sell_trades++;
        }
        daily_profile[bucket].total_volume += trade.size;
        daily_profile[bucket].total_trades++;
      }

      // Add this daily profile to our collection
      add_daily_profile(daily_profile, date_trades_pair.first);
    }
  }

  // Limit to the specified number of days
  if (composite_days_count_ > 0 &&
      daily_profiles_.size() > static_cast<size_t>(composite_days_count_)) {
    // Keep only the most recent composite_days_count_ profiles
    std::sort(daily_profiles_.begin(), daily_profiles_.end(),
              [](const DailyVolumeProfile& a, const DailyVolumeProfile& b) {
                return a.date > b.date;  // Sort in descending order (most recent first)
              });

    daily_profiles_.erase(daily_profiles_.begin() + composite_days_count_, daily_profiles_.end());
  }

  // Now aggregate all daily profiles into a composite profile
  if (daily_profiles_.empty()) return;

  // Find the overall price range across all daily profiles
  double min_price = std::numeric_limits<double>::max();
  double max_price = std::numeric_limits<double>::lowest();

  for (const auto& daily_profile : daily_profiles_) {
    for (const auto& level : daily_profile.daily_profile) {
      min_price = std::min(min_price, level.price);
      max_price = std::max(max_price, level.price);
    }
  }

  if (max_price <= min_price) return;

  // Create composite profile with consistent price levels
  const size_t num_levels = NUM_PRICE_LEVELS;
  double range = max_price - min_price;
  double bucket_size = range / num_levels;

  composite_volume_profile_.clear();
  composite_volume_profile_.resize(num_levels);

  for (size_t i = 0; i < num_levels; ++i) {
    composite_volume_profile_[i].price = min_price + (i + 0.5) * bucket_size;
    composite_volume_profile_[i].buy_volume = 0;
    composite_volume_profile_[i].sell_volume = 0;
    composite_volume_profile_[i].total_volume = 0;
    composite_volume_profile_[i].buy_trades = 0;
    composite_volume_profile_[i].sell_trades = 0;
    composite_volume_profile_[i].total_trades = 0;
  }

  // Aggregate volume from all daily profiles
  for (const auto& daily_profile : daily_profiles_) {
    for (const auto& daily_level : daily_profile.daily_profile) {
      // Find the corresponding level in the composite profile
      size_t composite_idx = static_cast<size_t>((daily_level.price - min_price) / bucket_size);
      if (composite_idx < composite_volume_profile_.size()) {
        composite_volume_profile_[composite_idx].buy_volume += daily_level.buy_volume;
        composite_volume_profile_[composite_idx].sell_volume += daily_level.sell_volume;
        composite_volume_profile_[composite_idx].total_volume += daily_level.total_volume;
        composite_volume_profile_[composite_idx].buy_trades += daily_level.buy_trades;
        composite_volume_profile_[composite_idx].sell_trades += daily_level.sell_trades;
        composite_volume_profile_[composite_idx].total_trades += daily_level.total_trades;
      }
    }
  }

  // Calculate composite POC and max volume
  composite_max_volume_ = 0.0;
  composite_poc_price_ = composite_volume_profile_[0].price;
  double poc_volume = 0.0;

  for (const auto& level : composite_volume_profile_) {
    double total = level.buy_volume + level.sell_volume;
    composite_max_volume_ =
        std::max(composite_max_volume_, std::max(level.buy_volume, level.sell_volume));
    if (total > poc_volume) {
      poc_volume = total;
      composite_poc_price_ = level.price;
    }
  }

  // Calculate composite value area
  calculate_composite_value_area();
}

// Method to calculate value area for composite profile
void VolumeProfilePanel::calculate_composite_value_area() {
  if (composite_volume_profile_.empty()) {
    composite_vah_price_ = 0.0;
    composite_val_price_ = 0.0;
    return;
  }

  // Calculate total volume in the composite profile
  double total_volume = 0.0;
  for (const auto& level : composite_volume_profile_) {
    total_volume += level.total_volume;
  }

  if (total_volume <= 0) {
    composite_vah_price_ = 0.0;
    composite_val_price_ = 0.0;
    return;
  }

  // Target volume for value area (based on profile_settings_.vaPercent % of total volume)
  double target_volume = (static_cast<double>(profile_settings_.vaPercent) / 100.0) * total_volume;

  // Find the POC index (Point of Control - highest volume)
  size_t poc_index = 0;
  double max_total_volume = 0.0;
  for (size_t i = 0; i < composite_volume_profile_.size(); ++i) {
    double total =
        composite_volume_profile_[i].buy_volume + composite_volume_profile_[i].sell_volume;
    if (total > max_total_volume) {
      max_total_volume = total;
      poc_index = i;
    }
  }

  // Expand from POC outward to capture the required volume, centered around POC
  size_t start_idx = poc_index;
  size_t end_idx = poc_index;
  double current_volume = composite_volume_profile_[poc_index].total_volume;

  // Expand upward (higher prices) and downward (lower prices) alternately
  // until we reach the target volume, keeping the expansion balanced around POC
  while (current_volume < target_volume) {
    // Decide whether to expand up or down
    bool expand_up = false;
    bool expand_down = false;

    // Check if we can expand in each direction
    if (start_idx > 0) expand_down = true;
    if (end_idx < composite_volume_profile_.size() - 1) expand_up = true;

    // If we can't expand in either direction, break
    if (!expand_up && !expand_down) break;

    // If we can only expand in one direction, do that
    if (!expand_up && expand_down) {
      start_idx--;
      current_volume += composite_volume_profile_[start_idx].total_volume;
    } else if (expand_up && !expand_down) {
      end_idx++;
      current_volume += composite_volume_profile_[end_idx].total_volume;
    } else {
      // We can expand in both directions - try to keep it centered around POC
      // Calculate potential volumes for each direction
      double vol_up = composite_volume_profile_[end_idx + 1].total_volume;
      double vol_down = composite_volume_profile_[start_idx - 1].total_volume;

      // To center around POC, we should try to balance the expansion
      // If both sides have similar volume, expand the side that currently has a smaller range
      size_t current_upper_range = end_idx - poc_index;
      size_t current_lower_range = poc_index - start_idx;

      if (current_lower_range < current_upper_range) {
        // Current lower range is smaller, expand downward to balance
        start_idx--;
        current_volume += composite_volume_profile_[start_idx].total_volume;
      } else if (current_upper_range < current_lower_range) {
        // Current upper range is smaller, expand upward to balance
        end_idx++;
        current_volume += composite_volume_profile_[end_idx].total_volume;
      } else {
        // Ranges are equal, expand toward the side with more volume to capture more volume
        // efficiently
        if (vol_up >= vol_down) {
          end_idx++;
          current_volume += composite_volume_profile_[end_idx].total_volume;
        } else {
          start_idx--;
          current_volume += composite_volume_profile_[start_idx].total_volume;
        }
      }
    }

    // If we've captured enough volume, break
    if (current_volume >= target_volume) break;
  }

  // Set the VAH and VAL prices
  composite_vah_price_ = composite_volume_profile_[end_idx].price;
  composite_val_price_ = composite_volume_profile_[start_idx].price;
}

// Method to update composite profile based on current settings
void VolumeProfilePanel::update_composite_profile() {
  if (profile_mode_ == ProfileMode::Composite) {
    build_composite_profile();
  }
}

// Method to highlight zones where profiles diverge significantly
void VolumeProfilePanel::highlight_profile_divergence(ImDrawList* draw_list) {
  // Only highlight divergence if we have both today's and yesterday's profiles
  if (volume_profile_.empty() || yesterday_volume_profile_.empty()) {
    return;
  }

  // Define threshold for significant divergence (e.g., 50% difference in volume)
  const double divergence_threshold = 0.50;  // 50% difference threshold

  // Compare corresponding price levels between today's and yesterday's profiles
  // We'll iterate through today's profile and find corresponding levels in yesterday's profile
  for (size_t i = 0; i < volume_profile_.size(); ++i) {
    const auto& current_level = volume_profile_[i];

    // Find the closest price level in yesterday's profile
    size_t closest_idx = 0;
    double min_diff = std::abs(current_level.price - yesterday_volume_profile_[0].price);

    for (size_t j = 1; j < yesterday_volume_profile_.size(); ++j) {
      double diff = std::abs(current_level.price - yesterday_volume_profile_[j].price);
      if (diff < min_diff) {
        min_diff = diff;
        closest_idx = j;
      }
    }

    const auto& yesterday_level = yesterday_volume_profile_[closest_idx];

    // Calculate total volumes for comparison
    double current_total_vol = current_level.total_volume;
    double yesterday_total_vol = yesterday_level.total_volume;

    // Calculate percentage difference
    double avg_volume = (current_total_vol + yesterday_total_vol) / 2.0;
    if (avg_volume > 0) {
      double volume_diff = std::abs(current_total_vol - yesterday_total_vol) / avg_volume;

      // If difference exceeds threshold, highlight this zone
      if (volume_diff > divergence_threshold) {
        // Convert price to pixel coordinates for drawing
        ImVec2 top_left = ImPlot::PlotToPixels(-max_volume_, current_level.price);
        ImVec2 bottom_right =
            ImPlot::PlotToPixels(max_volume_, current_level.price - price_bucket_size_);

        // Draw a semi-transparent rectangle to highlight the divergence
        // Use orange color to indicate significant divergence
        draw_list->AddRectFilled(top_left, bottom_right,
                                 IM_COL32(255, 165, 0, 80));  // Orange with low opacity

        // Add a border to make the highlight more visible
        draw_list->AddRect(top_left, bottom_right, IM_COL32(255, 165, 0, 150), 0.0f, 0,
                           2.0f);  // Orange border
      }
    }
  }
}

}  // namespace BTQuant
