#pragma once

#include <cmath>
#include <deque>
#include <numeric>
#include <vector>

namespace BTQuant {

class TechnicalIndicator {
 public:
  virtual ~TechnicalIndicator() = default;
  virtual void update(float value) = 0;
  virtual float get_value() const = 0;
  virtual bool is_ready() const = 0;
  virtual void reset() = 0;
};

class EMAIndicator : public TechnicalIndicator {
 public:
  EMAIndicator(int period) : period_(period), alpha_(2.0f / (period + 1.0f)) {}

  void update(float value) override {
    if (!initialized_) {
      current_ema_ = value;
      initialized_ = true;
    } else {
      current_ema_ = (value - current_ema_) * alpha_ + current_ema_;
    }
    count_++;
  }

  float get_value() const override { return current_ema_; }
  bool is_ready() const override { return count_ >= period_; }
  void reset() override {
    current_ema_ = 0.0f;
    initialized_ = false;
    count_ = 0;
  }

 private:
  int period_;
  float alpha_;
  float current_ema_ = 0.0f;
  bool initialized_ = false;
  int count_ = 0;
};

class SMAIndicator : public TechnicalIndicator {
 public:
  SMAIndicator(int period) : period_(period) {}

  void update(float value) override {
    history_.push_back(value);
    sum_ += value;
    if (history_.size() > period_) {
      sum_ -= history_.front();
      history_.pop_front();
    }
  }

  float get_value() const override { return history_.empty() ? 0.0f : sum_ / history_.size(); }
  bool is_ready() const override { return history_.size() >= period_; }
  void reset() override {
    history_.clear();
    sum_ = 0.0f;
  }

 private:
  size_t period_;
  std::deque<float> history_;
  float sum_ = 0.0f;
};

class RSIIndicator : public TechnicalIndicator {
 public:
  RSIIndicator(int period) : period_(period), alpha_(1.0f / period) {}

  void update(float value) override {
    if (last_value_set_) {
      float diff = value - last_value_;
      float gain = std::max(0.0f, diff);
      float loss = std::max(0.0f, -diff);

      if (!initialized_) {
        avg_gain_ += gain;
        avg_loss_ += loss;
        count_++;
        if (count_ >= period_) {
          avg_gain_ /= (float)period_;
          avg_loss_ /= (float)period_;
          initialized_ = true;
        }
      } else {
        avg_gain_ = (gain - avg_gain_) * alpha_ + avg_gain_;
        avg_loss_ = (loss - avg_loss_) * alpha_ + avg_loss_;
      }
    }
    last_value_ = value;
    last_value_set_ = true;
  }

  float get_value() const override {
    if (!initialized_ || avg_loss_ == 0.0f) return 100.0f;
    float rs = avg_gain_ / avg_loss_;
    return 100.0f - (100.0f / (1.0f + rs));
  }
  bool is_ready() const override { return initialized_; }
  void reset() override {
    avg_gain_ = 0.0f;
    avg_loss_ = 0.0f;
    last_value_ = 0.0f;
    last_value_set_ = false;
    initialized_ = false;
    count_ = 0;
  }

 private:
  size_t period_;
  float alpha_;
  float avg_gain_ = 0.0f;
  float avg_loss_ = 0.0f;
  float last_value_ = 0.0f;
  bool last_value_set_ = false;
  bool initialized_ = false;
  size_t count_ = 0;
};

class MACDIndicator : public TechnicalIndicator {
 public:
  MACDIndicator(int fast_p = 12, int slow_p = 26, int signal_p = 9)
      : fast_ema_(fast_p), slow_ema_(slow_p), signal_ema_(signal_p) {}

  void update(float value) override {
    fast_ema_.update(value);
    slow_ema_.update(value);
    if (fast_ema_.is_ready() && slow_ema_.is_ready()) {
      float macd_line = fast_ema_.get_value() - slow_ema_.get_value();
      signal_ema_.update(macd_line);
    }
  }

  float get_value() const override { return fast_ema_.get_value() - slow_ema_.get_value(); }
  float get_signal() const { return signal_ema_.get_value(); }
  float get_histogram() const { return get_value() - get_signal(); }

  bool is_ready() const override {
    return fast_ema_.is_ready() && slow_ema_.is_ready() && signal_ema_.is_ready();
  }
  void reset() override {
    fast_ema_.reset();
    slow_ema_.reset();
    signal_ema_.reset();
  }

 private:
  EMAIndicator fast_ema_, slow_ema_, signal_ema_;
};

class BollingerBandIndicator : public TechnicalIndicator {
 public:
  BollingerBandIndicator(int period, double std_dev = 2.0)
      : period_(period), std_dev_(std_dev), sma_indicator_(period) {}

  void update(float value) override {
    sma_indicator_.update(value);
    prices_.push_back(value);
    if (prices_.size() > static_cast<size_t>(period_)) {
      prices_.pop_front();
    }
  }

  float get_value() const override {
    if (!sma_indicator_.is_ready()) return 0.0f;
    return sma_indicator_.get_value(); // Return middle band
  }

  float get_upper_band() const {
    if (!is_ready()) return 0.0f;
    float sma_val = sma_indicator_.get_value();
    float variance = calculate_variance();
    float std_dev_val = std::sqrt(variance);
    return sma_val + (static_cast<float>(std_dev_) * std_dev_val);
  }

  float get_lower_band() const {
    if (!is_ready()) return 0.0f;
    float sma_val = sma_indicator_.get_value();
    float variance = calculate_variance();
    float std_dev_val = std::sqrt(variance);
    return sma_val - (static_cast<float>(std_dev_) * std_dev_val);
  }

  bool is_ready() const override {
    return sma_indicator_.is_ready() && prices_.size() >= static_cast<size_t>(period_);
  }

  void reset() override {
    sma_indicator_.reset();
    prices_.clear();
  }

 private:
  float calculate_variance() const {
    if (prices_.empty()) return 0.0f;

    float mean = sma_indicator_.get_value();
    float sum_sq_diff = 0.0f;

    for (float price : prices_) {
      float diff = price - mean;
      sum_sq_diff += diff * diff;
    }

    return sum_sq_diff / static_cast<float>(prices_.size());
  }

 private:
  int period_;
  double std_dev_;
  SMAIndicator sma_indicator_;
  std::deque<float> prices_;
};

class StochasticIndicator : public TechnicalIndicator {
 public:
  StochasticIndicator(int k_period = 14, int d_period = 3, int slowing_period = 3)
      : k_period_(k_period), d_period_(d_period), slowing_period_(slowing_period) {}

  void update(float value) override {
    prices_.push_back(value);
    if (prices_.size() > static_cast<size_t>(k_period_)) {
      prices_.pop_front();
    }

    if (prices_.size() == static_cast<size_t>(k_period_)) {
      float highest_high = *std::max_element(prices_.begin(), prices_.end());
      float lowest_low = *std::min_element(prices_.begin(), prices_.end());

      if (highest_high != lowest_low) {
        float k_val = ((value - lowest_low) / (highest_high - lowest_low)) * 100.0f;
        k_values_.push_back(k_val);

        if (k_values_.size() > static_cast<size_t>(d_period_)) {
          k_values_.pop_front();
        }
      }
    }
  }

  float get_value() const override {
    if (k_values_.empty()) return 0.0f;
    // Return the %D value (moving average of %K)
    float sum = std::accumulate(k_values_.begin(), k_values_.end(), 0.0f);
    return sum / static_cast<float>(k_values_.size());
  }

  float get_k_value() const {
    if (k_values_.empty()) return 0.0f;
    return k_values_.back(); // Return most recent %K value
  }

  bool is_ready() const override {
    return !k_values_.empty();
  }

  void reset() override {
    prices_.clear();
    k_values_.clear();
  }

 private:
  int k_period_;
  int d_period_;
  int slowing_period_;
  std::deque<float> prices_;
  std::deque<float> k_values_;
};

class ATRIndicator : public TechnicalIndicator {
 public:
  ATRIndicator(int period = 14) : period_(period) {}

  void update(float value) override {
    current_price_ = value;

    if (last_price_ > 0.0f) { // Skip first update
      float tr = calculate_true_range(value);
      tr_values_.push_back(tr);

      if (tr_values_.size() > static_cast<size_t>(period_)) {
        tr_values_.pop_front();
      }
    }

    last_price_ = value;
  }

  float get_value() const override {
    if (tr_values_.empty()) return 0.0f;
    float sum = std::accumulate(tr_values_.begin(), tr_values_.end(), 0.0f);
    return sum / static_cast<float>(tr_values_.size());
  }

  bool is_ready() const override {
    return tr_values_.size() >= static_cast<size_t>(period_);
  }

  void reset() override {
    tr_values_.clear();
    last_price_ = 0.0f;
    current_price_ = 0.0f;
  }

 private:
  float calculate_true_range(float current_price) const {
    if (last_price_ <= 0.0f) return 0.0f; // First price

    float tr1 = std::abs(current_price - last_price_);
    float tr2 = high_price_ > 0.0f ? std::abs(current_price - high_price_) : 0.0f;
    float tr3 = low_price_ > 0.0f ? std::abs(current_price - low_price_) : 0.0f;

    // Update high/low for next calculation
    high_price_ = std::max(current_price, last_price_);
    low_price_ = std::min(current_price, last_price_);

    return std::max({tr1, tr2, tr3});
  }

 private:
  int period_;
  std::deque<float> tr_values_;
  float last_price_ = 0.0f;
  float current_price_ = 0.0f;
  mutable float high_price_ = 0.0f;
  mutable float low_price_ = 0.0f;
};

// Parabolic SAR Indicator
class PSARIndicator : public TechnicalIndicator {
 public:
  PSARIndicator(float acceleration_step = 0.02f, float acceleration_max = 0.2f)
      : acceleration_step_(acceleration_step), acceleration_max_(acceleration_max) {}

  void update(float value) override {
    if (values_.size() < 2) {
      values_.push_back(value);
      current_sar_ = value;
      return;
    }

    values_.push_back(value);
    if (values_.size() > 3) {
      values_.pop_front();
    }

    // Simple implementation of Parabolic SAR
    float ep = (values_.size() >= 2) ? *std::max_element(values_.begin(), values_.end()) : value;
    float sar = current_sar_ + acceleration_step_ * (ep - current_sar_);

    current_sar_ = sar;
    initialized_ = true;
  }

  float get_value() const override { return current_sar_; }
  bool is_ready() const override { return initialized_; }
  void reset() override {
    values_.clear();
    current_sar_ = 0.0f;
    initialized_ = false;
  }

 private:
  float acceleration_step_;
  float acceleration_max_;
  float current_sar_ = 0.0f;
  bool initialized_ = false;
  std::deque<float> values_;
};

// Commodity Channel Index Indicator
class CCIIndicator : public TechnicalIndicator {
 public:
  CCIIndicator(int period = 20) : period_(period) {}

  void update(float value) override {
    values_.push_back(value);
    if (values_.size() > static_cast<size_t>(period_)) {
      values_.pop_front();
    }
  }

  float get_value() const override {
    if (values_.size() < static_cast<size_t>(period_)) return 0.0f;

    // Calculate typical price (we'll use the value as typical price)
    float sma = std::accumulate(values_.begin(), values_.end(), 0.0f) / values_.size();

    // Calculate mean deviation
    float mean_dev = 0.0f;
    for (float val : values_) {
      mean_dev += std::abs(val - sma);
    }
    mean_dev /= values_.size();

    // Calculate CCI
    if (mean_dev == 0.0f) return 0.0f;

    float cci = (values_.back() - sma) / (0.015f * mean_dev);
    return cci;
  }

  bool is_ready() const override { return values_.size() >= static_cast<size_t>(period_); }
  void reset() override {
    values_.clear();
  }

 private:
  int period_;
  std::deque<float> values_;
};

// Williams %R Indicator
class WilliamsRIndicator : public TechnicalIndicator {
 public:
  WilliamsRIndicator(int period = 14) : period_(period) {}

  void update(float value) override {
    values_.push_back(value);
    if (values_.size() > static_cast<size_t>(period_)) {
      values_.pop_front();
    }
  }

  float get_value() const override {
    if (values_.size() < static_cast<size_t>(period_)) return 0.0f;

    if (values_.size() < 2) return -50.0f; // Neutral value

    auto minmax = std::minmax_element(values_.begin(), values_.end());
    float highest_high = *minmax.second;
    float lowest_low = *minmax.first;

    if (highest_high == lowest_low) return -50.0f; // Avoid division by zero

    float wr = -100.0f * (highest_high - values_.back()) / (highest_high - lowest_low);
    return wr;
  }

  bool is_ready() const override { return values_.size() >= static_cast<size_t>(period_); }
  void reset() override {
    values_.clear();
  }

 private:
  int period_;
  std::deque<float> values_;
};

}  // namespace BTQuant
