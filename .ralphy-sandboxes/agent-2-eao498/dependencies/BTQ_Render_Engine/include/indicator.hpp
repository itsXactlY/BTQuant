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

  float get_value() const override {
    return history_.empty() ? 0.0f : sum_ / history_.size();
  }
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
    if (!initialized_ || avg_loss_ == 0.0f)
      return 100.0f;
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

  float get_value() const override {
    return fast_ema_.get_value() - slow_ema_.get_value();
  }
  float get_signal() const { return signal_ema_.get_value(); }
  float get_histogram() const { return get_value() - get_signal(); }

  bool is_ready() const override {
    return fast_ema_.is_ready() && slow_ema_.is_ready() &&
           signal_ema_.is_ready();
  }
  void reset() override {
    fast_ema_.reset();
    slow_ema_.reset();
    signal_ema_.reset();
  }

private:
  EMAIndicator fast_ema_, slow_ema_, signal_ema_;
};

} // namespace BTQuant
