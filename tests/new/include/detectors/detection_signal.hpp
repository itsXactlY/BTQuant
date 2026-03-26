#pragma once

#include <cstdint>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

namespace BTQuant {

enum class DetectionType {
  SPREAD_ARBITRAGE,
  SPOOFING,
  WHALE_FRONTRUN,
  STOP_HUNT,
  LIQUIDITY_IMBALANCE,
  UNKNOWN
};

enum class Severity { LOW, MEDIUM, HIGH, CRITICAL };

struct DetectionSignal {
  std::string id;
  DetectionType type;
  Severity severity;
  std::string symbol;
  std::string exchange;
  std::string description;
  double confidence; // 0.0 to 1.0
  uint64_t timestamp;
  std::unordered_map<std::string, double> metadata;

  std::string to_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "DetectionSignal("
        << "id=" << id << ", type=";

    switch (type) {
    case DetectionType::SPREAD_ARBITRAGE:
      oss << "SPREAD_ARBITRAGE";
      break;
    case DetectionType::SPOOFING:
      oss << "SPOOFING";
      break;
    case DetectionType::WHALE_FRONTRUN:
      oss << "WHALE_FRONTRUN";
      break;
    case DetectionType::STOP_HUNT:
      oss << "STOP_HUNT";
      break;
    case DetectionType::LIQUIDITY_IMBALANCE:
      oss << "LIQUIDITY_IMBALANCE";
      break;
    case DetectionType::UNKNOWN:
      oss << "UNKNOWN";
      break;
    }

    oss << ", severity=";
    switch (severity) {
    case Severity::LOW:
      oss << "LOW";
      break;
    case Severity::MEDIUM:
      oss << "MEDIUM";
      break;
    case Severity::HIGH:
      oss << "HIGH";
      break;
    case Severity::CRITICAL:
      oss << "CRITICAL";
      break;
    }

    oss << ", symbol=" << symbol << ", exchange=" << exchange
        << ", description=" << description << ", confidence=" << confidence
        << ", timestamp=" << timestamp << ")";

    return oss.str();
  }
};

} // namespace BTQuant