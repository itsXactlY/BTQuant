#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Includes instead of forward declarations
#include "config/config_loader.hpp"
#include "detectors/detection_signal.hpp"
#include "hotspine_extended_reader.hpp"

namespace BTQuant::Detectors {

// Detector interface
class IDetector {
public:
  virtual ~IDetector() = default;

  // Detector metadata
  virtual std::string get_name() const = 0;
  virtual std::string get_version() const = 0;
  virtual std::string get_description() const = 0;

  // Detection methods
  virtual std::optional<BTQuant::DetectionSignal>
  detect(const std::string &symbol) = 0;
  virtual std::vector<BTQuant::DetectionSignal>
  detect_all(const std::string &symbol);
  virtual void update_orderbook(const std::string &exchange,
                                const std::string &symbol);
  virtual void update_trade(const BTQuant::TradeData &trade);

  // Configuration
  virtual void configure(
      const std::unordered_map<std::string, Config::ConfigValue> &config) = 0;
  virtual std::unordered_map<std::string, Config::ConfigValue>
  get_configuration() const = 0;

  // Statistics
  virtual uint64_t get_detections() const = 0;
  virtual void reset_statistics() = 0;

  // Lifecycle
  virtual bool initialize() = 0;
  virtual void shutdown() = 0;

  // Dependencies
  virtual std::vector<std::string> get_dependencies() const;
  virtual void set_dependency(const std::string &name,
                              std::shared_ptr<void> dependency);

protected:
  std::unordered_map<std::string, std::shared_ptr<void>> dependencies_;
};

// Factory function type
using DetectorFactory = std::function<std::unique_ptr<IDetector>()>;

// Plugin information
struct PluginInfo {
  std::string name;
  std::string version;
  std::string description;
  std::string path;
  std::vector<std::string> dependencies;
  DetectorFactory factory;
};

// Detector registry
class DetectorRegistry {
public:
  static DetectorRegistry &instance();

  // Register a detector
  bool register_detector(const std::string &name, DetectorFactory factory);
  bool register_detector(const PluginInfo &plugin);

  // Load detector from shared library
  std::shared_ptr<IDetector> load_plugin(const std::string &path);

  // Get detector by name
  std::shared_ptr<IDetector> get_detector(const std::string &name);

  // Get all registered detectors
  std::vector<std::string> get_available_detectors() const;
  std::vector<std::shared_ptr<IDetector>> get_all_detectors();

  // Check if detector is registered
  bool has_detector(const std::string &name) const;

  // Create detector instance with configuration
  std::shared_ptr<IDetector> create_detector(
      const std::string &name,
      const std::unordered_map<std::string, Config::ConfigValue> &config = {});

  // Auto-register all built-in detectors
  void register_builtin_detectors();

  // Clear all registered detectors
  void clear();

private:
  DetectorRegistry() = default;
  ~DetectorRegistry() = default;
  DetectorRegistry(const DetectorRegistry &) = delete;
  DetectorRegistry &operator=(const DetectorRegistry &) = delete;

  std::unordered_map<std::string, PluginInfo> plugins_;
  std::unordered_map<std::string, std::shared_ptr<IDetector>> instances_;
  mutable std::mutex mutex_;
};

} // namespace BTQuant::Detectors

// Plugin export macro for C++ detectors
#define BTQUANT_DETECTOR_PLUGIN(DetectorClass)                                 \
  extern "C" {                                                                 \
  std::unique_ptr<BTQuant::Detectors::IDetector> create_detector() {           \
    return std::make_unique<DetectorClass>();                                  \
  }                                                                            \
  BTQuant::Detectors::PluginInfo get_plugin_info() {                           \
    return {DetectorClass::PLUGIN_NAME,                                        \
            DetectorClass::PLUGIN_VERSION,                                     \
            DetectorClass::PLUGIN_DESCRIPTION,                                 \
            "",                                                                \
            std::vector<std::string>(                                          \
                std::begin(DetectorClass::PLUGIN_DEPENDENCIES),                \
                std::end(DetectorClass::PLUGIN_DEPENDENCIES)),                 \
            create_detector};                                                  \
  }                                                                            \
  }
