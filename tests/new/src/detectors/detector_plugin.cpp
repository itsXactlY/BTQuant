#include "detectors/detector_plugin.hpp"
#include "hotspine_extended_reader.hpp"
#include <dlfcn.h>
#include <iostream>

namespace BTQuant::Detectors {

std::vector<DetectionSignal> IDetector::detect_all(const std::string &symbol) {
  if (auto signal = detect(symbol)) {
    std::vector<DetectionSignal> result;
    result.push_back(*signal);
    return result;
  }
  return {};
}

void IDetector::update_orderbook(const std::string &exchange,
                                 const std::string &symbol) {
  // Default: no action
}

void IDetector::update_trade(const TradeData &trade) {
  // Default: no action
}

std::vector<std::string> IDetector::get_dependencies() const { return {}; }

void IDetector::set_dependency(const std::string &name,
                               std::shared_ptr<void> dependency) {
  dependencies_[name] = dependency;
}

DetectorRegistry &DetectorRegistry::instance() {
  static DetectorRegistry instance;
  return instance;
}

bool DetectorRegistry::register_detector(const std::string &name,
                                         DetectorFactory factory) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (plugins_.find(name) != plugins_.end()) {
    std::cerr << "Warning: Detector '" << name << "' already registered"
              << std::endl;
    return false;
  }

  PluginInfo info;
  info.name = name;
  info.factory = factory;
  plugins_[name] = info;

  return true;
}

bool DetectorRegistry::register_detector(const PluginInfo &plugin) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (plugins_.find(plugin.name) != plugins_.end()) {
    std::cerr << "Warning: Detector '" << plugin.name << "' already registered"
              << std::endl;
    return false;
  }

  plugins_[plugin.name] = plugin;
  return true;
}

std::shared_ptr<IDetector>
DetectorRegistry::load_plugin(const std::string &path) {
  void *handle = dlopen(path.c_str(), RTLD_NOW);
  if (!handle) {
    std::cerr << "Error loading plugin: " << dlerror() << std::endl;
    return nullptr;
  }

  auto create_func = reinterpret_cast<std::unique_ptr<IDetector> (*)()>(
      dlsym(handle, "create_detector"));

  if (!create_func) {
    std::cerr << "Error: create_detector function not found in plugin"
              << std::endl;
    dlclose(handle);
    return nullptr;
  }

  auto detector = create_func();
  if (!detector) {
    std::cerr << "Error: Failed to create detector instance" << std::endl;
    dlclose(handle);
    return nullptr;
  }

  auto info_func =
      reinterpret_cast<PluginInfo (*)()>(dlsym(handle, "get_plugin_info"));

  if (info_func) {
    PluginInfo info = info_func();
    std::lock_guard<std::mutex> lock(mutex_);
    info.path = path;
    plugins_[info.name] = info;
  }

  return detector;
}

std::shared_ptr<IDetector>
DetectorRegistry::get_detector(const std::string &name) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = instances_.find(name);
  if (it != instances_.end()) {
    return it->second;
  }

  auto plugin_it = plugins_.find(name);
  if (plugin_it != plugins_.end() && plugin_it->second.factory) {
    std::shared_ptr<IDetector> detector(plugin_it->second.factory().release());
    instances_.emplace(name, detector);
    return detector;
  }

  return nullptr;
}

std::vector<std::string> DetectorRegistry::get_available_detectors() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<std::string> names;
  for (const auto &[name, _] : plugins_) {
    names.push_back(name);
  }
  return names;
}

std::vector<std::shared_ptr<IDetector>> DetectorRegistry::get_all_detectors() {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<std::shared_ptr<IDetector>> detectors;
  for (auto &[name, plugin] : plugins_) {
    if (instances_.find(name) == instances_.end() && plugin.factory) {
      std::shared_ptr<IDetector> detector(plugin.factory().release());
      instances_.emplace(name, detector);
    }
    if (instances_.find(name) != instances_.end()) {
      detectors.push_back(instances_[name]);
    }
  }
  return detectors;
}

bool DetectorRegistry::has_detector(const std::string &name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return plugins_.find(name) != plugins_.end();
}

std::shared_ptr<IDetector> DetectorRegistry::create_detector(
    const std::string &name,
    const std::unordered_map<std::string, Config::ConfigValue> &config) {

  auto detector = get_detector(name);
  if (detector) {
    detector->configure(config);
    detector->initialize();
  }
  return detector;
}

void DetectorRegistry::register_builtin_detectors() {
  // Built-in detectors are linked statically
}

void DetectorRegistry::clear() {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto &[name, instance] : instances_) {
    if (instance) {
      instance->shutdown();
    }
  }

  plugins_.clear();
  instances_.clear();
}

} // namespace BTQuant::Detectors
