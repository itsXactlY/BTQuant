#include "config/config_loader.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
extern char **environ;

namespace BTQuant::Config {

// Environment variable prefix
static const char *ENV_PREFIX = "BTQ_";

ConfigLoader &ConfigLoader::instance() {
  static ConfigLoader instance;
  return instance;
}

bool ConfigLoader::initialize(const std::string &app_path) {
  if (initialized_) {
    return true;
  }

  app_path_ = app_path;

  // Set up configuration search paths
  const char *xdg_config = std::getenv("XDG_CONFIG_HOME");
  const char *home = std::getenv("HOME");

  config_search_paths_.push_back("./config");
  config_search_paths_.push_back("../config");

  if (xdg_config) {
    config_search_paths_.push_back(std::string(xdg_config) + "/btquant");
  }

  if (home) {
    config_search_paths_.push_back(std::string(home) + "/.config/btquant");
  }

  config_search_paths_.push_back("/etc/btquant");

  // Try to find main config file
  for (const auto &path : config_search_paths_) {
    std::string config_path = path + "/btquant.yaml";
    if (std::filesystem::exists(config_path)) {
      main_config_path_ = config_path;
      break;
    }
  }

  // Try to find symbol mapping file
  for (const auto &path : config_search_paths_) {
    std::string mapping_path = path + "/symbol_mappings.yaml";
    if (std::filesystem::exists(mapping_path)) {
      symbol_mapping_path_ = mapping_path;
      break;
    }
  }

  initialized_ = true;
  return true;
}

bool ConfigLoader::load() {
  if (!initialized_) {
    initialize();
  }

  // Load environment variables first (highest priority)
  load_environment_variables();

  // Load main configuration file
  if (!main_config_path_.empty()) {
    load_config_file(main_config_path_);
  }

  // Load symbol mappings
  if (!symbol_mapping_path_.empty()) {
    load_config_file(symbol_mapping_path_);
  }

  // Check if config is empty by getting a read lock and checking size
  auto config_guard = config_.read_lock();
  return !(*config_guard).empty();
}

std::optional<ConfigValue> ConfigLoader::get(const std::string &section,
                                             const std::string &key) const {
  auto config_guard = config_.read_lock();
  auto section_it = (*config_guard).find(section);
  if (section_it == (*config_guard).end()) {
    return std::nullopt;
  }

  auto key_it = section_it->second.find(key);
  if (key_it == section_it->second.end()) {
    return std::nullopt;
  }

  return key_it->second;
}

bool ConfigLoader::set(const std::string &section, const std::string &key,
                       const ConfigValue &value) {
  ConfigValue old_value;
  bool had_old = false;

  // First, check if there's an old value
  {
    auto config_guard = config_.read_lock();
    auto section_it = (*config_guard).find(section);
    if (section_it != (*config_guard).end()) {
      auto key_it = section_it->second.find(key);
      if (key_it != section_it->second.end()) {
        old_value = key_it->second;
        had_old = true;
      }
    }
  }

  // Update the config using RCU
  config_.update([section, key, value](ConfigMap &config_map) {
    config_map[section][key] = value;
  });

  // Update the source map using RCU
  source_map_.update([section, key](std::unordered_map<std::string, ConfigSource> &source_map_ref) {
    source_map_ref[section + "." + key] = ConfigSource::CLI_ARGUMENT;
  });

  // Notify callbacks
  for (const auto &callback : change_callbacks_) {
    callback(section, key, had_old ? old_value : ConfigValue{}, value);
  }

  return true;
}

std::vector<std::string> ConfigLoader::get_sections() const {
  std::vector<std::string> sections;
  auto config_guard = config_.read_lock();
  for (const auto &[section, _] : *config_guard) {
    sections.push_back(section);
  }
  return sections;
}

void ConfigLoader::register_change_callback(ConfigChangeCallback callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  change_callbacks_.push_back(callback);
}

bool ConfigLoader::reload() {
  // Clear the config using RCU
  config_.update([](ConfigMap &config_map) {
    config_map.clear();
  });

  // Clear the source map using RCU
  source_map_.update([](std::unordered_map<std::string, ConfigSource> &source_map_ref) {
    source_map_ref.clear();
  });

  return load();
}

std::optional<ConfigSource>
ConfigLoader::get_source(const std::string &section,
                         const std::string &key) const {
  auto source_guard = source_map_.read_lock();
  auto it = (*source_guard).find(section + "." + key);
  if (it != (*source_guard).end()) {
    return it->second;
  }
  return std::nullopt;
}

std::string ConfigLoader::export_to_json() const {
  std::stringstream ss;
  auto config_guard = config_.read_lock();
  
  ss << "{\n";
  bool first_section = true;

  for (const auto &[section, values] : *config_guard) {
    if (!first_section)
      ss << ",\n";
    first_section = false;

    ss << "  \"" << section << "\": {\n";
    bool first_key = true;

    for (const auto &[key, value] : values) {
      if (!first_key)
        ss << ",\n";
      first_key = false;

      ss << "    \"" << key << "\": ";

      switch (value.type()) {
      case ConfigValue::Type::STRING:
        ss << "\"" << value.as_string() << "\"";
        break;
      case ConfigValue::Type::INT64:
        ss << value.as_int64();
        break;
      case ConfigValue::Type::DOUBLE:
        ss << value.as_double();
        break;
      case ConfigValue::Type::BOOLEAN:
        ss << (value.as_bool() ? "true" : "false");
        break;
      case ConfigValue::Type::STRING_VECTOR:
        ss << "[" << value.to_string() << "]";
        break;
      case ConfigValue::Type::INT64_VECTOR:
        ss << "[" << value.to_string() << "]";
        break;
      case ConfigValue::Type::DOUBLE_VECTOR:
        ss << "[" << value.to_string() << "]";
        break;
      case ConfigValue::Type::NULL_VALUE:
        ss << "null";
        break;
      }
    }

    ss << "\n  }";
  }

  ss << "\n}\n";
  return ss.str();
}

std::string ConfigLoader::export_to_yaml() const {
  std::stringstream ss;
  auto config_guard = config_.read_lock();
  
  ss << "# Auto-generated configuration export\n\n";

  for (const auto &[section, values] : *config_guard) {
    ss << section << ":\n";

    for (const auto &[key, value] : values) {
      ss << "  " << key << ": ";

      switch (value.type()) {
      case ConfigValue::Type::STRING:
        ss << "\"" << value.as_string() << "\"";
        break;
      case ConfigValue::Type::INT64:
        ss << value.as_int64();
        break;
      case ConfigValue::Type::DOUBLE:
        ss << value.as_double();
        break;
      case ConfigValue::Type::BOOLEAN:
        ss << (value.as_bool() ? "true" : "false");
        break;
      case ConfigValue::Type::STRING_VECTOR:
        ss << "[" << value.to_string() << "]";
        break;
      case ConfigValue::Type::INT64_VECTOR:
        ss << "[" << value.to_string() << "]";
        break;
      case ConfigValue::Type::DOUBLE_VECTOR:
        ss << "[" << value.to_string() << "]";
        break;
      case ConfigValue::Type::NULL_VALUE:
        ss << "null";
        break;
      }

      ss << "\n";
    }

    ss << "\n";
  }

  return ss.str();
}

void ConfigLoader::load_environment_variables() {
  // Load BTQ_ prefixed environment variables
  // Collect all updates to apply in a single RCU update
  std::unordered_map<std::string, std::unordered_map<std::string, ConfigValue>> updates;
  std::unordered_map<std::string, ConfigSource> source_updates;

  for (char **env = environ; *env != nullptr; env++) {
    std::string env_str = *env;
    if (env_str.rfind(ENV_PREFIX, 0) == 0) {
      size_t eq_pos = env_str.find('=');
      if (eq_pos != std::string::npos) {
        std::string name = env_str.substr(0, eq_pos);
        std::string value = env_str.substr(eq_pos + 1);

        // Parse config section and key from env var name
        // BTQ_MONITORING_POLL_INTERVAL_MS -> monitoring.poll_interval_ms
        std::string section_key = name.substr(strlen(ENV_PREFIX));
        std::transform(section_key.begin(), section_key.end(),
                       section_key.begin(), ::tolower);

        size_t underscore_pos = section_key.find('_');
        if (underscore_pos != std::string::npos) {
          std::string section = section_key.substr(0, underscore_pos);
          std::string key = section_key.substr(underscore_pos + 1);
          std::replace(key.begin(), key.end(), '_', '.');

          // Try to detect type
          ConfigValue parsed_value;
          if (value == "true" || value == "false") {
            parsed_value = ConfigValue(value == "true");
          } else if (value.find('.') != std::string::npos) {
            try {
              parsed_value = ConfigValue(std::stod(value));
            } catch (...) {
              parsed_value = ConfigValue(value);
            }
          } else {
            try {
              parsed_value =
                  ConfigValue(static_cast<int64_t>(std::stoll(value)));
            } catch (...) {
              parsed_value = ConfigValue(value);
            }
          }

          updates[section][key] = parsed_value;
          source_updates[section + "." + key] = ConfigSource::ENVIRONMENT;
        }
      }
    }
  }

  // Apply all updates using RCU
  config_.update([&updates](ConfigMap &config_map) {
    for (const auto &[section, values] : updates) {
      for (const auto &[key, value] : values) {
        config_map[section][key] = value;
      }
    }
  });

  source_map_.update([&source_updates](std::unordered_map<std::string, ConfigSource> &source_map_ref) {
    for (const auto &[key, source] : source_updates) {
      source_map_ref[key] = source;
    }
  });
}

void ConfigLoader::load_config_file(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "Warning: Could not open config file: " << path << std::endl;
    return;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string content = buffer.str();

  // Simple YAML parser (key: value pairs)
  std::string current_section;

  std::istringstream stream(content);
  std::string line;

  // Collect all updates to apply in a single RCU update
  std::unordered_map<std::string, std::unordered_map<std::string, ConfigValue>> updates;
  std::unordered_map<std::string, ConfigSource> source_updates;

  while (std::getline(stream, line)) {
    // Skip empty lines and comments
    if (line.empty() || line[0] == '#')
      continue;

    // Remove leading/trailing whitespace
    size_t start = line.find_first_not_of(" \t");
    size_t end = line.find_last_not_of(" \t");
    if (start == std::string::npos)
      continue;
    line = line.substr(start, end - start + 1);

    // Check for section header
    if (line[0] != ' ' && line[0] != '\t' &&
        line.find(':') != std::string::npos) {
      size_t colon_pos = line.find(':');
      std::string section = line.substr(0, colon_pos);
      std::string value = line.substr(colon_pos + 1);

      // Remove quotes from section name
      section.erase(std::remove(section.begin(), section.end(), '"'),
                    section.end());
      section.erase(std::remove(section.begin(), section.end(), '\''),
                    section.end());

      current_section = section;
      updates[current_section] = {};

      // Handle inline value
      std::string trimmed_value = value;
      start = trimmed_value.find_first_not_of(" \t");
      end = trimmed_value.find_last_not_of(" \t");
      if (start != std::string::npos && end != std::string::npos) {
        trimmed_value = trimmed_value.substr(start, end - start + 1);
        if (!trimmed_value.empty()) {
          updates[current_section]["_section_value"] = ConfigValue(trimmed_value);
        }
      }
    }
    // Key-value pair
    else if (current_section.empty()) {
      continue;
    } else {
      size_t colon_pos = line.find(':');
      if (colon_pos == std::string::npos)
        continue;

      std::string key = line.substr(0, colon_pos);
      std::string value = line.substr(colon_pos + 1);

      // Get indentation level
      size_t indent = key.find_first_not_of(" \t");
      if (indent == std::string::npos)
        continue;
      key = key.substr(indent);

      // Remove leading/trailing whitespace from value
      start = value.find_first_not_of(" \t");
      end = value.find_last_not_of(" \t");
      if (start != std::string::npos && end != std::string::npos) {
        value = value.substr(start, end - start + 1);
      } else {
        value = "";
      }

      // Remove quotes
      key.erase(std::remove(key.begin(), key.end(), '"'), key.end());
      key.erase(std::remove(key.begin(), key.end(), '\''), key.end());
      value.erase(std::remove(value.begin(), value.end(), '"'), value.end());
      value.erase(std::remove(value.begin(), value.end(), '\''), value.end());

      // Parse value
      ConfigValue parsed_value;
      if (value == "true" || value == "false") {
        parsed_value = ConfigValue(value == "true");
      } else if (value.empty()) {
        // Empty value, skip
        continue;
      } else if (value.find('.') != std::string::npos ||
                 value.find('e') != std::string::npos ||
                 value.find('E') != std::string::npos) {
        try {
          parsed_value = ConfigValue(std::stod(value));
        } catch (...) {
          parsed_value = ConfigValue(value);
        }
      } else {
        try {
          parsed_value = ConfigValue(static_cast<int64_t>(std::stoll(value)));
        } catch (...) {
          parsed_value = ConfigValue(value);
        }
      }

      updates[current_section][key] = parsed_value;
      source_updates[current_section + "." + key] = ConfigSource::LOCAL_CONFIG;
    }
  }

  // Apply all updates using RCU
  config_.update([&updates](ConfigMap &config_map) {
    for (const auto &[section, values] : updates) {
      for (const auto &[key, value] : values) {
        config_map[section][key] = value;
      }
    }
  });

  source_map_.update([&source_updates](std::unordered_map<std::string, ConfigSource> &source_map_ref) {
    for (const auto &[key, source] : source_updates) {
      source_map_ref[key] = source;
    }
  });
}

void ConfigLoader::merge_configurations() {
  // Higher priority sources override lower ones
  // Order: ENVIRONMENT > CLI > USER > LOCAL > SYSTEM > BUILTIN
  // This is handled by the order of loading, so no special implementation needed for RCU
}

std::string ConfigLoader::get_env_var(const std::string &name) {
  const char *value = std::getenv(name.c_str());
  return value ? std::string(value) : "";
}

bool ConfigLoader::get_env_var_as_bool(const std::string &name,
                                       bool default_val) {
  std::string value = get_env_var(name);
  if (value.empty())
    return default_val;
  return value == "true" || value == "1" || value == "yes";
}

int64_t ConfigLoader::get_env_var_as_int(const std::string &name,
                                         int64_t default_val) {
  std::string value = get_env_var(name);
  if (value.empty())
    return default_val;
  try {
    return std::stoll(value);
  } catch (...) {
    return default_val;
  }
}

double ConfigLoader::get_env_var_as_double(const std::string &name,
                                           double default_val) {
  std::string value = get_env_var(name);
  if (value.empty())
    return default_val;
  try {
    return std::stod(value);
  } catch (...) {
    return default_val;
  }
}

} // namespace BTQuant::Config
