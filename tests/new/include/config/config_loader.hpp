#pragma once

#include <string>
#include <unordered_map>
#include <optional>
#include <vector>
#include <functional>
#include <variant>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <cctype>

namespace BTQuant::Config {

// Configuration value types
class ConfigValue {
public:
    // Supported types
    enum class Type {
        STRING,
        INT64,
        DOUBLE,
        BOOLEAN,
        STRING_VECTOR,
        INT64_VECTOR,
        DOUBLE_VECTOR,
        NULL_VALUE
    };

    // Constructors for each type
    ConfigValue() : type_(Type::NULL_VALUE) {}
    explicit ConfigValue(const std::string& value) : type_(Type::STRING), string_value_(value) {}
    explicit ConfigValue(int64_t value) : type_(Type::INT64), int64_value_(value) {}
    explicit ConfigValue(double value) : type_(Type::DOUBLE), double_value_(value) {}
    explicit ConfigValue(bool value) : type_(Type::BOOLEAN), bool_value_(value) {}
    explicit ConfigValue(const std::vector<std::string>& value) : type_(Type::STRING_VECTOR), string_vector_value_(value) {}
    explicit ConfigValue(const std::vector<int64_t>& value) : type_(Type::INT64_VECTOR), int64_vector_value_(value) {}
    explicit ConfigValue(const std::vector<double>& value) : type_(Type::DOUBLE_VECTOR), double_vector_value_(value) {}

    // Get type
    Type type() const { return type_; }

    // Get value methods
    std::string as_string() const;
    int64_t as_int64() const;
    double as_double() const;
    bool as_bool() const;
    std::vector<std::string> as_string_vector() const;
    std::vector<int64_t> as_int64_vector() const;
    std::vector<double> as_double_vector() const;

    // Conversion methods
    bool to_string(std::string& result) const;
    bool to_int64(int64_t& result) const;
    bool to_double(double& result) const;
    bool to_bool(bool& result) const;

    // Assignment operators
    ConfigValue& operator=(const std::string& value) {
        type_ = Type::STRING;
        string_value_ = value;
        return *this;
    }

    ConfigValue& operator=(int64_t value) {
        type_ = Type::INT64;
        int64_value_ = value;
        return *this;
    }

    ConfigValue& operator=(double value) {
        type_ = Type::DOUBLE;
        double_value_ = value;
        return *this;
    }

    ConfigValue& operator=(bool value) {
        type_ = Type::BOOLEAN;
        bool_value_ = value;
        return *this;
    }

    // Comparison operators
    bool operator==(const ConfigValue& other) const;
    bool operator!=(const ConfigValue& other) const;

    // String representation
    std::string to_string() const;

private:
    Type type_;
    
    // Union for storing values
    std::string string_value_;
    int64_t int64_value_;
    double double_value_;
    bool bool_value_;
    std::vector<std::string> string_vector_value_;
    std::vector<int64_t> int64_vector_value_;
    std::vector<double> double_vector_value_;
};

// ConfigValue method implementations

inline std::string ConfigValue::as_string() const {
    switch (type_) {
        case Type::STRING: return string_value_;
        case Type::INT64: return std::to_string(int64_value_);
        case Type::DOUBLE: return std::to_string(double_value_);
        case Type::BOOLEAN: return bool_value_ ? "true" : "false";
        case Type::STRING_VECTOR: {
            std::string result = "[";
            for (size_t i = 0; i < string_vector_value_.size(); ++i) {
                if (i > 0) result += ", ";
                result += "\"" + string_vector_value_[i] + "\"";
            }
            result += "]";
            return result;
        }
        case Type::INT64_VECTOR: {
            std::string result = "[";
            for (size_t i = 0; i < int64_vector_value_.size(); ++i) {
                if (i > 0) result += ", ";
                result += std::to_string(int64_vector_value_[i]);
            }
            result += "]";
            return result;
        }
        case Type::DOUBLE_VECTOR: {
            std::string result = "[";
            for (size_t i = 0; i < double_vector_value_.size(); ++i) {
                if (i > 0) result += ", ";
                result += std::to_string(double_vector_value_[i]);
            }
            result += "]";
            return result;
        }
        case Type::NULL_VALUE:
        default: return "";
    }
}

inline int64_t ConfigValue::as_int64() const {
    switch (type_) {
        case Type::INT64: return int64_value_;
        case Type::STRING: return std::stoll(string_value_);
        case Type::DOUBLE: return static_cast<int64_t>(double_value_);
        case Type::BOOLEAN: return bool_value_ ? 1 : 0;
        default: return 0;
    }
}

inline double ConfigValue::as_double() const {
    switch (type_) {
        case Type::DOUBLE: return double_value_;
        case Type::INT64: return static_cast<double>(int64_value_);
        case Type::STRING: return std::stod(string_value_);
        case Type::BOOLEAN: return bool_value_ ? 1.0 : 0.0;
        default: return 0.0;
    }
}

inline bool ConfigValue::as_bool() const {
    switch (type_) {
        case Type::BOOLEAN: return bool_value_;
        case Type::STRING: {
            std::string lower = string_value_;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            return (lower == "true" || lower == "1" || lower == "yes" || lower == "on");
        }
        case Type::INT64: return int64_value_ != 0;
        case Type::DOUBLE: return double_value_ != 0.0;
        default: return false;
    }
}

inline std::vector<std::string> ConfigValue::as_string_vector() const {
    if (type_ == Type::STRING_VECTOR) return string_vector_value_;
    return {};
}

inline std::vector<int64_t> ConfigValue::as_int64_vector() const {
    if (type_ == Type::INT64_VECTOR) return int64_vector_value_;
    return {};
}

inline std::vector<double> ConfigValue::as_double_vector() const {
    if (type_ == Type::DOUBLE_VECTOR) return double_vector_value_;
    return {};
}

inline bool ConfigValue::to_string(std::string& result) const {
    if (type_ == Type::STRING) {
        result = string_value_;
        return true;
    }
    return false;
}

inline bool ConfigValue::to_int64(int64_t& result) const {
    if (type_ == Type::INT64) {
        result = int64_value_;
        return true;
    }
    return false;
}

inline bool ConfigValue::to_double(double& result) const {
    if (type_ == Type::DOUBLE) {
        result = double_value_;
        return true;
    }
    return false;
}

inline bool ConfigValue::to_bool(bool& result) const {
    if (type_ == Type::BOOLEAN) {
        result = bool_value_;
        return true;
    }
    return false;
}

inline bool ConfigValue::operator==(const ConfigValue& other) const {
    if (type_ != other.type_) return false;
    
    switch (type_) {
        case Type::STRING: return string_value_ == other.string_value_;
        case Type::INT64: return int64_value_ == other.int64_value_;
        case Type::DOUBLE: return double_value_ == other.double_value_;
        case Type::BOOLEAN: return bool_value_ == other.bool_value_;
        case Type::STRING_VECTOR: return string_vector_value_ == other.string_vector_value_;
        case Type::INT64_VECTOR: return int64_vector_value_ == other.int64_vector_value_;
        case Type::DOUBLE_VECTOR: return double_vector_value_ == other.double_vector_value_;
        case Type::NULL_VALUE: return true;
        default: return false;
    }
}

inline bool ConfigValue::operator!=(const ConfigValue& other) const {
    return !(*this == other);
}

inline std::string ConfigValue::to_string() const {
    return as_string();
}

// Configuration section
using ConfigSection = std::unordered_map<std::string, ConfigValue>;

// Full configuration map
using ConfigMap = std::unordered_map<std::string, ConfigSection>;

// Configuration source priority
enum class ConfigSource {
    ENVIRONMENT,
    CLI_ARGUMENT,
    USER_CONFIG,
    LOCAL_CONFIG,
    SYSTEM_CONFIG,
    BUILTIN_DEFAULT
};

// Configuration change callback
using ConfigChangeCallback = std::function<void(const std::string& section, 
                                                 const std::string& key,
                                                 const ConfigValue& old_value,
                                                 const ConfigValue& new_value)>;

class ConfigLoader {
public:
    static ConfigLoader& instance();
    
    // Initialize configuration system
    bool initialize(const std::string& app_path = "");
    
    // Load configuration from all sources
    bool load();
    
    // Get configuration value
    std::optional<ConfigValue> get(const std::string& section, 
                                    const std::string& key) const;
    
    // Get with type conversion
    template<typename T>
    std::optional<T> get_as(const std::string& section, 
                             const std::string& key) const;
    
    // Set configuration value
    bool set(const std::string& section, 
             const std::string& key, 
             const ConfigValue& value);
    
    // Get all sections
    std::vector<std::string> get_sections() const;
    
    // Register change callback
    void register_change_callback(ConfigChangeCallback callback);
    
    // Reload configuration
    bool reload();
    
    // Get configuration source for a value
    std::optional<ConfigSource> get_source(const std::string& section,
                                            const std::string& key) const;
    
    // Export configuration
    std::string export_to_json() const;
    std::string export_to_yaml() const;
    
    // Environment variable helpers
    static std::string get_env_var(const std::string& name);
    static bool get_env_var_as_bool(const std::string& name, bool default_val = false);
    static int64_t get_env_var_as_int(const std::string& name, int64_t default_val = 0);
    static double get_env_var_as_double(const std::string& name, double default_val = 0.0);
    
private:
    ConfigLoader() = default;
    ~ConfigLoader() = default;
    ConfigLoader(const ConfigLoader&) = delete;
    ConfigLoader& operator=(const ConfigLoader&) = delete;
    
    void discover_config_files();
    void load_environment_variables();
    void load_config_file(const std::string& path);
    void merge_configurations();
    
    ConfigMap config_;
    std::unordered_map<std::string, ConfigSource> source_map_;
    std::vector<ConfigChangeCallback> change_callbacks_;
    std::vector<std::string> config_search_paths_;
    std::string app_path_;
    std::string main_config_path_;
    std::string symbol_mapping_path_;
    bool initialized_ = false;
    mutable std::mutex mutex_;
};

// Template implementations

// Helper function to convert string to other types
template<typename T>
inline bool convert_from_string(const std::string& str, T& result);

template<>
inline bool convert_from_string<int64_t>(const std::string& str, int64_t& result) {
    try {
        result = std::stoll(str);
        return true;
    } catch (...) {
        return false;
    }
}

template<>
inline bool convert_from_string<double>(const std::string& str, double& result) {
    try {
        result = std::stod(str);
        return true;
    } catch (...) {
        return false;
    }
}

template<>
inline bool convert_from_string<bool>(const std::string& str, bool& result) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    result = (lower == "true" || lower == "1" || lower == "yes" || lower == "on");
    return true;
}

// Helper to convert value to string
inline std::string value_to_string(const ConfigValue& value) {
    return value.to_string();
}

template<typename T>
std::optional<T> ConfigLoader::get_as(const std::string& section,
                                        const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto section_it = config_.find(section);
    if (section_it == config_.end()) {
        return std::nullopt;
    }
    
    auto key_it = section_it->second.find(key);
    if (key_it == section_it->second.end()) {
        return std::nullopt;
    }
    
    const ConfigValue& value = key_it->second;
    
    // Direct type match
    if constexpr (std::is_same_v<T, std::string>) {
        return value.as_string();
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return value.as_int64();
    } else if constexpr (std::is_same_v<T, double>) {
        return value.as_double();
    } else if constexpr (std::is_same_v<T, bool>) {
        return value.as_bool();
    } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
        return value.as_string_vector();
    } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
        return value.as_int64_vector();
    } else if constexpr (std::is_same_v<T, std::vector<double>>) {
        return value.as_double_vector();
    }
    
    return std::nullopt;
}

// Convenience macros for configuration access
#define CONFIG_GET(section, key, type, default_value) \
    ConfigLoader::instance().get_as<type>((section), (key)).value_or(default_value)

#define CONFIG_REQUIRED(section, key, type) \
    ConfigLoader::instance().get_as<type>((section), (key))

} // namespace BTQuant::Config
