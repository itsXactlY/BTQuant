#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace BTQuant::Logging {

// Log levels
enum class LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARNING = 2,
  ERROR = 3,
  CRITICAL = 4
};

// Log message structure
struct LogMessage {
  LogLevel level;
  std::string timestamp;
  std::string component;
  std::string file;
  int line;
  std::string function;
  std::string message;
  std::unordered_map<std::string, std::string> context;
};

// Log sink interface
class ILogSink {
public:
  virtual ~ILogSink() = default;
  virtual void write(const LogMessage &message) = 0;
  virtual void flush() = 0;
  virtual bool is_enabled(LogLevel level) const = 0;
  virtual std::string get_name() const = 0;
};

// Console log sink
class ConsoleLogSink : public ILogSink {
public:
  void write(const LogMessage &message) override;
  void flush() override;
  bool is_enabled(LogLevel level) const override;
  std::string get_name() const override { return "console"; }
};

// File log sink
class FileLogSink : public ILogSink {
public:
  explicit FileLogSink(const std::string &path);
  ~FileLogSink() override;

  void write(const LogMessage &message) override;
  void flush() override;
  bool is_enabled(LogLevel level) const override;
  std::string get_name() const override { return "file"; }

  bool open(const std::string &path);
  void close();

private:
  std::string path_;
  FILE *file_ = nullptr;
  std::mutex file_mutex_;
};

// Dynamic logger class
class DynamicLogger {
public:
  static DynamicLogger &instance();

  // Initialize logger
  bool initialize(const std::string &config_path = "");

  // Set global log level
  void set_level(LogLevel level);
  void set_level(const std::string &component, LogLevel level);
  void set_level_from_string(const std::string &level_str);

  // Set minimum level for all components
  void set_min_level(LogLevel level);

  // Add log sink
  void add_sink(std::shared_ptr<ILogSink> sink);
  void remove_sink(const std::string &name);

  // Check if logging is enabled for a component and level
  bool is_enabled(LogLevel level, const std::string &component) const;

  // Log message
  void log(LogLevel level, const std::string &component,
           const std::string &message, const std::string &file = "",
           int line = 0, const std::string &function = "",
           std::unordered_map<std::string, std::string> context = {});

  // Convenience methods
  void debug(const std::string &component, const std::string &message,
             const std::string &file = "", int line = 0,
             const std::string &function = "");
  void info(const std::string &component, const std::string &message,
            const std::string &file = "", int line = 0,
            const std::string &function = "");
  void warning(const std::string &component, const std::string &message,
               const std::string &file = "", int line = 0,
               const std::string &function = "");
  void error(const std::string &component, const std::string &message,
             const std::string &file = "", int line = 0,
             const std::string &function = "");
  void critical(const std::string &component, const std::string &message,
                const std::string &file = "", int line = 0,
                const std::string &function = "");

  // Dynamic configuration
  void configure(const std::unordered_map<std::string, std::string> &config);

  // Component registration
  void register_component(const std::string &name);
  void unregister_component(const std::string &name);

  // Statistics
  uint64_t get_message_count(LogLevel level) const;
  uint64_t get_total_message_count() const;
  void reset_statistics();

  // Shutdown
  void shutdown();

private:
  DynamicLogger() = default;
  ~DynamicLogger();
  DynamicLogger(const DynamicLogger &) = delete;
  DynamicLogger &operator=(const DynamicLogger &) = delete;

  void write_message(LogLevel level, const std::string &component,
                     const std::string &message, const std::string &file,
                     int line, const std::string &function);

  std::string get_timestamp();

  std::atomic<LogLevel> global_level_{LogLevel::INFO};
  std::atomic<LogLevel> min_level_{LogLevel::DEBUG};
  std::unordered_map<std::string, LogLevel> component_levels_;
  std::unordered_map<std::string, std::shared_ptr<ILogSink>> sinks_;
  std::unordered_map<int, uint64_t> message_counts_;
  std::atomic<uint64_t> total_messages_{0};
  std::atomic<bool> initialized_{false};
  std::mutex mutex_;
  std::mutex init_mutex_;
};

// Log stream helper
class LogStream {
public:
  LogStream(DynamicLogger &logger, LogLevel level, const std::string &component,
            const std::string &file, int line, const std::string &function)
      : logger_(logger), level_(level), component_(component), file_(file),
        line_(line), function_(function) {}

  ~LogStream() {
    std::string content = ss_.str();
    if (!content.empty()) {
      logger_.log(level_, component_, content, file_, line_, function_);
    }
  }

  std::ostringstream &stream() { return ss_; }

private:
  DynamicLogger &logger_;
  LogLevel level_;
  std::string component_;
  std::string file_;
  int line_;
  std::string function_;
  std::ostringstream ss_;
};

// Macro for logging with location
#define BTQ_LOG_DEBUG(component, message)                                      \
  LogStream(DynamicLogger::instance(), LogLevel::DEBUG, component, __FILE__,   \
            __LINE__, __FUNCTION__)                                            \
      .stream()

#define BTQ_LOG_INFO(component, message)                                       \
  LogStream(DynamicLogger::instance(), LogLevel::INFO, component, __FILE__,    \
            __LINE__, __FUNCTION__)                                            \
      .stream()

#define BTQ_LOG_WARNING(component, message)                                    \
  LogStream(DynamicLogger::instance(), LogLevel::WARNING, component, __FILE__, \
            __LINE__, __FUNCTION__)                                            \
      .stream()

#define BTQ_LOG_ERROR(component, message)                                      \
  LogStream(DynamicLogger::instance(), LogLevel::ERROR, component, __FILE__,   \
            __LINE__, __FUNCTION__)                                            \
      .stream()

#define BTQ_LOG_CRITICAL(component, message)                                   \
  LogStream(DynamicLogger::instance(), LogLevel::CRITICAL, component,          \
            __FILE__, __LINE__, __FUNCTION__)                                  \
      .stream()

// Convenience macro without stream
#define BTQ_LOG(component, level, message)                                     \
  DynamicLogger::instance().log(level, component, message, __FILE__, __LINE__, \
                                __FUNCTION__)

} // namespace BTQuant::Logging
