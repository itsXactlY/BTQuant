#include "ccapi_cpp/ccapi_session.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <sys/resource.h>

double getProcessCpuSeconds() {
  rusage u{};
  getrusage(RUSAGE_SELF, &u);
  double user = u.ru_utime.tv_sec + u.ru_utime.tv_usec / 1e6;
  double sys  = u.ru_stime.tv_sec + u.ru_stime.tv_usec / 1e6;
  return user + sys;
}

namespace ccapi {

Logger* Logger::logger = nullptr;

class MyEventHandler : public EventHandler {
 public:
  MyEventHandler(std::size_t printEveryN,
                 const std::string& logPath = "okx_stats.csv")
      : printEveryN_(printEveryN),
        start_(std::chrono::steady_clock::now()),
        lastPrint_(start_),
        cpuStart_(getProcessCpuSeconds()),
        logFile_(logPath, std::ios::out) {
    // CSV header
    logFile_ << "t_wall,total_events,inst_rate,avg_rate,cpu_all,cpu_per_sec\n";
  }

  void processEvent(const Event& event, Session* /*sessionPtr*/) override {
    const auto& eventType = event.getType();

    // Session + subscription status
    if (eventType == Event::Type::SESSION_STATUS ||
        eventType == Event::Type::SUBSCRIPTION_STATUS) {
      std::cout << event.toPrettyString(2, 2) << std::endl;
      return;
    }

    // stream data
    if (eventType == Event::Type::SUBSCRIPTION_DATA) {
      ++subscriptionDataEventCount_;

      if (subscriptionDataEventCount_ % printEveryN_ == 0) {
        auto now       = std::chrono::steady_clock::now();
        double wallAll = std::chrono::duration<double>(now - start_).count();
        double wallWin = std::chrono::duration<double>(now - lastPrint_).count();

        double cpuNow    = getProcessCpuSeconds();
        double cpuAll    = cpuNow - cpuStart_;
        double cpuPerSec = cpuAll / wallAll;
        double instRate  = printEveryN_ / wallWin;
        double avgRate   = subscriptionDataEventCount_ / wallAll;

        // console
        std::cout << "\n=== STATS ===\n"
                  << "Total events: " << subscriptionDataEventCount_ << "\n"
                  << "Window: " << wallWin << " s, total: " << wallAll << " s\n"
                  << "Instant rate: " << instRate << " ev/s\n"
                  << "Average rate: " << avgRate << " ev/s\n"
                  << "CPU time: " << cpuAll << " s ("
                  << cpuPerSec << " CPU s / wall s)\n";

        // CSV line
        logFile_ << std::fixed << std::setprecision(6)
                 << wallAll << ","
                 << subscriptionDataEventCount_ << ","
                 << instRate << ","
                 << avgRate << ","
                 << cpuAll << ","
                 << cpuPerSec << "\n";

        lastPrint_ = now;
      }
    }
  }

 private:
    std::size_t printEveryN_;
    std::size_t subscriptionDataEventCount_ = 0;
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point lastPrint_;
    double cpuStart_;
    std::ofstream logFile_;
};

}  // namespace ccapi

// bring ccapi types into global ns for brevity
using ::ccapi::Event;
using ::ccapi::MyEventHandler;
using ::ccapi::Queue;
using ::ccapi::Request;
using ::ccapi::Session;
using ::ccapi::SessionConfigs;
using ::ccapi::SessionOptions;
using ::ccapi::Subscription;
using ::ccapi::UtilSystem;

// for sorting by 24h quote volume
struct Ticker {
  std::string instrument;
  double quoteVolume24h{};
};

int main(int argc, char** argv) {
  const std::string exchange = "okx";

  // env overrides
  int numSymbols       = UtilSystem::getEnvAsInt("NUM_SYMBOLS", INT_MAX);   // default: all
  int stopTimeSeconds  = UtilSystem::getEnvAsInt("STOP_TIME_IN_SECONDS", 60);
  int printEveryN      = UtilSystem::getEnvAsInt("PRINT_EVERY_N", 1000);

  SessionOptions sessionOptions;
  SessionConfigs sessionConfigs;
  MyEventHandler eventHandler(static_cast<std::size_t>(printEveryN));
  Session session(sessionOptions, sessionConfigs, &eventHandler);

  Request request(Request::Operation::GET_TICKERS, exchange);
  request.appendParam({{"INSTRUMENT_TYPE", "SPOT"}});

  Queue<Event> eventQueue;
  session.sendRequest(request, &eventQueue);  // blocking until response

  auto eventList = eventQueue.purge();
  if (eventList.empty()) {
    std::cerr << "No response to GET_TICKERS\n";
    return EXIT_FAILURE;
  }

  const auto& message = eventList.front().getMessageList().front();
  std::vector<Ticker> tickers;
  tickers.reserve(message.getElementList().size());

  for (const auto& element : message.getElementList()) {
    const auto& instrument = element.getValue("INSTRUMENT");
    double volume24h       = std::stod(element.getValue("VOLUME_24H"));

    if (volume24h <= 0) {
      continue;
    }

    double open  = std::stod(element.getValue("OPEN_24H_PRICE"));
    double high  = std::stod(element.getValue("HIGH_24H_PRICE"));
    double low   = std::stod(element.getValue("LOW_24H_PRICE"));
    double last  = std::stod(element.getValue("LAST_PRICE"));
    double avg   = (open + high + low + last) / 4.0;
    double quote = avg * volume24h;

    tickers.push_back({instrument, quote});
  }

  if (tickers.empty()) {
    std::cerr << "No tickers with positive 24h volume\n";
    return EXIT_FAILURE;
  }

  std::sort(tickers.begin(), tickers.end(),
            [](const Ticker& a, const Ticker& b) {
              return a.quoteVolume24h > b.quoteVolume24h;
            });

  int maxSymbols = std::min(numSymbols,
                            static_cast<int>(tickers.size()));
  if (maxSymbols <= 0) {
    maxSymbols = static_cast<int>(tickers.size());
  }

  std::vector<std::string> instruments;
  instruments.reserve(maxSymbols);
  for (int i = 0; i < maxSymbols; ++i) {
    instruments.push_back(tickers[i].instrument);
  }

  std::cout << "Subscribing to " << instruments.size()
            << " OKX SPOT instruments (TRADE + DEPTH)\n";

  std::vector<Subscription> subscriptionList;
  subscriptionList.reserve(instruments.size() * 2);

  for (const auto& instrument : instruments) {
    subscriptionList.emplace_back(exchange, instrument, "MARKET_DEPTH");
    subscriptionList.emplace_back(exchange, instrument, "TRADE");
  }

  std::cout << "Total subscriptions: " << subscriptionList.size() << "\n";
  session.subscribe(subscriptionList);
  std::cout << "Streaming for " << stopTimeSeconds << " seconds...\n";
  std::this_thread::sleep_for(std::chrono::seconds(stopTimeSeconds));

  session.stop();
  std::cout << "Bye\n";
  return EXIT_SUCCESS;
}
