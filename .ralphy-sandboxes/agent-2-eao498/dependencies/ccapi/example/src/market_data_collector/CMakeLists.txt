add_executable(market_data_collector
    main.cpp
    config_loader.cpp
    mssql_bulk_inserter.cpp
    candle_aggregator.cpp
    market_data_processor.cpp
    exchange_connection_manager.cpp
    market_data_collector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tests/new/src/config/config_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tests/new/src/symbol_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tests/new/src/utils/dynamic_logger.cpp
)

add_executable(test_config_loading
    test_config_loading.cpp
    config_loader.cpp
    mssql_bulk_inserter.cpp
    candle_aggregator.cpp
    market_data_processor.cpp
    exchange_connection_manager.cpp
    market_data_collector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tests/new/src/config/config_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tests/new/src/symbol_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tests/new/src/utils/dynamic_logger.cpp
)

target_include_directories(test_config_loading PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tests/new/include
)

target_link_libraries(test_config_loading
    PRIVATE
        odbc
        pthread
        hotspine_writer
)

# Test for dynamic configuration integration
add_executable(test_dynamic_config
    test_dynamic_config.cpp
    config_loader.cpp
    mssql_bulk_inserter.cpp
    candle_aggregator.cpp
    market_data_processor.cpp
    exchange_connection_manager.cpp
    market_data_collector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tests/new/src/config/config_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tests/new/src/symbol_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tests/new/src/hotspine_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tests/new/src/hotspine_extended_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tests/new/src/utils/dynamic_logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tests/new/src/utils/alert_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tests/new/src/utils/shm_health_monitor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tests/new/src/utils/recovery_controller.cpp
)

target_include_directories(test_dynamic_config PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tests/new/include
)

target_link_libraries(test_dynamic_config
    PRIVATE
        odbc
        pthread
        hotspine_writer
        hotspine_reader
)

target_include_directories(market_data_collector PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tests/new/include
)

target_link_libraries(market_data_collector
    PRIVATE
        odbc
        pthread
        hotspine_writer
)

target_compile_definitions(market_data_collector
    PRIVATE
        CCAPI_ENABLE_SERVICE_MARKET_DATA
        CCAPI_ENABLE_EXCHANGE_BINANCE
        CCAPI_ENABLE_EXCHANGE_OKX
        CCAPI_ENABLE_EXCHANGE_BYBIT
)

# Suppress Boost ASIO warning in older Boost versions (buffer overflow in basic_resolver_results)
# This is a known issue in Boost < 1.75, the warning is a false positive for IPv6 resolver results
target_compile_options(market_data_collector PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wno-stringop-overflow;-Wno-stringop-overread>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-uninitialized;-Wno-string-plus-int>
)
