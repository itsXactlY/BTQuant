add_executable(market_data_collector
    main.cpp
    mssql_bulk_inserter.cpp
    candle_aggregator.cpp
    market_data_processor.cpp
    exchange_connection_manager.cpp
    market_data_collector.cpp
)

add_executable(test_config_loading
    test_config_loading.cpp
    main.cpp
    mssql_bulk_inserter.cpp
    candle_aggregator.cpp
    market_data_processor.cpp
    exchange_connection_manager.cpp
    market_data_collector.cpp
)

target_include_directories(test_config_loading PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(test_config_loading
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

target_include_directories(market_data_collector PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(market_data_collector
    PRIVATE
        odbc
        pthread
        hotspine_writer
)
