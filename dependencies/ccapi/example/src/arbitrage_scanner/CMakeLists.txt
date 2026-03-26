add_executable(arbitrage_scanner
    main.cpp
    arbitrage_scanner.cpp
    arbitrage_types.h
    arbitrage_scanner.h
)

target_compile_definitions(arbitrage_scanner
    PRIVATE
        CCAPI_ENABLE_SERVICE_MARKET_DATA
        CCAPI_ENABLE_EXCHANGE_BINANCE
        CCAPI_ENABLE_EXCHANGE_OKX
        CCAPI_ENABLE_EXCHANGE_BYBIT
        CCAPI_ENABLE_EXCHANGE_KRAKEN
        CCAPI_ENABLE_EXCHANGE_COINBASE
)

target_include_directories(arbitrage_scanner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# target_link_libraries(arbitrage_scanner
#     PRIVATE
#         odbc
#         pthread
# )
