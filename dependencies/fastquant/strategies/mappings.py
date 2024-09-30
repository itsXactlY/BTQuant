# Import from package
from fastquant.strategies import (
    RSIStrategy,
    SMACStrategy,
    BaseStrategy,
    MACDStrategy,
    EMACStrategy,
    BBandsStrategy,
    BuyAndHoldStrategy,
    SentimentStrategy,
    CustomStrategy,
    TernaryStrategy,
    QQE_Example,
    Order_Chain_Kioseff_Trading,
    SMA_Cross_MESAdaptivePrime,
    SuperSTrend_Scalper,
)

# Register your strategy here
STRATEGY_MAPPING = {
    "rsi": RSIStrategy,
    "smac": SMACStrategy,
    "macd": MACDStrategy,
    "emac": EMACStrategy,
    "bbands": BBandsStrategy,
    "buynhold": BuyAndHoldStrategy,
    "sentiment": SentimentStrategy,
    "custom": CustomStrategy,
    "ternary": TernaryStrategy,
    "qqe": QQE_Example,
    "OrChainKioseff": Order_Chain_Kioseff_Trading,
    "msa": SMA_Cross_MESAdaptivePrime,
    "STScalp": SuperSTrend_Scalper,
}
