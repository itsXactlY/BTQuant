# 🚀 **BTQuant Framework** - Unleashing the Power of Quantitative Trading

> **Work in Progress**: If you came this far, don't miss out—[Join our Discord](https://discord.gg/Y7uBxmRg3Z) to stay updated!

---

## **📌 Overview**

BTQuant is a groundbreaking framework crafted for **Backtesting**, **Forward Trading**, and **Live Trading**. Whether you're a seasoned quant or a new algorithmic trader, BTQuant brings powerful features that integrate effortlessly with **Backtrader** and **JackRabbitRelay**—offering the closest simulation of real-world trading conditions. With 1-second WebSocket data feeds and access to **400+ exchanges**, the framework is engineered to give you total control over your trading environment.

---

## **🔥 Jaw-Dropping Features**

### 🔎 **1. Backtesting**
- **Comprehensive Strategy Testing**: Powered by **Backtrader**, our backtesting module allows traders to test strategies on historical data, simulating real-market conditions with precision.
- **MS SQL Integration (Optional)**: Easily manage large volumes of data with **Microsoft SQL Server**—streamlining your backtesting and strategy analysis.
- **Optimized Infrastructure**: Effortless setup that ensures scalability and security for data-driven decisions.

### 🚀 **2. Forward Trading**
- **Next-Level Simulation**: Forward test on **400+ simulated exchanges** with **JackRabbitRelay**—a powerful tool to mimic market conditions and test strategies before going live.
- **Broker Independence**: No need to rely on external brokers. **Act as your own broker**, giving you unmatched flexibility.
- **Hyper-Realistic Environments**: Designed to simulate minute-by-minute market dynamics, enabling robust performance analysis.

### ⏱️ **3. Live Trading**
- **Real-Time Trading**: Live trade using **1-second WebSocket data feeds** from **Binance**, **ByBit**, and now **PancakeSwap** for decentralized exchanges.
- **Seamless Strategy Deployment**: Deploy your trading strategy in as few as **50 lines of code**, empowering you to focus on market analysis rather than infrastructure headaches.
- **DCA Logic**: Implement automatic **Dollar Cost Averaging (DCA)** to refine your entry and exit points.
- **PancakeSwap**: Experience decentralized trading with **1-second WebSocket data**.

---

## **📂 Getting Started**

### **🖥️ Pre-requisites**
- **Linux VPS** for **JackRabbitRelay** (Tested on: **Arch (Garuda Linux)**, **Debian 12.5 (Bookworm)**)
- **Python 3.12+**
- **Custom-Backtrader** (found in the `dependencies` folder)
- [**JackRabbitRelay**](https://github.com/rapmd73/JackrabbitRelay) for live trading
- [**JackRabbitRelay Mimic**](https://github.com/rapmd73/JackrabbitRelay/wiki/Jackrabbit-Mimic) for forward trading
- Large datasets in CSV format—**[Binance.com 1minute Candle Stash at Protondrive](https://drive.proton.me/urls/K19ADZ4DZM#D9s3zyRrZH1m)**

### **🔧 Optional**
- **Microsoft SQL Server** for advanced data handling and backtesting capabilities

---

## **🚀 Installation**

To install **JackRabbitRelay**, follow the detailed guides:

- [JackRabbitRelay GitHub](https://github.com/rapmd73/JackrabbitRelay)
- [Installation Guide](https://github.com/rapmd73/JackrabbitRelay/wiki/Installation-and-Setup#installing-and-setting-up-version-2)
- [Mimic "Brokerage" Setup](https://github.com/rapmd73/JackrabbitRelay/wiki/Jackrabbit-Mimic)

### **⚡ Quick Setup for BTQuant**

Run this command in your terminal:

```bash
curl -fsSL https://raw.githubusercontent.com/itsXactlY/BTQuant/refs/heads/dev/install.sh | sh
```

This script will handle the setup of the virtual environment, installation of dependencies, and configuration of the framework for Linux systems.


---

## **💻 Running the Framework**

### **For Backtesting:**
```bash
python3 Crypto_CCXT_BollingerBands.py
```

### **For Forward/Live Trading:**
- **Setup**: Copy `dontcommit.py.template` to `dontcommit.py` and fill in the identification strings from your JackRabbitRelay setup, or Privatekey for Web3 Access.
- **Run Forward Trading on Binance:**
```bash
python3 __LIVE_Binance.py
```
- **Run Forward Trading on PancakeSwap:**
```bash
python3 __LIVE__PancakeSwap.py
```

---

## **🚀 Why BTQuant?**

### **What We Offer That Others Don’t**
- **Mimic 400+ exchanges** for forward testing with unrivaled accuracy.
- **Live 1-second WebSocket data feeds** for real-time trading, **bypassing common data delays**.
- **Act as your own broker**, giving you total control and independence.
- Built-in **DCA logic**, reducing your entry risks.
- Plug-and-play **50-line code** setup for strategy deployment.
- **MS SQL Integration** for traders handling large datasets, ensuring security and efficiency.
- **PancakeSwap integration** for decentralized trading experience.

---

> **🚀 Ready to Take Your Trading to the Next Level?**  
> Clone, install, and get trading now!

> **Join our Community**: Have questions? Need help? [Join our Discord](https://discord.gg/Y7uBxmRg3Z)


