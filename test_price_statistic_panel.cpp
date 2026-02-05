#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QPushButton>
#include <QTimer>
#include <QRandomGenerator>
#include <iostream>

#include "dependencies/BTQRenderEngine/include/components/pricestatisticpanel.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QMainWindow window;
    window.setWindowTitle("Price Statistic Panel Test");
    window.resize(1200, 600);

    QWidget *centralWidget = new QWidget(&window);
    QVBoxLayout *layout = new QVBoxLayout(centralWidget);

    // Create the price statistic panel
    PriceStatisticPanel *panel = new PriceStatisticPanel(centralWidget);
    
    // Add sample data
    QVector<PriceStatData> sampleData;
    QStringList symbols = {"AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "ADBE", "PYPL",
                          "INTC", "AMD", "ORCL", "IBM", "SAP", "CRM", "NOW", "SHOP", "SQ", "DOCU"};
    
    for (const QString &symbol : symbols) {
        PriceStatData data;
        data.symbol = symbol;
        data.lastPrice = 100.0 + QRandomGenerator::global()->bounded(200.0);
        data.change = (QRandomGenerator::global()->bounded(100) - 50) / 10.0; // Random change between -5 and 5
        data.changePercent = (data.change / data.lastPrice) * 100;
        data.volume = QRandomGenerator::global()->bounded(1000000) + 100000;
        data.high = data.lastPrice + QRandomGenerator::global()->bounded(5.0);
        data.low = data.lastPrice - QRandomGenerator::global()->bounded(5.0);
        data.open = data.lastPrice + (QRandomGenerator::global()->bounded(100) - 50) / 100.0;
        data.close = data.lastPrice;
        
        sampleData.append(data);
    }
    
    panel->setData(sampleData);
    
    layout->addWidget(panel);
    
    // Add a button to add more data
    QPushButton *addDataButton = new QPushButton("Add Random Data Row", centralWidget);
    QObject::connect(addDataButton, &QPushButton::clicked, [panel]() {
        PriceStatData newData;
        static int counter = 0;
        newData.symbol = QString("SYM%1").arg(++counter);
        newData.lastPrice = 50.0 + QRandomGenerator::global()->bounded(300.0);
        newData.change = (QRandomGenerator::global()->bounded(100) - 50) / 10.0;
        newData.changePercent = (newData.change / newData.lastPrice) * 100;
        newData.volume = QRandomGenerator::global()->bounded(1000000) + 100000;
        newData.high = newData.lastPrice + QRandomGenerator::global()->bounded(5.0);
        newData.low = newData.lastPrice - QRandomGenerator::global()->bounded(5.0);
        newData.open = newData.lastPrice + (QRandomGenerator::global()->bounded(100) - 50) / 100.0;
        newData.close = newData.lastPrice;
        
        panel->addData(newData);
        
        // Calculate and print statistics
        double avgPrice, volatility, priceRange;
        panel->calculateStatistics(avgPrice, volatility, priceRange);
        std::cout << "Statistics - Avg Price: " << avgPrice 
                  << ", Volatility: " << volatility 
                  << ", Price Range: " << priceRange << std::endl;
    });
    
    layout->addWidget(addDataButton);
    
    window.setCentralWidget(centralWidget);
    window.show();

    return app.exec();
}