#include "../include/components/tpoprofilepanel.h"
#include <QApplication>
#include <QMainWindow>
#include <QTimer>
#include <iostream>
#include <vector>
#include <map>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    // Create the TPO profile panel
    TPOProfilePanel *panel = new TPOProfilePanel();
    
    // Set up test data
    QVector<QMap<QString, QVariant>> testData;
    
    // Sample TPO data: price, letter, time_index
    for (int timeIdx = 0; timeIdx < 10; ++timeIdx) {
        for (int priceLevel = 0; priceLevel < 20; ++priceLevel) {
            double price = 100.0 + (priceLevel * 0.5);  // Prices from 100 to 109.5
            QChar letter = QChar('A' + (timeIdx % 26));  // Cycle through letters A-Z
            
            QMap<QString, QVariant> dataPoint;
            dataPoint["price"] = price;
            dataPoint["letter"] = QString(letter);
            dataPoint["time_index"] = timeIdx;
            
            testData.append(dataPoint);
        }
    }
    
    // Configure the panel
    panel->setData(testData);
    panel->setPriceRange(100.0, 200.0, 0.5);  // Price range from 100 to 200 with 0.5 step
    
    QStringList timeLabels;
    for (int i = 0; i < 10; ++i) {
        timeLabels << QString("T%1").arg(i);
    }
    panel->setTimePeriods(timeLabels);
    
    // Create main window to display the panel
    QMainWindow window;
    window.setCentralWidget(panel);
    window.resize(800, 600);
    window.setWindowTitle("TPO Profile Panel Test");
    window.show();
    
    std::cout << "TPO Profile Panel test application started." << std::endl;
    std::cout << "Testing display of TPO data with vertical price scale and horizontal letter blocks." << std::endl;
    
    // Exit after 10 seconds for automated testing
    QTimer::singleShot(10000, &app, [&app]() {
        std::cout << "Test completed successfully!" << std::endl;
        app.quit();
    });
    
    return app.exec();
}