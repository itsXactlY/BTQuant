Howto CCAPI from scratch:



git clone https://github.com/itsXactlY/ccapi # important to use this fork for market data collector, etc...
cd ccapi
mkdir example/build
cd example/build
rm -rf * (if rebuild from scratch)
cmake ..
cmake --build .