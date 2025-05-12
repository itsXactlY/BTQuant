^CTraceback (most recent call last):
  File "/home/alca/projects/BTQuant/testing_ccxt.py", line 111, in <module>
    run()
    ~~~^^
  File "/home/alca/projects/BTQuant/testing_ccxt.py", line 106, in run
    cerebro.run(live=True, runonce=False)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/alca/projects/BTQuant/.btq/lib/python3.13/site-packages/backtrader/cerebro.py", line 1127, in run
    runstrat = self.runstrategies(iterstrat)
  File "/home/alca/projects/BTQuant/.btq/lib/python3.13/site-packages/backtrader/cerebro.py", line 1298, in runstrategies
    self._runnext(runstrats)
    ~~~~~~~~~~~~~^^^^^^^^^^^
  File "/home/alca/projects/BTQuant/.btq/lib/python3.13/site-packages/backtrader/cerebro.py", line 1630, in _runnext
    strat._next()
    ~~~~~~~~~~~^^
  File "/home/alca/projects/BTQuant/.btq/lib/python3.13/site-packages/backtrader/strategy.py", line 351, in _next
    self._next_observers(minperstatus)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/home/alca/projects/BTQuant/.btq/lib/python3.13/site-packages/backtrader/strategy.py", line 379, in _next_observers
    observer._next()
    ~~~~~~~~~~~~~~^^
  File "/home/alca/projects/BTQuant/.btq/lib/python3.13/site-packages/backtrader/lineiterator.py", line 280, in _next
    self.next()
    ~~~~~~~~~^^
  File "/home/alca/projects/BTQuant/.btq/lib/python3.13/site-packages/backtrader/observers/broker.py", line 110, in next
    self.lines.value[0] = value = self._owner.broker.getvalue()
                                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/alca/projects/BTQuant/.btq/lib/python3.13/site-packages/backtrader/brokers/ccxtbroker.py", line 89, in getvalue
    return self.store.getvalue(self.currency)
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
  File "/home/alca/projects/BTQuant/.btq/lib/python3.13/site-packages/backtrader/stores/ccxtstore.py", line 113, in retry_method
    time.sleep(self.exchange.rateLimit / 1000)
    ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt

## rework getvalue to use with websocket 


=========================
BTQuant TODO List
=========================

Critical Priority
=============
✅ **Make sure on startup all needed data are fetched** 
   - Specially the DCA logic needs more attention, to calculate everything properly, to avoid buys when its not the right time (wrong entry calculation, ignores calculated dca price, etc.)

High Priority
=============

✅ **Integrate Jackrabbit Relay to its full extend**  
   - Ensure modular REST API support for all future exchanges  
   - Maintain seamless integration with existing BTQ structure  

✅ **Rebuild x125 Cross Liquidation Strategy**  
   - Rework the famous long/short strategy on liquidation events  
   - Optimize execution speed and capital efficiency  

✅ **Expand Out-of-the-Box Strategies**  
   - Develop additional private strategies for future public release  
   - Keep them modular and scalable  

Medium Priority
===============

✅ **Optimize Performance for Large Dataset Processing**  
   - Ensure efficient processing of large historical datasets (Replacing Pandas with Polars -> https://chengzhizhao.com/4-faster-pandas-alternatives-for-data-analysis)  
   - Improve memory handling in Backtrader with MSSQL  

Low Priority
============

✅ **Rework Windows Installer**  
   - Minimal effort unless demand increases  
   - Lack of Windows support as a security-related reason  

Unknowns / Open Ideas
=====================

✅ **???**  
   - Open for suggestions  
