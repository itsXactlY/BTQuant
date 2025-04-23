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
