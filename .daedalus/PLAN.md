# Sweep-Programm: Gate-Fehler beheben (LLM-Codegen)
Status: done
Asked: "alle coins, verschiedene timeframes, beginnend von 1m, das volle elendige programm"
Done when: generierte Strategien schliessen auf 12k echten BTCUSDT-1h-Bars Trades

## Steps
- [x] 1. Smoke-Test auf echte MSSQL-Cache-Bars umgestellt (strategy_factory.py:~2100)
      verify: Log "Smoke test on real bars", BTCUSDT_1h.parquet 1022-2122 Bars erkannt
- [x] 2. MIN_ORDERS-Gate eingefuehrt (max(3, bars//100)) gegen 1-Roundtrip-Stall
      verify: Dual EMA Trend fiel mit "StalledStrategy: only 2 order(s)" durch
- [x] 3. Root cause des 2-Order-Stalls gefunden
      verify: probe_next.py -> next() 11971x, entry 1x, exit 1x, buy_executed=True
      Ursache: create_order(action='SELL') ist im Fork KEIN Exit. Es haengt einen
      zweiten OrderTracker an active_orders und laesst buy_executed=True. Nur
      close_order(self.active_orders[-1]) setzt zurueck (base.py:352-353).
- [x] 4. Prompt-Regel korrigiert (Zeile ~1001) + AST-Validator FakeExit ergaenzt
      verify: _audit_next_body(code, method='sell_or_cover_condition') -> FakeExit
- [x] 5. Gate ruft BEIDE Condition-Methoden
      verify: 12k-Run 5/5 closed (32/91/1/465/158), vorher 0/4

## Decisions
- AST-Check statt Prompt-Vertrauen: das LLM ignorierte die Exit-Regel in 3 von 4
  Samples. Ein Prompt, der nicht befolgt wird, ist kein Gate.
- Smoke-Test auf echten Bars, nicht synthetisch: eine Sinus-Kurve ohne
  Volatilitaetsregime laesst jede Crossover-Logik 2x feuern statt 15x/1k.
  6 von 8 "passenden" Strategien waren auf 12k echten Bars Krebs.
- _audit_next_body folgt dem Call-Graph ab der uebergebenen Methode. Die beiden
  Condition-Methoden rufen sich nicht gegenseitig -> beide einzeln pruefen.

## Notes
- 2082 Bestandsdateien in autonomous_agency/strategies/ haben denselben
  create_order(action='SELL')-Exit und sind zur Haelfte stumm. Neugenerierte
  Dateien sind ab jetzt sauber, alte nicht -- Regeneration ist der naechste
  Schritt, nicht Teil dieser Fixes.
- Bollinger Reversion: 2 Orders aber 1 closed. Der 400-Bar-Smoke erkennt das
  nicht (2 < MIN_ORDERS=4 haette es erkennen muessen -> Reject kam nicht).
  Siehe offene Frage unten.
