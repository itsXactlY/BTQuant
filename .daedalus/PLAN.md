# Sweep-Programm: Gate-Fehler beheben (LLM-Codegen)
Status: done
Asked: "alle coins, verschiedene timeframes, beginnend von 1m, das volle elendige programm"
Done when: generierte Strategien schliessen auf 12k echten BTCUSDT-1h-Bars Trades

## Steps
- [x] 1. Smoke-Test auf echte MSSQL-Cache-Bars umgestellt
      verify: BTCUSDT_1h.parquet erkannt, 6/8 "passende" waren auf 12k Krebs
- [x] 2. MIN_ORDERS-Gate (max(3, bars//100)) gegen 1-Roundtrip-Stall
      verify: "StalledStrategy: only 2 order(s) in 400 bars (need 4)"
- [x] 3. Root cause #1: create_order(action='SELL') ist im Fork kein Exit
      verify: probe_next.py -> next() 11971x, entry 1x, buy_executed=True
- [x] 4. Prompt-Regel korrigiert + AST-Validator FakeExit
- [x] 5. Gate ruft BEIDE Condition-Methoden (Call-Graph erreicht den Exit nicht)
- [x] 6. Root cause #2: StalePositionFlag, self.in_position nie zurueckgesetzt
      verify: probe_flag.py -> 371 Entry-Signale, 1 Order, in_position=True,
      buy_executed=False, active_orders=0 (flat und bereit, re-enter nie)
- [x] 7. StalePositionFlag-Validator, __init__ von der Reset-Seite ausgenommen
      verify: 5/5 korrekt gegen Ground Truth des 12k-Laufs, 0 Mismatches
- [x] 8. End-to-End: 4/4 akzeptiert = 4/4 mit geschlossenen Trades
      64/91/464/687 closed; 2 Rejects (NoOrderPlaced, ATR IndexError)

## Decisions
- AST-Check statt Prompt-Vertrauen: das LLM ignorierte die Exit-Regel in 3 von 4
  Samples. Ein Prompt, der nicht befolgt wird, ist kein Gate.
- Smoke-Test auf echten Bars: eine Sinus-Kurve hat keine Volatilitaetsregime,
  jede Crossover-Logik feuert 2x statt 15x/1k.
- _audit_next_body folgt dem Call-Graph ab der uebergebenen Methode. Die beiden
  Condition-Methoden rufen sich nicht gegenseitig -> beide einzeln pruefen.
- Der Stale-Flag-Check ist bewusst KLASSENWEIT statt reachable, und schliesst
  __init__ aus. reachable sah den Exit nie; __init__'s `= False` liess die
  Regel genau die Strategie passieren, die sie fangen soll.

## Notes
- 2082 Bestandsdateien in autonomous_agency/strategies/ haben dieselben zwei
  Bugs und sind zur Haelfte stumm. Neugeneriert ist ab jetzt sauber, alte nicht.
  Regeneration ist der naechste Schritt, nicht Teil dieser Fixes.
- Die Validatoren sind gegen 5 Ground-Truth-Dateien geprueft, nicht gegen eine
  breite Stichprobe. Vor dem Sweep sollte ein groesserer Batch laufen.
