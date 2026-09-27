# PubBTQuant — Gate-Fixes 2026-09-27 (nicht in mazemaker, Pod war down)

## Bug 1: FakeExit — create_order(action='SELL') ist kein Exit
Im Fork (`~/.btq/lib/python3.14/site-packages/backtrader/strategies/base.py`)
hängt `create_order(action='SELL')` einen zweiten OrderTracker an `active_orders`
und lässt `buy_executed=True`. Nur `self.close_order(self.active_orders[-1])`
setzt zurück (base.py:352-353 → reset_position_state).
Messung 12k BTCUSDT 1h: create_order('SELL') → 2 Orders / 0 closed;
close_order → 212 Orders / 106 closed. Instrument: `.daedalus/probe_next.py`
(next() 11971×, entry 1×, exit 1×, buy_executed=True).

## Bug 2: StalePositionFlag — in_position wird nie zurückgesetzt
Entry setzt `self.in_position = True`, Exit-Zweig lässt es True.
Messung 12k BTCUSDT 1h, Bollinger Reversion: **371 Entry-Signale, 1 Order**.
Endzustand buy_executed=False, active_orders=0 — flat und bereit, kein Re-Entry.
Instrument: `.daedalus/probe_flag.py`.
Der Base kann das nicht retten: er kennt das Flag nicht.

## Zwei Fallstricke in der Validator-Regel (beide zuerst falsch)
1. Der Stale-Check ist **klassenweit**, nicht über den `reachable`-Call-Graph.
   Entry- und Exit-Methode rufen sich nicht gegenseitig → der Entry-Audit sah
   den `= False` im Exit nie und meldete jede Strategie mit Flag als stale
   (Falsch-Positiv auf 4 gesunden Strategien).
2. `__init__` ist von der **Reset-Seite ausgeschlossen**. Sonst zählt
   `self.in_position = False` im Konstruktor als Rücksetzen und die Regel lässt
   genau die Strategie durch, die sie fangen soll (Falsch-Negativ auf Bollinger).

## Gate läuft über BEIDE Condition-Methoden
`_audit_next_body` folgt dem Call-Graph ab der übergebenen Methode. Da
Entry/Exit sich nicht gegenseitig aufrufen, muss der Gate
`buy_or_short_condition` UND `sell_or_cover_condition` einzeln prüfen.

## Smoke-Test auf echten Bars
Eine Sinuskurve hat keine Volatilitätsregime → Crossover feuert 2× statt 15×/1k.
6 von 8 "passenden" Strategien waren auf 12k echten Bars Krebs. Test läuft jetzt
auf `.btq_cache/mssql/BTCUSDT_1h.parquet`. Zusätzlich MIN_ORDERS = max(3, bars//100).

## Lessons
- Ein Prompt, dem das LLM nicht folgt (3 von 4 Samples ignorierten die
  Exit-Regel), ist kein Gate. Systematische LLM-Fehler gehören in den Validator.
- Ein AST-Validator, der Entry und Exit vergleicht, darf nicht dem Call-Graph
  der einzelnen Methode folgen. Und `__init__`-Initialisierung ist kein
  Zustandsrückbau. Beide Fehler fielen nur durch Abgleich gegen echte
  Backtest-Zahlen auf.

## Stand
Commit e5abb0c4 (Bug 1) + Folge-Commit (Bug 2). 4/4 akzeptiert = 4/4 closed
(64/91/464/687 auf 12k). Validatoren gegen 5 Ground-Truth-Dateien geprüft,
nicht gegen breite Stichprobe — vor dem Sweep größeren Batch laufen lassen.

## Offen
2082 Bestandsdateien in `autonomous_agency/strategies/` haben beide Bugs und
sind zur Hälfte stumm. Neugeneriert ist sauber, alte nicht. Regeneration als
nächster Schritt, nicht Teil dieser Fixes.
