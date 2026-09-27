import sys
sys.path.insert(0, '/home/alca/projects/PubBTQuant')
import inspect
import re

import backtrader as bt

k = bt.ind.DirectionalMovementIndex
print('DirectionalMovementIndex params:', [f'{x[0]}={x[1]}' for x in k.params][:10])
print('  lines decls in source:', re.findall(r'lines = \(([^)]*)\)', inspect.getsource(k)))
print('ADX params:', [f'{x[0]}={x[1]}' for x in bt.ind.ADX.params][:8])
print('PSAR params:', [f'{x[0]}={x[1]}' for x in bt.ind.PSAR.params][:10])
for n in ('DEMA', 'TEMA', 'TRIX', 'Ichimoku', 'PivotPoint', 'CrossOver',
          'Highest', 'Lowest', 'Vortex', 'ChandelierExit', 'SMA', 'EMA',
          'ExponentialMovingAverage', 'WeightedMovingAverage', 'SMMA'):
    k = getattr(bt.ind, n, None)
    if k is None:
        print(f'  {n:28s} MISSING')
        continue
    try:
        ps = [f'{x[0]}={x[1]}' for x in k.params][:6]
    except Exception as e:  # noqa: BLE001
        ps = f'({e})'
    print(f'  {n:28s} {ps}')
print('Highest.__init__ args:')
try:
    print(' ', inspect.signature(bt.ind.Highest.__init__))
except Exception as e:  # noqa: BLE001
    print(' ', e)
