import backtrader as bt

class T(bt.Strategy):
    params = (("myrefit", 50),)
    def __init__(self):
        super().__init__()
        out = []
        try:
            out.append("self.myrefit   -> %r" % (self.myrefit,))
        except AttributeError as e:
            out.append("self.myrefit   -> AttributeError: %s" % e)
        out.append("self.p.myrefit -> %r" % (self.p.myrefit,))
        with open("/home/alca/projects/PubBTQuant/.daedalus/param_attr.txt", "w") as f:
            f.write("\n".join(out) + "\n")

feed = bt.feeds.PseudoData(bt.TimeFrame.Days, n=5)
c = bt.Cerebro(stdstats=False, runonce=False)
c.adddata(feed)
c.addstrategy(T, myrefit=7)
c.run()
