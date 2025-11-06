#!/usr/bin/env python3
from __future__ import annotations
#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Copyright (C) 2015-2020 Daniel Rodriguez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


__version__ = '1.10.0'

__btversion__ = tuple(int(x) for x in __version__.split('.'))

import io
import os
import random
import sys
import threading
import time
from typing import Callable, Any

try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
    _COLOR = True
except ImportError:
    _COLOR = False
    class _DummyFore:
        RED = GREEN = YELLOW = CYAN = MAGENTA = BLUE = WHITE = ""
    class _DummyStyle:
        BRIGHT = NORMAL = RESET_ALL = ""
    Fore = _DummyFore()
    Style = _DummyStyle()

ART = r"""
               ...                            
             ;::::;                           
           ;::::; :;                          
         ;:::::'   :;                         
        ;:::::;     ;.                        
       ,:::::'       ;           OOa         
       ::::::;       ;          OOOOL        
       ;:::::;       ;         OOOOOOcO       
      ,;::::::;     ;'         / OOOOOaO      
    ;:::::::::`. ,,,;.        /  / DOOOWOO    
  .';:::::::::::::::::;,     /  /     OOAOO   
 ,::::::;::::::;;;;::::;,   /  /        OSOO  
;`::::::`'::::::;;;::::: ,#/  /          DHOO 
:`:::::::`;::::::;;::: ;::#  /            DEOO
::`:::::::`;:::::::: ;::::# /              ORO
`:`:::::::`;:::::: ;::::::#/               DOE
 :::`:::::::`;; ;:::::::::##                OO
 ::::`:::::::`;::::::::;:::#                OO
 `:::::`::::::::::::;'`:;::#                O 
  `:::::`::::::::;' /  / `:#                  
   ::::::`:::::;'  /  /   `#              
"""

MESSAGES = [
    "Stay SAFU: small size, tight stops, big brain.",
    "Mindset > entry. Even a good setup can’t fix tilt.",
    "Don’t ape. If it looks like free money, you’re the product.",
    "Printer goes brrrr only for those who survive sideways.",
    "Cut losers fast. DCA is for plans, not for panic.",
    "If you can’t sleep with the position, the position owns you.",
    "No signal today? Good. Flat is also a position, degen.",
    "Your edge is risk management, not the magic indicator pack v42.",
    "Stop backtesting until it’s perfect. Trade something that’s merely not trash.",
    "If you need hope, the trade is already dead.",
    "Leverage doesn’t create alpha, it magnifies stupidity.",
    "Fees are the house edge. You are not the house.",
    "You’re not marrying the coin, you’re speed-dating volatility.",
    "FOMO is just you donating liquidity with emotions attached.",
    "Green day? Log results, walk away. Don’t speedrun giving it back.",
    "Red day? No revenge trading. Close terminal, touch grass, pet dog.",
    "HODL is a strategy. It’s just not always *your* strategy.",
    "If your plan fits in a tweet, it’s probably not a plan.",
    "More monitors won’t fix bad discipline.",
    "Survive first, flex later. Screenshots are optional, solvency is not.",
    "Markets don’t punish stupidity, they patiently invoice it with interest.",
    "You didn’t lose money, you paid tuition. The market just doesn’t issue diplomas.",
    "Every blown account started with: ‘It’s just this one time, I’ll size bigger.’",
    "The chart doesn’t care about your bills, your dreams, or your backstory.",
    "Hope is the most expensive position you can hold.",
    "The moment you pray for a candle, you’ve already lost the trade.",
    "You’re not early, you’re exit liquidity with extra steps.",
    "That ‘sure thing’ setup? The market saw your confidence and adjusted.",
    "There is no rock bottom, only whatever your broker allows as minimum margin.",
    "The market won’t ruin your life. You’ll do that yourself by not having a stop.",
    "Algorithms don’t hate you, they just farm your impulses efficiently.",
    "You don’t blow up in one trade; you blow up in a thousand tiny rationalizations.",
    "Discipline is what you think you have until the third red candle.",
    "Your edge died the moment you started needing the money.",
    "You either learn to close bad trades, or you learn to explain them to debt collectors.",
    "Size like you’re wrong, not like the universe owes you a reversal.",
    "Risk limits are the thin line between ‘temporary drawdown’ and ‘origin story of a cautionary tale.’",
    "The chart is not attacking you. It’s just revealing who you really are under pressure.",
    "Losing trades don’t make you a loser. Refusing to adapt does.",
    "You’re not fighting the market. You’re fighting your own need to be right.",
    "You didn’t get rekt, you carefully handcrafted a donation to people who can read a risk:reward ratio.",
    "That wasn’t a liquidation, that was the market forcibly uninstalling your ego.",
    "You didn’t blow the account, you slowly converted intelligence into candles and cope.",
    "This isn’t ‘unlucky price action’, this is years of bad habits presenting the invoice.",
    "You weren’t hunted; you publicly broadcast your stop to the whole orderbook like an idiot with a megaphone.",
    "That green candle you chased? It was just the hearse arriving for your collateral.",
    "You didn’t ‘average in’, you lovingly marinated your account before serving it to market makers.",
    "Your PnL graph looks like a crime scene report nobody wants to file.",
    "You called it ‘high conviction’; the market called it ‘obvious leverage with extra seasoning’.",
    "You didn’t lose discipline, you never had any—just hope wearing a headset.",
    "You didn’t get trapped in a range, you voluntarily built yourself a coffin out of limit orders.",
    "Every time you moved the stop, you weren’t managing risk, you were extending your own execution date.",
    "You didn’t miss the top; you insisted on buying the last ticket on the Titanic at a markup.",
    "Your journal doesn’t show a learning curve, it shows a body count.",
    "You didn’t suffer a black swan, you walked into regular volatility with a clown-sized position.",
    "It wasn’t a ‘flash crash’, it was your IQ syncing with your balance in real time.",
    "You weren’t front-running smart money; you were volunteering as organic fertilizer for their entries.",
    "That ‘one more add’ wasn’t conviction, it was your brain putting the noose on your margin.",
    "You didn’t get liquidated in a freak move, you spent weeks arranging the perfect angle of impact.",
    "Your strategy isn’t systematic, it’s just improvisational self-harm for your net worth.",
    "The market didn’t humble you, it just scraped off whatever delusion was still solvent.",
    "You’re not early to the trend, you’re late to the funeral and still trying to sell flowers.",
    "You didn’t ‘trust the process’; you abandoned it the second a red candle hurt your feelings.",
    "That cascading loss wasn’t bad luck, it was you stacking bad decisions like kindling.",
    "You don’t trade a system, you trade a mood—and your equity curve shows the mood swings.",
    "The market didn’t take your money, you escorted it to the exit like a loyal but clueless butler.",
    "You didn’t over-optimize, you surgically removed any chance the strategy had to survive live conditions.",
    "Your ‘long-term thesis’ is just cope you invented after refusing to hit close.",
    "You aren’t stuck in a drawdown; you’re stuck in denial with a broker connection.",
    "You didn’t lose the game; you never learned the rules, just mashed the buy button until reality answered.",    
]


WELCOME_LINE = (
    "Welcome to the BTQuant feeding trough, son."
)

def _clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _type_line(text: str, delay: float = 0.01, out=None) -> None:
    if out is None:
        out = sys.stdout
    for ch in text:
        print(ch, end="", flush=True, file=out)
        time.sleep(delay)
    print(file=out)


def ptu(
    cycles: int = 1,
    art_delay: float = 0.02,
    msg_delay: float = 0.015,
    out=None,
) -> None:
    if out is None:
        out = sys.stdout

    for _ in range(cycles):
        for line in ART.splitlines():
            if line.strip():
                if _COLOR:
                    print(Fore.MAGENTA + Style.BRIGHT + line + Style.RESET_ALL, file=out)
                else:
                    print(line, file=out)
            else:
                print(line, file=out)
            out.flush()
            time.sleep(art_delay)

        print(file=out)
        msg = random.choice(MESSAGES)
        for ch in msg:
            if _COLOR:
                print(Fore.CYAN + ch + Style.RESET_ALL, end="", flush=True, file=out)
            else:
                print(ch, end="", flush=True, file=out)
            time.sleep(msg_delay)
        print("\n", file=out)
        out.flush()

class PTUStdout(io.TextIOBase):
    def __init__(self, orig):
        self.orig = orig
        self.buffer = []
        self.capturing = True

    def write(self, s: str) -> int:
        if not s:
            return 0
        if self.capturing:
            self.buffer.append(s)
        else:
            self.orig.write(s)
        return len(s)

    def flush(self) -> None:
        self.orig.flush()

    def end_phase(self) -> None:
        if self.capturing:
            self.capturing = False
            if self.buffer:
                self.orig.write("".join(self.buffer))
                self.buffer = []
            self.orig.flush()

def run_with_ptu_front(
    target: Callable[..., Any],
    *args,
    duration: float = 10.0,
    **kwargs,
) -> Any:
    orig_stdout = sys.stdout
    wrapper = PTUStdout(orig_stdout)
    sys.stdout = wrapper

    stop_event = threading.Event()

    def anim():
        _clear()
        if _COLOR:
            print(Fore.RED + Style.BRIGHT + WELCOME_LINE + Style.RESET_ALL, file=orig_stdout)
        else:
            print(WELCOME_LINE, file=orig_stdout)
        orig_stdout.flush()
        time.sleep(1.5)

        start = time.time()
        while not stop_event.is_set() and (time.time() - start) < duration:
            ptu(cycles=1, art_delay=0.01, msg_delay=0.01, out=orig_stdout)
        _clear()
        wrapper.end_phase()

    anim_thread = threading.Thread(target=anim, daemon=True)
    anim_thread.start()

    try:
        result = target(*args, **kwargs)
    finally:
        stop_event.set()
        anim_thread.join(timeout=1.0)
        wrapper.end_phase()
        sys.stdout = orig_stdout

    return result
