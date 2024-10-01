import backtrader as bt
import joblib

identify = "D[Z2k!~D96#UK0-<mC(~x$a12WvhN.a)Z.H}#.Jch%XV!l<ew^odLGgjJ5xp.C.XgBsB]E,Z59GiruPZjpo23[Q;5};(C}i%IuqZ<J&ZUY9YcF(pH7q~DcX-Q{,HM1QN=zFJx)WxWJp9^gbBLW6+<@_Um[5$(D62lMb2:^5l&n]f3wzmS8N8Um!{~a<FYjGv}/B+>FS/x-=V$[N2oH,V-P~@5yhTwod7bVChB}P1GvhZ>lIc;m<rE5PHreg}kyxHA-Zz19s_Na/!@=}sK3}4XN$xIFBWO,>DZthe=$Ty.pV}K)e3yn0hvExm}-A5v65sGRmx}[%3L[SNC:7ac)[#exU#n[U2#NOU0gX&FsXqE[^sx{aJ=K{apk>)ykRfOQTzn23]*P~rXP]oGiRvwC.~4DCdg3I~#dT6c4/5KpkKNBbv09!%*W:Lh=&>-j>n68_)0*c/)iXF&gF6CnQ/=9{jrvtyzb5O<G}C{O*Ef}(de,pv!CHTeteOD#HME2#06Oy9z7_s0WUZS6Tu*d(Y)AZz;0J{p0wjqK(!k,OMG2p}~rRwPC.c7c47qWVN@N$Aq>pgdHqznmMsXF&J5hEU3c/m9v]<jL4gMODybHxX*B!6GC31a-pUT_xRMp-*cE$:V$)L.B9<]FHT$g!D~#dU%xy]=~Nwz{f+cF>Ug7al0XtLFlkdX9@mcGrt4CLo:L-vN)TS2vBq6~f%*8_-6Q*a;#4NYoMo3C=j+:.Xq02:4^+%<~r7F,RkQS4-w$!]DxQj0C$~UOzrb@^&BL7;df7ZuG%T=:sG^W-2kyPZ#}8]X[b~olK4EhbZ5y5*lg)z7zfNt<[:*]ACvaLFCNt/_/s-{7GsyCit!U=U-y1CH<wZ]H,a&%hI9Ot_}.E>AO8c#VDh{)Gv.hl^,GtTS4S@_zk.nVzVoCd<32+xXfkJo^TPnteBL!:uNnc<@ZS@G7uf4W7h]!*=_21nT{=YV[sRUIlp>4,B<W#hOu$j^>w$=x1_HKZW7:Zu2G>gPZAourS*hwGsg.Bb+[>WiqLX-!*LhD3R%8U=YA$T;LHXM!^cF_$F9hv@&mRd#B(,)f(5!6fj]8zK3A}s[NqSCZ-^U}5elP{Vce*Zm^!G{h2_>pyU@B5au$@Y<Mml.~ceobRG-0rE.ZY)$6#H{]+:Fb(D:D@[<&9ep9-eS9*[FD6zGH-@>D0])MbJ9QfuQu1hFIWZwWq*HjQF=LWN[KWKyM;+Iq/&F]<{~)hY+qe>M=B]Z!Q>Rn9Dg_fN-~s<t}IiN06Gzmr=/8oNA)]4$D_o0ui9*z$Hr.$q-^!k!OpnU<XB^:TF_/.51fcK+D)*D(6qmy!.W.5>Jsh.a!mUr3vz[A2Kb{LZNVpM62fUnWAR}u{kgd8.B6)DV*q{&<rfyi;5~<+4;(mev/XLBHG@OZ%8yCjf8gW[7C-:VxnPOvHe5}ruJkO)n+[beN(iMx8_Dgv.{]^4l_nq,H$W@DktKoq,rkoY#Q+Ma(HJ/JYE_Z}-5&ieb&nwA4-SOL;v=oX:i%r+DX8bwl}L6lVfSypN*b/m4]a)@S%s[{BPxEO*bFZ%kO#]2*rm6tiIA,n9}Hy6Y&s$=MCqv>bxMfNT@jyXgSH!d]pM*opcL~QRMywj3-M68MCUrquHasC=+4GYG(XzIrMLcqH5WEkMtj2jznJIiv1[.iW<yWwF9=/vfP4wT[0LvJ-,lDx!dg&X,kQ6>M^&yWyeC>}[u5Jmbu*QRhtz!i~Iw=T2)JVRM2P*ZyNKd~Ev=AhiK%9K1lD93WRt5~!(Gg10H)LAXs~KbVHaQl_WM03a8M~>mfKkg[=Ao>h1>i>&r%[@KP<u>Ep!YJ+T+&r+1@S(xJmj9vWM[Z5aW89P[Cn7eGmJxa}nPgdo#tefl.-}@rk*:>(^]95KA-awpj0>wU@fNbnP@{q#fA,3BKC~H4#*GxnXl_%Oix1A4g2)H^]UYa#u:ZzcnN,h47LM#gl6;!o-7Lv/A-1tgvT;Y#i1o3h:vH<KM$Mdnp_UpE-{Mxv2{"
# jrr_webhook_url = "http://65.21.116.91:80"
jrr_webhook_url = "http://127.0.0.1:80"
jrr_order_history = ""
jrr_order_history_sub1 = ""



bsc_privaccount1 = ""
bsc_privaccountaddress = ""

import importlib
import sys
import os
import pandas as pd
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/mssql/MsSQL/build/lib.linux-x86_64-cpython-312/fast_mssql.cpython-312-x86_64-linux-gnu.so'))
spec = importlib.util.spec_from_file_location("fast_mssql", module_path)
fast_mssql = importlib.util.module_from_spec(spec)
sys.modules["fast_mssql"] = fast_mssql
spec.loader.exec_module(fast_mssql)

# SQL Server connection details
server = 'localhost'
database = 'BacktraderData'
username = 'sa'
password = ''
driver = '{ODBC Driver 18 for SQL Server}'  # Adjust the driver version if necessary

connection_string = (f'DRIVER={driver};'
                     f'SERVER={server};'
                     f'DATABASE={database};'
                     f'UID={username};'
                     f'PWD={password};'
                     f'TrustServerCertificate=yes;')

class MSSQLData(bt.feeds.PandasData):
  @classmethod
  def get_data_from_db(cls, connection_string, coin, timeframe, start_date, end_date):
      start_timestamp = int(start_date.timestamp() * 1000)
      end_timestamp = int(end_date.timestamp() * 1000)

      query = f"""
      SELECT 
          TimestampStart, 
          [Open], 
          [High], 
          [Low], 
          [Close], 
          Volume
      FROM {coin}
      WHERE Timeframe = '{timeframe}'
      AND TimestampStart BETWEEN {start_timestamp} AND {end_timestamp}
      ORDER BY TimestampStart
      OPTION(USE HINT('ENABLE_PARALLEL_PLAN_PREFERENCE'))
      """
      
      data = fast_mssql.fetch_data_from_db(connection_string, query)
      
      df = pd.DataFrame(data, columns=['TimestampStart', 'Open', 'High', 'Low', 'Close', 'Volume'])
      df['TimestampStart'] = pd.to_datetime(df['TimestampStart'].astype(int), unit='ms')
      numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
      df[numeric_columns] = df[numeric_columns].astype(float)
      df.set_index('TimestampStart', inplace=True)
      return df

  @classmethod
  def get_all_pairs(cls, connection_string):
      query = "SELECT DISTINCT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
      data = fast_mssql.fetch_data_from_db(connection_string, query)
      return [row[0] for row in data]


def ptu():
    art = [
        r'''
               ...                            
             ;::::;                           
           ;::::; :;                          
         ;:::::'   :;                         
        ;:::::;     ;.                        
       ,:::::'       ;           OOa\         
       ::::::;       ;          OOOOL\        
       ;:::::;       ;         OOOOOOcO       
      ,;::::::;     ;'         / OOOOOaO      
    ;:::::::::`. ,,,;.        /  / DOOOWOO    
  .';:::::::::::::::::;,     /  /     OOAOO   
 ,::::::;::::::;;;;::::;,   /  /        OSOO  
;`::::::`'::::::;;;::::: ,#/  /          DHOO 
:`:::::::`;::::::;;::: ;::#  /            DEOO
::`:::::::`;:::::::: ;::::# /              ORO
`:`:::::::`;:::::: ;::::::#/               DOE
 :::`:::::::`;; ;:::::::::##                OO
 ::::`:::::::`;::::::::;:::#                OO
 `:::::`::::::::::::;'`:;::#                O 
  `:::::`::::::::;' /  / `:#                  
   ::::::`:::::;'  /  /   `#              
'''

    ]
    for line in art:
        print(line)

