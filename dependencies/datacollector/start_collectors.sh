cat > supervisord.conf << 'EOF'
[unix_http_server]
file=supervisor.sock

[supervisord]
logfile=supervisord.log
pidfile=supervisord.pid
childlogdir=logs

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix://supervisor.sock

[program:collector_main]
command=%(here)s/market_data_collector config_main.json
directory=%(here)s
autostart=true
autorestart=true
stdout_logfile=logs/collector_main.log
stderr_logfile=logs/collector_main_err.log

[program:collector_bybit_01]
command=%(here)s/market_data_collector config_bybit_01.json
directory=%(here)s
autostart=true
autorestart=true
stdout_logfile=logs/collector_bybit_01.log
stderr_logfile=logs/collector_bybit_01_err.log

[program:collector_bybit_02]
command=%(here)s/market_data_collector config_bybit_02.json
directory=%(here)s
autostart=true
autorestart=true
stdout_logfile=logs/collector_bybit_02.log
stderr_logfile=logs/collector_bybit_02_err.log

[program:collector_bybit_03]
command=%(here)s/market_data_collector config_bybit_03.json
directory=%(here)s
autostart=true
autorestart=true
stdout_logfile=logs/collector_bybit_03.log
stderr_logfile=logs/collector_bybit_03_err.log

[program:collector_bybit_04]
command=%(here)s/market_data_collector config_bybit_04.json
directory=%(here)s
autostart=true
autorestart=true
stdout_logfile=logs/collector_bybit_04.log
stderr_logfile=logs/collector_bybit_04_err.log

[program:collector_bybit_05]
command=%(here)s/market_data_collector config_bybit_05.json
directory=%(here)s
autostart=true
autorestart=true
stdout_logfile=logs/collector_bybit_05.log
stderr_logfile=logs/collector_bybit_05_err.log

[group:collectors]
programs=collector_main,collector_bybit_01,collector_bybit_02,collector_bybit_03,collector_bybit_04,collector_bybit_05
EOF