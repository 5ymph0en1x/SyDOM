# SyDOM
Sophisticated Scalper Bot for BitMEX using orders imbalance analysis

------------------------------------

Configuration

Just replace the key and secret fields with yours and adjust the number of contracts traded in sydom.py

Run `pip install -r requirements.txt`

------------------------------------

Execution

`python sydom.py`

During the first run, the script will automatically generate a model (if missing) based on the 3 previous days of price action. This model is retrained every day at 6:00 AM UTC. Basically, you don't have anything to do once the bot is launched...

------------------------------------

Donations to allow further developments

BTC: 3BMEXbS4Neu5KwsiATuZVowmwYD3UPMuxo
