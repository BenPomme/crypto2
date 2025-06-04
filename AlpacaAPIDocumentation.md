Alpaca API – Comprehensive Guide to Trading & Market Data
Overview of Alpaca API
Alpaca is an API-first brokerage platform that supports trading in US stocks and cryptocurrencies via modern REST and WebSocket APIs
docs.alpaca.markets
. With Alpaca’s API, you can manage your account, place trades (stocks or crypto), access real-time and historical market data, and even stream live updates. Alpaca offers both live trading and paper trading (simulated trading) environments, allowing you to test strategies risk-free before going live. All functionality – from account management to market data – is accessible through API endpoints or convenient SDKs, making it easy to build algorithmic trading strategies or full-fledged trading applications.
Getting API Access and Authentication
To use Alpaca’s API, you must first create an account on Alpaca and obtain API keys (a key ID and secret). These API keys authenticate your requests. Include them in the HTTP headers APCA-API-KEY-ID and APCA-API-SECRET-KEY for every request
docs.alpaca.markets
. For example, using curl you might include -H "APCA-API-KEY-ID: YOUR_KEY_ID" -H "APCA-API-SECRET-KEY: YOUR_SECRET_KEY" in the request. Alpaca provides two environments: live trading and paper trading. Each environment has its own base URL and a distinct set of API keys. To use paper trading (for simulated trades), set the API endpoint to https://paper-api.alpaca.markets and use your paper API keys
docs.alpaca.markets
. For live trading, use https://api.alpaca.markets with your live keys. Always keep your secret key safe and do not expose it in client-side code. API Base URLs:
Live Trading API: https://api.alpaca.markets
Paper Trading API: https://paper-api.alpaca.markets
All endpoints described below should be prefixed with the appropriate base URL. Both HTTP GET (for retrieving data) and POST/DELETE/PATCH (for creating or modifying data) are used, and all responses are in JSON. Time fields are typically in ISO8601 format (RFC 3339 timestamps, usually in UTC). Ensure your system clock is accurate, as certain endpoints (e.g., placing orders) may be time-sensitive.
Alpaca Trading API Endpoints and Functionality
Alpaca’s trading API covers everything from account information and orders to portfolio data and more. Below we detail each major category of functionality, including the available REST endpoints, parameters, and example usage.
Account Information and Management
Account Data: The /v2/account endpoint (GET) returns information about your trading account. This includes your cash balance, buying power, portfolio value, and status flags (e.g., whether trading is blocked or if the account is flagged as a pattern day trader)
docs.alpaca.markets
. For instance, GET /v2/account will show if your account is currently restricted from trading (trading_blocked flag), how much buying power (cash or margin) is available, and other details like equity and margin status. This endpoint is read-only; it’s a good practice to check your account status and buying power before placing orders. Below is an example in Python using Alpaca’s SDK:
python
Copy
Edit
account = trading_client.get_account()
if account.trading_blocked:
    print("Account is restricted from trading.")
print(f"Equity: ${account.equity}, Buying Power: ${account.buying_power}")
Account Configuration: Alpaca allows certain account settings to be configured via API. The /v2/account/configurations endpoints (GET/PATCH) let you view and update settings such as:
dtbp_check: Day-trade buying power check mode (entry, exit, or both) – controls whether the system checks your day-trading buying power on order entry, exit, or both
medium.com
.
trade_confirm_email: Enable/disable trade confirmation emails (all or none)
medium.com
.
suspend_trade: A boolean to immediately halt all trading on your account (acts like an emergency kill-switch you can toggle via API)
medium.com
.
no_shorting: A boolean to disable short selling on your account (if true, any attempts to short sell will be rejected)
medium.com
.
For example, to disable shorting and email confirmations, you would send a PATCH to /v2/account/configurations with {"no_shorting": true, "trade_confirm_email": "none"}. These settings give you control over risk and preferences directly through the API
medium.com
medium.com
. Note that margin trading (and thus short selling) is only applicable to equities; crypto trades are always fully funded (no margin) and are not subject to pattern-day-trader (PDT) rules
docs.alpaca.markets
. Account Activities & History: You can retrieve a log of account activities (trades, cash transfers, dividends, etc.) via the /v2/account/activities endpoint. This can be filtered by activity type. For example, setting activity_types=FILL returns executions of orders (fills), whereas TRANS would return cash transactions like deposits/withdrawals
docs.alpaca.markets
. Each activity record includes details such as the activity type, date/time, symbol, quantity, price, etc. This is useful for auditing and record-keeping. In addition, the /v2/account/portfolio/history endpoint provides historical performance of your portfolio – typically an array of timestamps and your account’s equity value at those times (e.g., daily equity for the last 90 days, or intraday if specified). You can use this to chart your portfolio value over time or compute returns. Clock & Calendar: Alpaca offers endpoints to query the market schedule. The /v2/clock (GET) endpoint returns the current market time and whether the market is open, along with the time to next open or close. This is handy to check if it’s within trading hours. The /v2/calendar (GET) endpoint returns the market calendar (trading days) including scheduled market open and close times, holidays, etc. You can query, for example, the next week of trading days to know on which days the market will be closed or have shortened sessions. Always consider using these endpoints or their data to avoid placing orders when the market is closed or to schedule your trading strategies appropriately. Corporate Actions: Alpaca automatically handles mandatory corporate actions (like stock splits, mergers, dividends) on your account positions. For example, if a stock in your portfolio splits 2-for-1, your position quantity and average entry price are adjusted accordingly by Alpaca. If you need details about corporate actions, Alpaca provides a Corporate Actions API (e.g., /v2/corporate_actions/announcements) where you can retrieve announcements of upcoming actions (such as symbol changes, mergers, splits). This is more often used by broker applications, but it’s available if you want to programmatically know about events affecting stocks you trade. Voluntary corporate actions (which require shareholder decision) typically are not applicable via API for individual accounts – you would be notified outside the API if relevant.
Asset Listings and Market Assets
Alpaca maintains a master list of all assets (stocks and crypto) available for trading or data. The /v2/assets (GET) endpoint returns an array of assets, each with details like: symbol, full name, exchange (for stocks), asset_class (us_equity or crypto), whether the asset is tradable and marginable, and other flags. By default, this endpoint returns all U.S. equities. You can filter the results using query parameters or in the SDK. For example, you might request only active tradable assets on NASDAQ. In the JavaScript SDK:
js
Copy
Edit
alpaca.getAssets({ status: "active" }).then(assets => {
  const nasdaqAssets = assets.filter(asset => asset.exchange == "NASDAQ");
  console.log(nasdaqAssets);
});
In Python, using the newer SDK:
python
Copy
Edit
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

req = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
assets = trading_client.get_all_assets(req)
# Filter or search within assets list as needed
This would give you all active U.S. equity assets
docs.alpaca.markets
docs.alpaca.markets
. If you want to get crypto assets, you can specify asset_class=AssetClass.CRYPTO in the request to list supported cryptocurrencies. Alpaca currently supports dozens of crypto trading pairs (over 20+ unique crypto assets across ~50 pairs)
docs.alpaca.markets
. Crypto asset symbols are represented as BASE/QUOTE pairs (e.g. "BTC/USD" for Bitcoin trading against USD). Each asset object will indicate if it is tradable on Alpaca. Use the assets endpoint to validate symbols or to discover available instruments.
Placing and Managing Orders
Order Placement: Orders are created by sending a POST to the /v2/orders endpoint. At minimum, an order requires: a symbol (e.g., "AAPL" or "BTC/USD"), a qty (quantity of shares or fractional shares, or in the case of crypto, quantity of coins), or alternatively a notional dollar amount, the side (buy or sell), the type of order, and a time_in_force. The API supports standard order types: market, limit, stop, stop_limit, and trailing_stop, as well as advanced orders like bracket orders (one-cancels-other bracket) and more. – Market Orders: A market order executes immediately at the best available price. You only need to specify type: "market" and typically a time_in_force (e.g. "day" for day order). For example, to buy 1 share of Apple at market price:
js
Copy
Edit
alpaca.createOrder({
  symbol: "AAPL",
  qty: 1,
  side: "buy",
  type: "market",
  time_in_force: "day"
});
In Python, using the new SDK, this would be done by constructing a MarketOrderRequest and submitting it:
python
Copy
Edit
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

order_data = MarketOrderRequest(symbol="SPY", qty=0.023, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
order = trading_client.submit_order(order_data=order_data)
This example buys 0.023 shares of SPY (i.e. a fractional share) as a Day order
docs.alpaca.markets
docs.alpaca.markets
. Fractional trading is supported on Alpaca for most equities, allowing very small share amounts (as little as $1 worth). Here, 0.023 shares is an example of a fractional quantity. – Limit Orders: A limit order executes only at your specified price or better. In addition to the above fields, specify type: "limit" and a limit_price. For example, to sell 1 share of AMD with a limit price of $20.50 that will only execute at market open, you could send:
js
Copy
Edit
alpaca.createOrder({
  symbol: "AMD",
  qty: 1,
  side: "sell",
  type: "limit",
  time_in_force: "opg",        // opg = execute at market open
  limit_price: 20.50
});
You can also place fractional limit orders by specifying a fractional qty or using notional. For instance, in Python if you want to sell roughly $4000 worth of Bitcoin at a limit price of $17,000, you could specify notional=4000 instead of qty
docs.alpaca.markets
. The API will calculate the quantity (approximately 0.2353 BTC in this case) based on the price. Note: In crypto trading, notional order sizing is common, and Alpaca supports it for both stocks and crypto. However, notional sizing is not allowed in extended-hours trading
docs.alpaca.markets
docs.alpaca.markets
 (extended hours require explicit share/coin quantity). – Stop Orders: A stop order triggers a market order when a certain stop price is reached. You place it with type: "stop" and a stop_price. For example, a sell stop order might be used as a stop-loss: e.g., “sell 10 shares of XYZ if price falls to $50”. Note that a plain stop order becomes a market order when triggered. If you need a stop that becomes a limit order (to avoid selling far below a trigger price), use a stop-limit order, providing both stop_price and limit_price with type: "stop_limit". – Trailing Stop Orders: A trailing stop automatically adjusts the stop price as the market moves in your favor. You specify a trail_price (an absolute dollar amount) or trail_percent (percentage) instead of a fixed stop price. For a sell trailing stop, the stop price will “trail” the stock’s high-water mark by the given amount/percent
docs.alpaca.markets
docs.alpaca.markets
. For example, a trailing stop sell with trail_percent: 1.0 will set the stop price at 1% below the stock’s peak price while the order is active (continuously updating). If the price reverses by 1% from its peak, a market sell is triggered. In JSON this would be:
js
Copy
Edit
alpaca.createOrder({
  symbol: "AAPL",
  qty: 1,
  side: "sell",
  type: "trailing_stop",
  trail_percent: 1.0,    // or use trail_price: 1.00 for $1 trailing
  time_in_force: "day"
});
Trailing stops are useful for locking in profits with a moving stop-loss. In the Python SDK, you’d use TrailingStopOrderRequest(trail_price=..., or trail_percent=...) similarly
docs.alpaca.markets
docs.alpaca.markets
. – Bracket Orders (Take-Profit/Stop-Loss OCO): A bracket order is a primary order with two conditional exit orders (a profit-taking limit and a stop-loss). When the primary order fills, the two exit orders are activated. If one exit order executes, the other is cancelled (One-Cancels-Other logic). To place a bracket, submit an order with order_class= "bracket" and include a take_profit object and stop_loss object in the payload
docs.alpaca.markets
. For example, to buy 5 shares of SPY with a profit target of $400 and a stop loss at $300:
python
Copy
Edit
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderClass

bracket_order = MarketOrderRequest(
    symbol="SPY",
    qty=5,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY,
    order_class=OrderClass.BRACKET,
    take_profit=TakeProfitRequest(limit_price=400),
    stop_loss=StopLossRequest(stop_price=300)
)
trading_client.submit_order(order_data=bracket_order)
In JSON (raw REST API), this corresponds to including "order_class": "bracket", and nested "take_profit": {"limit_price": 400} and "stop_loss": {"stop_price": 300} in the POST /orders payload
docs.alpaca.markets
. Bracket orders ensure you have an exit plan on both sides – if the price hits 400, you sell for profit, if it falls to 300, you stop out. You can also specify a stop_loss.limit_price if you want the stop to be a stop-limit (to avoid selling below a certain price). Note that bracket orders are only allowed as DAY or GTC, and Alpaca will reject combinations that don’t conform to its rules (e.g., bracket + extended_hours=true is not supported
docs.alpaca.markets
). – Other Order Options: Alpaca supports time-in-force (TIF) values: day, gtc (good-til-canceled), opg (at market open), cls (at market close), ioc (immediate-or-cancel), and fok (fill-or-kill). Certain order types have restrictions on TIF (e.g., opg orders must be submitted before market open). Alpaca also allows orders during extended hours (pre-market 4AM-9:30AM ET, and after-hours 4PM-8PM ET) for limit and stop-limit orders. To make an order eligible for extended hours, you must set the boolean parameter extended_hours=true in the order request
docs.alpaca.markets
. Only limit (including stop-limit) and market orders with TIF of day or gtc can be placed in extended hours
docs.alpaca.markets
. If extended_hours is true and you submit after 8PM or before 4AM, the order will be queued for the next session. Keep in mind liquidity is lower in extended hours and Alpaca imposes some limitations (for example, notional order sizes (dollar-based) are not allowed in extended hours trading
docs.alpaca.markets
). Always check clock to see if the market is in regular or extended session when needed. Order Modification & Cancellation: Alpaca provides a replace order feature to modify open orders. Instead of cancelling and placing a new order, you can send a PATCH to /v2/orders/{order_id} with the fields you want to update (currently permitted: quantity, limit_price, stop_price, and time_in_force)
medium.com
. This will attempt to replace the order atomically. If the original order gets filled before the replacement reaches the exchange, the replacement will be rejected (and you’ll get an update via the stream)
medium.com
. Replacing is useful to adjust orders (e.g., move a limit price) without losing queue priority in certain cases. To cancel an order, you can call DELETE on /v2/orders/{order_id}. Alpaca also provides a bulk cancel: DELETE /v2/orders with no ID will cancel all open orders
docs.alpaca.markets
. The response will list which orders were cancelled. Use caution with this in live trading. Note that you cannot cancel orders that are already filled or those that are in the process of executing (you might get a partial fill and a partial cancel in such cases). Order Status and Responses: When you place an order (POST /orders), you will receive a response JSON representing the order object. Important fields include: id (a unique order UUID), client_order_id (an optional client-defined ID you can set for your own tracking), status (new, accepted, filled, partially_filled, canceled, etc.), filled_qty, filled_avg_price, and timestamps. Use these to verify the order was accepted. If the order is rejected (status rejected), the response will include a reason (e.g., “insufficient buying power” or “market closed”). You can also retrieve orders after placement via GET endpoints:
GET /v2/orders – list of your open orders (with optional query params like status=closed to include filled/cancelled ones, or filtering by date range).
GET /v2/orders/{order_id} – get a specific order by its UUID.
GET /v2/orders:by_client_order_id?client_order_id=XYZ – retrieve an order by the client ID you assigned it
docs.alpaca.markets
.
These allow you to query status at any time. However, a more efficient way to track orders is to use Alpaca’s streaming WebSocket for order updates (described in the WebSocket section below).
Positions and Portfolio Monitoring
Open Positions: The /v2/positions endpoint (GET) returns all your current open positions (one per symbol). Each position object includes fields such as symbol, qty (positive for long, negative for short), avg_entry_price (your average cost per share), market_value (current value), unrealized_pl (unrealized profit/loss), etc. Use this to monitor your portfolio holdings. You can also query a single symbol’s position via GET /v2/positions/{symbol} if you want just one
docs.alpaca.markets
. If you have no position in that symbol, it returns 404 (so handle accordingly). Closing Positions: To liquidate positions quickly, Alpaca provides shortcut endpoints. DELETE /v2/positions/{symbol} will submit a market order to close the entire position of that symbol (whether long or short)
docs.alpaca.markets
. Similarly, DELETE /v2/positions with no symbol will attempt to close all open positions in your account at market
docs.alpaca.markets
. These endpoints are useful for emergency liquidation or rotating a portfolio entirely to cash. The responses will indicate success or any errors (e.g., if an order was rejected). Behind the scenes, these endpoints place market orders on your behalf, so they are subject to normal trade rules (they won’t execute outside market hours unless the asset is crypto or if it’s within extended hours for stocks). Portfolio History: As mentioned, GET /v2/account/portfolio/history provides your account’s historical equity curve. You can specify parameters such as timeframe (daily, 15Min, etc.), date range, and whether to include extended hours values. The response includes arrays of timestamps, equity values, profit/loss, and potentially benchmark data. This is useful for analyzing performance over time – for example, computing your daily P&L or maximum drawdown. An example response (truncated) might look like:
json
Copy
Edit
{
  "timestamp": [1622505600, 1622592000, ...],
  "equity": [10000.00, 10150.25, ...],
  "profit_loss": [0, 150.25, ...],
  "profit_loss_pct": [0, 0.015025, ...],
  "base_value": 10000.00,
  "timeframe": "1D"
}
indicating that starting from a base value of $10,000, the equity on 2021-06-01 was $10,150.25 (a +1.5025% gain). You can adjust the timeframe to intraday (e.g., 5Min intervals for a shorter range) to see more granular equity changes. This endpoint is read-only and is especially handy for building performance charts or tracking metrics.
Additional Features: Clock, Calendar, and Funding
We’ve touched on the clock and calendar endpoints which help you align with market hours. Alpaca’s clock endpoint not only tells you if the market is open, but also the current timestamp and next open/close times, which is updated in real-time. The calendar can be queried for any date range to get scheduled open/close times (useful for planning around holidays or half-days). Crypto Funding and Transfers: If you trade crypto with Alpaca, you have a crypto account wallet. Alpaca allows you to deposit and withdraw cryptocurrencies via API. The /v2/crypto funding endpoints handle this: for example, GET /v2/crypto/wallets will list your crypto wallet balances by coin
docs.alpaca.markets
, and POST /v2/crypto/transfer (or specific endpoints like /crypto/withdrawals) can be used to request a withdrawal of crypto to an external wallet. Before withdrawing, you need to whitelist the destination address – Alpaca’s API provides endpoints to manage whitelisted addresses
docs.alpaca.markets
 for crypto withdrawals (for security, only pre-approved addresses can be used). You can also check the estimated blockchain network fee for a withdrawal via an endpoint
docs.alpaca.markets
. When depositing, Alpaca will provide you deposit addresses for each supported coin. All these allow you to integrate crypto funding flows in your app. (Note: For traditional brokerage cash funding – ACH or bank transfers – Alpaca does not offer a public trading API endpoint; those actions are typically done via the Alpaca dashboard or the Broker API for business integrations.)
Market Data API (Historical & Real-Time Data)
Alpaca’s Market Data API gives you access to both historical data and real-time streaming data for stocks and crypto. This is separate from the trading endpoints and uses different base URLs (data.alpaca.markets). Your trading API keys also authorize you for market data (with certain limits depending on your subscription).
Historical Market Data (REST API)
For stocks, Alpaca provides up to 5+ years of historical data including minute bars, daily bars, trades, quotes, and news. The base endpoint is https://data.alpaca.markets/v2/stocks. Key endpoints include:
GET /v2/stocks/{symbol}/bars – for candlestick/bar data. You can specify timeframe (e.g. 1Min, 5Min, 15Min, 1Day), a date start and end (or limit for number of bars) to retrieve. The response provides OHLCV bars. For example, you might call /v2/stocks/AAPL/bars?timeframe=1Day&start=2023-01-01&end=2023-12-31 for daily bars in 2023. There are also batch endpoints to get bars for multiple symbols in one call.
GET /v2/stocks/{symbol}/trades and /quotes – tick-level data for trades and quotes. You can query a time range and get every trade or quote in that interval (this can be large, so use carefully). For most applications, bars or the latest quote/trade are sufficient.
GET /v2/stocks/{symbol}/snapshot – a convenient call that returns a snapshot of the current market state for one symbol, including the latest trade, latest quote, and minute/daily bar. There is also a batch version for multiple symbols.
GET /v2/stocks/{symbol}/latest/trades (and /latest/quotes, /latest/bars) – quick endpoints to get just the most recent data point for that symbol
docs.alpaca.markets
docs.alpaca.markets
. For example, /latest/trades gives you the last trade price and timestamp, which is useful for getting real-time price without maintaining a stream.
For crypto, Alpaca offers aggregated data for supported crypto pairs. The base is https://data.alpaca.markets/v1beta3/crypto. Endpoints are analogous: /crypto/{symbol}/bars, /crypto/{symbol}/trades, etc., as well as snapshots and latest quotes/trades
docs.alpaca.markets
docs.alpaca.markets
. One difference is that crypto trades 24/7, so there are no “market hours” but you can still request bars in various timeframes (minute, hour, day). Crypto data is consolidated across the venues Alpaca uses for execution. If no trade occurred in a given bar interval, the bar will have zero volume and equal open/high/low/close determined by the midpoint of bid/ask around that time
docs.alpaca.markets
. This ensures continuous data even during low liquidity periods. Data Access Tiers: Alpaca provides free data and higher-tier (subscription) data. By default, stock data for free accounts uses the IEX feed (which is real-time quotes/trades from the IEX exchange only) and provides consolidated (SIP) data with a 15-minute delay
reddit.com
. If you need real-time full market data (SIP from all exchanges), Alpaca offers subscriptions. When using the API, you specify the feed source in the request. For example, to get real-time consolidated data (if you have a subscription), you add a parameter feed=sip. Without that, free users get IEX or delayed data. The documentation notes the difference: IEX vs SIP – IEX is a single exchange (limited volume, but free realtime), whereas SIP is the official consolidated tape (all exchanges)
docs.alpaca.markets
. Crypto data does not have such tiers – it’s free real-time, since crypto markets are inherently consolidated in Alpaca’s system. Usage Example: Using the Python SDK (old version) to get historical bars might look like:
python
Copy
Edit
import alpaca_trade_api as tradeapi

api = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets')
bars = api.get_bars("AAPL", tradeapi.TimeFrame.Day, "2023-01-01", "2023-03-01", adjustment='raw')
for bar in bars:
    print(bar.t, bar.o, bar.h, bar.l, bar.c, bar.v)
This would print each day’s timestamp and OHLCV for AAPL from January through February 2023. The REST API itself would be /v2/stocks/AAPL/bars?timeframe=1Day&start=2023-01-01.... You can also specify adjustment=split or adjustment=all if you want the historical prices adjusted for splits or dividends; by default, raw prices are returned (no corporate action adjustment). Rate Limits for Data: Market data APIs are subject to the same rate limit (200 requests per minute by default) as trading APIs
alpaca.markets
. If you need to pull a lot of data (for example, backtesting with long history), avoid making many small requests; instead use larger time windows per request or the batch endpoints that allow multiple symbols. Hitting the rate limit will result in HTTP 429 Too Many Requests
alpaca.markets
, in which case you should slow down and retry after a pause.
Real-time Streaming Data (WebSocket)
For applications that require live market data updates (price quotes, trades, live candlesticks), Alpaca offers a WebSocket streaming service. This is separate from the trading updates stream (different URL). Stock Data Stream: Connect to wss://stream.data.alpaca.markets/v2/{feed} for stocks
docs.alpaca.markets
. Here {feed} can be one of: iex (free real-time from IEX), sip (requires subscription, real-time full market), or delayed_sip (15-min delayed consolidated). For example, wss://stream.data.alpaca.markets/v2/iex for free tier stock stream. There’s also a sandbox URL for testing (stream.data.sandbox...). After connecting, you must authenticate by sending the same { "action": "auth", "key": "YOUR_KEY", "secret": "YOUR_SECRET" } message over the socket as you do with trading streams. If successful, you’ll receive a { "T": "success", "msg": "authenticated" } message (or an error). Once authed, you subscribe to specific data streams by sending a message specifying which symbols and data you want. The JSON format is:
json
Copy
Edit
{ 
  "action": "subscribe", 
  "trades": ["SYMBOL1", "SYMBOL2"], 
  "quotes": ["SYMBOL1", "SYMBOL2"], 
  "bars": ["SYMBOL1"] 
}
You can subscribe to any combination of trade ticks, quotes, and minute bars for given symbols
docs.alpaca.markets
. Wildcards like "bars": ["*"] subscribe you to all symbols, but that’s typically only used in the free sandbox or by special arrangement (it would be overwhelming for all market symbols!). The server will start pushing messages for each event. For example, a trade message looks like: {"T":"t","S":"AAPL","p":126.55,"s":1,"t":"2021-02-22T15:51:44.208Z",...} (where p is price, s is size)
docs.alpaca.markets
docs.alpaca.markets
. A quote message has "T":"q" and fields for bid and ask prices (bp, ap) and sizes
docs.alpaca.markets
docs.alpaca.markets
. A bar message has "T":"b" with OHLCV for the timeframe (with minute bars emitted every minute)
docs.alpaca.markets
. Additionally, the stream can send other events: "T":"error" messages if something goes wrong, or control messages like subscription confirmations. The stock stream also provides channels for trading status updates (halts/resumes) and LULD (limit up/down) events if you subscribe to "statuses" or "lulds". There are also channels for dailyBars (daily bar updates after market close) and updatedBars (if late trades modify a previous bar)
docs.alpaca.markets
. These are more advanced, but available if needed. Crypto Data Stream: The crypto streaming endpoint is wss://stream.data.alpaca.markets/v1beta3/crypto/us
docs.alpaca.markets
. It works similarly: you authenticate with your keys, then subscribe to channels. The available channels for crypto are trades, quotes, minute bars, daily bars, updated bars, and also an orderbook (book top-of-book data) channel. For instance, to stream Bitcoin trades and quotes you might send: {"action": "subscribe", "trades": ["BTC/USD"], "quotes": ["BTC/USD"]}. Crypto trade messages have a similar format with fields like p (price), s (size), and an extra tks field indicating taker side (buyer or seller)
docs.alpaca.markets
docs.alpaca.markets
. Crypto quote messages include bid/ask price and size
docs.alpaca.markets
docs.alpaca.markets
. One difference is that crypto streaming may aggregate data from multiple venues; however, Alpaca’s API presents it as a unified feed. Handling Stream Data: When you receive streaming data, you’ll get a separate message for each event. It’s up to you to update your UI or algorithm state accordingly. For example, on receiving a trade tick, update your last price; on a new bar, append it to your chart. The streaming connection will send a small ping every 30 seconds to keep alive. You should handle reconnect logic in case the connection drops (with exponential backoff to respect rate limits). Also note, there’s generally a limit of one concurrent websocket connection per account for the data streams, so you can subscribe to many symbols on one connection but shouldn’t open multiple connections (unless you have a paid plan that allows more)
github.com
.
Live Trade Updates via WebSocket
In addition to market data, Alpaca provides a trade updates stream for your account’s order events. This is extremely useful for tracking order statuses in real-time without polling REST endpoints. Trade/Account Updates Stream: Connect to wss://api.alpaca.markets/stream for live trading (or wss://paper-api.alpaca.markets/stream for paper accounts)
docs.alpaca.markets
. This is the trading data stream (different from the market data streams). After connecting, authenticate by sending your API key and secret in an "action": "auth" message, just like the data stream auth. On success, you’ll get {"stream": "authorization", "data": {"status": "authorized", ...}}
docs.alpaca.markets
docs.alpaca.markets
. Once authenticated, you can listen to the channel "trade_updates" by sending:
json
Copy
Edit
{ "action": "listen", "data": { "streams": ["trade_updates"] } }
The server will confirm your subscription with a "stream": "listening" message listing the streams you are subscribed to
docs.alpaca.markets
docs.alpaca.markets
. From that point, any events related to your orders will be pushed in real-time under "stream": "trade_updates". Events you’ll receive include
docs.alpaca.markets
docs.alpaca.markets
:
new – your order has been accepted and routed (it’s live).
fill – your order (or one of its legs) fully filled. Contains details like price, qty filled, and position_qty (your total position after the fill)
docs.alpaca.markets
.
partial_fill – a partial fill happened (order is not complete yet)
docs.alpaca.markets
.
canceled – your order was canceled (by you or the system)
docs.alpaca.markets
.
expired – the order expired due to its time-in-force (e.g., end of day)
docs.alpaca.markets
.
done_for_day – order reached end of day (for GTC orders that pause overnight).
replaced – an order was replaced by a new order (after a modify request)
docs.alpaca.markets
.
order_cancel_rejected or order_replacement_rejected – if an attempt to cancel or replace failed (rare).
Each message will have an order object with the full updated order info (same structure as the REST API order). For example, when you get a fill event, the order.status would now be "filled" and filled_qty equal to the order’s quantity. If it’s a partial fill, status might still be "partially_filled" and filled_qty < qty. The trade_updates stream is the best way to keep an up-to-date view of your orders and positions. You can use it to drive portfolio updates (e.g., on a fill event, update your position quantity and cash on hand, etc.). This stream also notifies you of important events like order rejection (e.g., due to insufficient funds or symbol halt) so you can respond immediately. One thing to note: the paper trading stream uses binary framing for messages (to optimize bandwidth) whereas the data streams use text JSON frames
docs.alpaca.markets
. The Alpaca SDKs handle this detail for you, but if you use a raw WebSocket client, be aware you might need to handle a binary message (MessagePack encoded). You can request JSON by ensuring the Content-Type: application/json header on connect, or just decode MessagePack if encountered. Most users won’t notice this if using official libraries.
Using the Alpaca Python SDK (alpaca-trade-api)
Alpaca provides official SDKs in multiple languages. For Python, the legacy SDK is alpaca-trade-api and the new one is alpaca-py. These abstract a lot of the HTTP calls and help manage streaming. Installation: You can install the Python SDK via pip:
bash
Copy
Edit
pip install alpaca-trade-api
This will install the library and its dependencies (ensure you’re on Python 3.7+ as required
github.com
github.com
). There is also alpaca-py (the newer SDK) which can be installed via pip install alpaca-py. The examples below generally apply to both, with minor differences in import paths. Authentication & Initialization: The SDK allows you to set your API keys via environment variables or pass them directly. For example:
python
Copy
Edit
import alpaca_trade_api as tradeapi

API_KEY = "<your api key>"
API_SECRET = "<your secret>"
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL)
This creates a client object for REST calls. Alternatively, set the environment variables APCA_API_KEY_ID and APCA_API_SECRET_KEY (and APCA_API_BASE_URL for paper vs live), then you can call tradeapi.REST() without parameters and it will pick them up. Getting Account Info: Using the api object:
python
Copy
Edit
account = api.get_account()
print(account.status, account.equity, account.buying_power)
This returns an Account object with attributes for all the account fields (status, equity, buying_power, etc.). For instance, account.trading_blocked would be a boolean if trading is blocked
docs.alpaca.markets
. You can print or log these to verify your account’s state. Placing Orders (via SDK): The SDK provides a submit_order() method. You can call it with arguments or, in the new SDK, by constructing order request objects (as we showed earlier). Using the older alpaca_trade_api, you could do:
python
Copy
Edit
api.submit_order(symbol="AAPL", qty=5, side="buy", type="limit", time_in_force="day", limit_price=150.00)
This will return an Order object if successful. You can access its id or other attributes (status, etc.). If there’s an error (like invalid params or insufficient funds), the SDK will raise an exception (e.g., APIError). It’s good practice to wrap order calls in try/except to handle these errors gracefully. For advanced orders like bracket or trailing stop, the older SDK allowed passing additional params like order_class="bracket", take_profit={"limit_price": ...}, stop_loss={"stop_price": ..., "limit_price": ...}, or trail_percent=.... The new SDK uses distinct classes but achieves the same result. Check the official examples for the exact syntax for complex orders. Fetching Orders and Positions:
To get open orders: api.list_orders(status="open") returns a list of Order objects.
To get a specific order by client id: api.get_order_by_client_order_id("my_id").
To cancel: api.cancel_order(order_id), or all: api.cancel_all_orders().
For positions: api.list_positions() gives a list of Position objects for each open position. For a particular symbol: api.get_position("AAPL"). And to close positions: api.close_position("AAPL") or api.close_all_positions() – these correspond to the DELETE endpoints for convenience. Market Data via SDK: The Python SDK also offers data fetching methods. In the legacy SDK, you’d use api.get_barset(symbols, timeframe, limit or start/end) which returns historical bars. (Note: get_barset has been replaced by get_bars in newer versions and eventually by the new SDK’s data module). Example:
python
Copy
Edit
bars = api.get_barset("SPY", "15Min", limit=100)
for bar in bars["SPY"]:
    print(bar.t, bar.c)  # time and close price
This gets the last 100 15-minute bars for SPY. The new alpaca-py SDK has a separate data client (alpaca.data.Api) that can be used similarly, and it provides async support if needed for streaming. Live Data Streaming with SDK: The Python SDK can also simplify WebSocket connections. In alpaca_trade_api, you can use the Stream or StreamConn class:
python
Copy
Edit
from alpaca_trade_api.stream import Stream

stream = Stream(API_KEY, API_SECRET, base_url=BASE_URL, data_feed='iex')  # or 'sip' if subscribed
# subscribe to data
stream.subscribe_trades(handler_function, "AAPL")
stream.subscribe_quotes(handler_function, "AAPL")
# subscribe to account updates
stream.subscribe_trade_updates(trade_update_handler)
stream.run()
Here, handler_function is your function to handle incoming data (the SDK will call it with Trade or Quote objects). trade_update_handler will handle order events. The SDK manages threading and reconnection for you, making it much easier to work with streams. The new alpaca-py has a similar stream management in its alpaca.data.stream and alpaca.trading.stream modules. Always refer to Alpaca’s GitHub documentation for the latest usage pattern, as the SDKs evolve. Other Language SDKs: Alpaca also provides official SDKs for JavaScript (NPM package @alpacahq/alpaca-trade-api), C# (.NET SDK in Alpaca.Markets), Go, and others
docs.alpaca.markets
docs.alpaca.markets
. Their usage is analogous: you initialize a client with keys, then call methods like getAccount(), placeOrder(), etc. The examples in Alpaca’s docs show code in multiple languages side by side for common tasks, which can be helpful if you use a different language.
Rate Limits, Errors, and Best Practices
API Rate Limits: Alpaca’s API is throttled to 200 requests per minute per account
alpaca.markets
. This limit is shared across all REST endpoints (and possibly data endpoints). If you exceed this, you’ll receive HTTP 429 Too Many Requests, and the response will include a message about rate limit. If your strategy needs more, Alpaca can grant higher limits (e.g., 1000/min) for non-retail (by request)
alpaca.markets
, but for most users 200/min is plenty. Best practices to avoid rate limits: batch requests when possible (for data), use streaming instead of polling for updates, and implement exponential backoff/retry for failed calls. Also consider that each API key (account) is separate – if you manage multiple accounts with separate keys, each has its own quota. Error Handling: The Alpaca API uses standard HTTP status codes. Common ones: 400 for bad request (e.g., invalid format or parameter), 401 or 403 for authentication issues (check your keys and permissions), 404 for not found (e.g., requesting a symbol that doesn’t exist or an order ID that’s wrong), and 422 for requests that are well-formed but cannot be processed (e.g., trying to sell more shares than you have, etc.). The response body typically contains a JSON with a field like {"message": "explanation of error"}. The Python SDK will raise exceptions in these cases (e.g., rest.APIError: {"code": 40410000, "message": "symbol not found"}), which you should catch. Pay attention to error messages – for example, "insufficient buying power" means you tried to buy beyond your resources, and "order not found" might mean you passed a wrong ID. Some errors are transient (like a timeout or 504 if Alpaca is busy); those you can retry after a brief wait. Common Pitfalls and Best Practices:
Paper vs Live Confusion: Ensure you’re using the correct API endpoint and keys for the environment. Many new users accidentally send orders to the paper API with live keys or vice versa, which results in auth errors. Keep separate configurations for paper and live.
Time in Force & Session Timing: If you place orders outside of market hours without extended_hours=true, they will be queued for the next open. If that’s not desired, use ioc or similar TIF to have them cancel automatically if not filled immediately. Conversely, if you want an order to execute in pre-market or post-market, remember to set extended_hours: true and use limit orders. Market orders outside regular hours are not accepted (except crypto).
Partial Fills & Order Lifecycle: Your buy or sell may not fill all at once. Always handle the possibility of partial_fill. The trade updates stream’s fill vs partial_fill event helps here. If you’re not using the stream, you might poll GET /orders/{id} until status is filled or done to ensure completion. Don’t assume an order is fully executed just because the API accepted it.
Cancelling and Replacing Orders: If you modify an order by cancelling, be aware there’s a race condition: the order might fill before cancellation. Alpaca’s replace API helps but not in all cases. Always check the trade updates – if you see a fill event after you sent cancel, you know part or all of the order got filled. Similarly, use client order IDs so you can idempotently track orders, especially if your system reconnects and wants to avoid duplicating orders.
Clock Synchronization: All timestamps from Alpaca are in UTC (e.g., "2025-06-04T07:30:00Z"). When sending start/end times for data or comparing against the market clock, be mindful of time zones. The clock endpoint returns times in Eastern Time (New York time) for open/close, but also often an is_open flag. Converting to your local time can help if needed. Generally, use the clock’s timestamp as a reliable current time reference.
Corporate Actions Impact: If you hold stocks through splits or dividends, your position quantities and cash may change. Alpaca processes these often after market close or before the next open. You might see new cash balance from a dividend (reflected in account.buying_power and an account activity of type DIV), or a position quantity jump (with a corresponding change in avg_entry_price) due to a split. It’s wise to periodically pull positions fresh rather than caching them indefinitely, so you capture these changes. The Position Average Entry Price Calculation in Alpaca’s FAQ explains how they adjust your cost basis in corporate actions
docs.alpaca.markets
.
Testing & Paper Trading: Always test your algorithm on the paper trading API first. The paper trading environment is very close to live, but note some differences: Paper trading will fill market orders immediately at a simulated price (usually the midpoint or last trade), which might not reflect real slippage. Paper also does not simulate things like auction imbalances or certain volatility halts. Nonetheless, it’s a safe playground. You can reset your paper account equity from the web dashboard if needed. Use paper to fine-tune your API usage and error handling.
Secure Your Keys: Never hard-code keys in code that could be exposed. If you’re open-sourcing or sharing code, use environment variables. For production, consider using separate key pairs for different applications or permissions (Alpaca allows generating multiple keys). If a key is compromised, revoke it immediately from the Alpaca dashboard.
Scaling and WebSockets: Alpaca allows one trading data stream per account (for order updates) – you don’t need more than one since it carries all your orders. For market data streams, free accounts are limited to one concurrent connection for IEX data
github.com
. If you have multiple consumers, you may need to have one service subscribe and then broadcast internally. If you plan to follow a large number of symbols live, consider the streaming approach rather than polling hundreds of quotes repeatedly. And if you are only interested in end-of-day or minute-by-minute data, the historical API with periodic calls might suffice (to reduce load).
Rate Limit Backoff: If you approach the 200/minute limit (which is roughly 3.3 requests per second), you should implement a short sleep or queueing mechanism. The 429 error response from Alpaca includes a header Retry-After (in seconds) or you can simply wait a few seconds and try again. Hitting the limit occasionally is okay (your request will just be throttled), but if you hit it consistently, reconsider your API usage pattern.
By following these practices and fully leveraging Alpaca’s robust API – including the real-time streams and comprehensive SDK – you can build powerful trading applications. Alpaca’s documentation and community Slack/forum are great resources if you run into questions. Remember to always refer to the official API docs for the most up-to-date details and to double-check any behavior (as the platform is continually improved). Happy trading with Alpaca! References:
Official Alpaca API Documentation
docs.alpaca.markets
docs.alpaca.markets
docs.alpaca.markets
docs.alpaca.markets
alpaca.markets
 (covers API usage, authentication, streaming, and limits)
Alpaca Knowledge Base and Support FAQ
alpaca.markets
alpaca.markets
 (rate limits and usage policies)
Alpaca Developer Guides (Orders, Account Configurations, etc.)
medium.com
docs.alpaca.markets
 – for advanced order types and account settings
Alpaca SDK GitHub Repositories
github.com
 and Examples
docs.alpaca.markets
docs.alpaca.markets
 (illustrating SDK usage in multiple languages)