import asyncio
import json
from kalshi_bot.kalshi_client import KalshiClient

async def main():
    kc = KalshiClient()
    event_id = "KXBTCD-25JUN1610"
    markets = kc.get_markets(event_id)
    matching_markets = markets["markets"]

    if not matching_markets:
        print(f"No markets found for event: {event_id}")
        return

    print("Available market tickers:")
    for m in matching_markets:
        print(m["ticker"])

    # Wait for BTC price file to be populated
    while True:
        price = get_btc_price_from_file()
        if price:
            break
        print("Waiting for BTC price...")
        await asyncio.sleep(1)

    # Select market closest to BTC price
    target_market = min(
        matching_markets,
        key=lambda m: abs(float(m["ticker"].split("-")[-1][1:]) - price)
    )

    if not target_market:
        print("No market found near current BTC price.")
        return

    target_market = target_market["ticker"]

    print(f"Using market ticker: {target_market}")
    print(f"Streaming market: {target_market}")
    asyncio.create_task(kc.subscribe_to_market(target_market))

    while True:
        await asyncio.sleep(1)
        prices = kc.get_mid_prices(target_market)

def get_btc_price_from_file():
    try:
        with open("btc_price.json", "r") as f:
            return json.load(f)["price"]
    except:
        return None


asyncio.run(main())

