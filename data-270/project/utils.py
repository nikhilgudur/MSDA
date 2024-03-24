import os
import logging
import json
import time
from datetime import datetime, timedelta
from pandas import DataFrame
from cryptocmd import CmcScraper


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

FETCH_RECORD_FILE = "./fetch_record.dat"


def can_fetch_data(coin_name: str) -> tuple[bool, dict]:

    can_fetch = False

    if os.path.isfile(FETCH_RECORD_FILE):
        with open("./fetch_record.dat", "r", encoding="utf-8") as f:
            if f.read().strip():
                f.seek(0)
                fetch_record = json.load(f)
            else:
                fetch_record = {}
                can_fetch = True

            if coin_name in fetch_record:
                fetch_record_time = fetch_record[coin_name]

                if datetime.strptime(fetch_record_time, "%c") < datetime.now() - timedelta(days=1):
                    can_fetch = True
            else:
                can_fetch = True

    return can_fetch, fetch_record


def get_coin_data_csv(coin_code: str, coin_name: str) -> None:

    can_fetch, fetch_record = can_fetch_data(coin_name)

    if can_fetch:
        log.info(f"Fetching data for {coin_name}")

        scraper = CmcScraper(coin_code=coin_code, coin_name=coin_name)
        scraper.export("csv", name=f"./coin_datasets/{coin_code}_all_time")

        with open(FETCH_RECORD_FILE, "w", encoding="utf-8") as f:
            fetch_record[coin_name] = time.ctime(time.time())
            json.dump(fetch_record, f)


def get_coin_data_df(coin_code: str, coin_name: str) -> DataFrame:

    scraper = CmcScraper(coin_code=coin_code, coin_name=coin_name)

    df = scraper.get_dataframe()

    return df


if __name__ == "__main__":
    get_coin_data_csv("btc", "bitcoin")
    get_coin_data_csv("eth", "ethereum")
    get_coin_data_csv("aave", "aave")
    get_coin_data_csv("doge", "dogecoin")
    # get_coin_data_csv("xrp", "ripple")
