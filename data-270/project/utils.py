from cryptocmd import CmcScraper
import os
import logging
from pandas import DataFrame as df


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_coin_data(coin_code: str, coin_name: str) -> df:
    if os.path.isfile(f"./datasets/{coin_code}_all_time.csv"):
        log.info(f"Dataset for {coin_name} already exists")
        return
    scraper = CmcScraper(coin_code=coin_code, coin_name=coin_name)

    # headers, data = scraper.get_data()

    # coin_json_data = scraper.get_data("json")

    scraper.export("csv", name=f"./coin_datasets/{coin_code}_all_time")

    df = scraper.get_dataframe()

    return df


if __name__ == "__main__":
    get_coin_data("btc", "bitcoin")

    get_coin_data("eth", "ethereum")

    get_coin_data("aave", "aave")

    get_coin_data("doge", "dogecoin")

    # get_coin_data("xrp", "ripple")
