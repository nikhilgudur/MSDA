from cryptocmd import CmcScraper


def get_coin_data(coin_code, coin_name):
    scraper = CmcScraper(coin_code=coin_code, coin_name=coin_name)

    # headers, data = scraper.get_data()

    # coin_json_data = scraper.get_data("json")

    scraper.export("csv", name=f"./coin_datasets/{coin_code}_all_time")

    df = scraper.get_dataframe()

    return df


get_coin_data("btc", "bitcoin")

get_coin_data("eth", "ethereum")

get_coin_data("aave", "aave")

get_coin_data("doge", "dogecoin")

get_coin_data("xrp", "ripple")
