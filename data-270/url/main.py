from cryptocmd import CmcScraper

scraper = CmcScraper(coin_code="btc", coin_name="bitcoin")

headers, data = scraper.get_data()

bitcoin_json_data = scraper.get_data("json")

scraper.export("csv", name="bitcoin_all_time")

df = scraper.get_dataframe()
