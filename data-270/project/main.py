from utils import get_coin_data_csv, get_coin_data_df

if __name__ == "__main__":
    # Fetch data for the following coins and save them as CSV files
    get_coin_data_csv("btc", "bitcoin")
    get_coin_data_csv("eth", "ethereum")
    get_coin_data_csv("aave", "aave")
    get_coin_data_csv("doge", "dogecoin")

    btc_df = get_coin_data_df("btc", "bitcoin")

    # print(btc_df.info())

    # print(btc_df.isna().sum())
