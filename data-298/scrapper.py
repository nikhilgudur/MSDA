import requests
import os
import sys
from tqdm import tqdm

URL = 'https://safebooru.org/index.php?page=dapi&s=post&q=index&json=1&limit=1000'


def get_posts() -> list | None:
    response = requests.get(URL)

    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code)
        sys.exit()
    return None


def get_images(posts):
    output_path = './data'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for post in tqdm(posts):
        img_url = post['file_url']
        img_name = img_url.split('/')[-1]

        response = requests.get(img_url)

        with open(f'{output_path}/{img_name}', 'wb') as file:
            file.write(response.content)

            file.close()

        tagfile_name = img_name.split(".")[0]

        with open(f'{output_path}/{tagfile_name}.txt', 'x') as file:
            file.write(post['tags'])

            file.close()


def main():
    posts = get_posts()

    # print(posts)

    get_images(posts)


if __name__ == "__main__":
    main()
