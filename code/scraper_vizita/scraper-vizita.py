from selenium import webdriver
import time
import os
import pandas as pd
import re
from tqdm import tqdm

MAIN_URL = "https://vizita.si/arhiv/popovi-zdravniki"


def setup_chrome_headless() -> webdriver.Chrome:
    chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)

    return driver


def get_main_forums(driver: webdriver.Chrome, url: str):
    def accept_cookies(driver: webdriver.Chrome) -> None:
        accept_cookies_button = driver.find_element(
            by="xpath",
            value="//onl-cookie/div/div/div/div[2]/a[1]",
        )
        accept_cookies_button.click()

    def get_max_number_of_pages(driver: webdriver.Chrome) -> int:
        pagination = driver.find_element(by="xpath", value="//onl-pager/div")
        last_page_href = pagination.find_element(by="xpath", value="./a[last()]").get_attribute(
            "href"
        )
        return int(re.search(r"\d+$", last_page_href).group())

    def delete_popup(driver: webdriver.Chrome) -> None:
        try:
            popup = driver.find_element(
                by="xpath",
                value="//onl-modal/div/div/a",
            )
            popup.click()
        except:
            pass

    def get_headlines_on_page(driver: webdriver.Chrome) -> list[dict]:
        headlines = driver.find_elements(
            by="xpath", value="//onl-archive/div/div/div/main/div[2]/a"
        )

        hls = []
        for headline in headlines:
            title = headline.find_element(by="xpath", value="./div[2]/a/div/div[2]/h1/span")
            url = headline.get_attribute("href")

            hls.append({"url": url, "title": title.text})

        return hls

    def get_data_from_headline(driver: webdriver.Chrome, headline: dict) -> list[dict]:
        driver.get(headline["url"])
        time.sleep(0.5)

        delete_popup(driver)
        try:
            question = driver.find_element(
                by="xpath",
                value="//div[@class='qa__item qa__item--question']//div[@class='qa__content']",
            )
            answer = driver.find_element(
                by="xpath",
                value="//div[@class='qa__item qa__item--answer']//div[@class='qa__content']",
            )
        except:
            return {"title": headline["title"], "question": "", "answer": ""}

        return {
            "title": headline["title"],
            "question": question.text,
            "answer": answer.text,
        }

    # Start of the function
    delete_popup(driver)
    driver.get(url)
    time.sleep(0.5)

    delete_popup(driver)
    accept_cookies(driver)
    time.sleep(0.5)

    delete_popup(driver)
    max_pages = get_max_number_of_pages(driver)
    # max_pages = 10

    bad_articles = 0
    res = []

    progress_bar = tqdm(
        total=max_pages,
        desc="Getting articles from pages",
        postfix=f"[BAD ARTICLES: {bad_articles}]",
    )

    for page in range(max_pages):
        driver.get(f"{url}?stran={page + 1}")
        time.sleep(0.5)

        # delete_popup(driver)
        headlines = get_headlines_on_page(driver)

        for headline in headlines:
            data_call = get_data_from_headline(driver, headline)
            if data_call["question"] == "" and data_call["answer"] == "":
                bad_articles += 1

            res.append(data_call)

        progress_bar.set_postfix_str(f"[BAD ARTICLES: {bad_articles}]")
        progress_bar.update(1)

    progress_bar.close()
    return res


def parse_end_to_end(driver: webdriver.Chrome, url: str):
    main_forms = get_main_forums(driver, url)
    df = pd.DataFrame(main_forms)

    current_directory = os.path.dirname(os.path.abspath(__file__))

    csv_file_path = os.path.join(current_directory, "vizita-si-scraper.csv")
    df.to_csv(csv_file_path, index=False, encoding="utf-8", sep=";")

    json_file_path = os.path.join(current_directory, "vizita-si-scraper.json")
    df.to_json(json_file_path, orient="records", force_ascii=False)


if __name__ == "__main__":
    driver = setup_chrome_headless()
    parse_end_to_end(driver, MAIN_URL)
    driver.close()
