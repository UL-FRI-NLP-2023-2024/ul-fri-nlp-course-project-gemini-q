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
        time.sleep(2)

        delete_popup(driver)
        try:
            driver.find_element(
                by="xpath",
                value="//div[@class='qa__item qa__item--question']",
            )
            driver.find_element(
                by="xpath",
                value="//div[@class='qa__item qa__item--answer']",
            )
        except:
            print("\nArticle not found for headline [" + headline["title"] + "]\n")
            return {"title": headline["title"], "question": "", "answer": ""}

        question = driver.find_element(
            by="xpath",
            value="//onl-article-embed/div/div/div/div[1]/div",
        ).text

        answer = driver.find_element(
            by="xpath",
            value="//onl-article-embed/div/div/div/div[2]/div",
        ).text

        return {"title": headline["title"], "question": question, "answer": answer}

    delete_popup(driver)
    driver.get(url)
    time.sleep(2)

    delete_popup(driver)
    accept_cookies(driver)
    time.sleep(2)

    delete_popup(driver)
    max_pages = get_max_number_of_pages(driver)
    max_pages = 1  # For testing purposes

    headlines = []
    for page in tqdm(range(max_pages), desc="Getting headlines from each page"):
        driver.get(f"{url}?page={page + 1}")
        time.sleep(2)

        delete_popup(driver)
        headlines_call = get_headlines_on_page(driver)
        headlines.extend(headlines_call)

    res = []
    for headline in tqdm(headlines, desc="Getting data from each headline"):
        data_call = get_data_from_headline(driver, headline)
        if data_call["question"] != "" and data_call["answer"] != "":
            res.append(data_call)

    return res


def parse_end_to_end(driver: webdriver.Chrome, url: str):
    main_forms = get_main_forums(driver, url)

    df = pd.DataFrame(main_forms)
    current_directory = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(current_directory, "vizita-si-scraper.csv")
    df.to_csv(csv_file_path, index=False, encoding="utf-8", sep=";")


if __name__ == "__main__":
    driver = setup_chrome_headless()
    parse_end_to_end(driver, MAIN_URL)
    driver.close()
