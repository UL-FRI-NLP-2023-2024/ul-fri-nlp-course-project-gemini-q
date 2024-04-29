from selenium import webdriver
import time
import os
import requests
import pandas as pd
import re
import json
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException,
)

# from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options


MAIN_URL = "https://med.over.net/forum"


def setup_firefox_headless() -> webdriver.Firefox:
    firefox_options = webdriver.FirefoxOptions()
    # firefox_options.add_argument("--headless")
    firefox_options.add_argument("start-maximized")
    firefox_options.add_argument("disable-infobars")
    firefox_options.add_argument("--disable-extensions")
    firefox_options.add_argument("--no-sandbox")
    firefox_options.add_argument("--disable-application-cache")
    firefox_options.add_argument("--disable-gpu")
    firefox_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Firefox(options=firefox_options)

    return driver


def setup_chrome_headless() -> webdriver.Chrome:
    # Set up Chrome options
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Ensures the browser runs in headless mode
    chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
    chrome_options.add_argument(
        "--disable-dev-shm-usage"
    )  # Overcome limited resource problems
    chrome_options.add_argument("--start-maximized")  # Maximizes Chrome window size
    chrome_options.add_argument(
        "--disable-infobars"
    )  # Disables the "Chrome is being controlled" infobar
    chrome_options.add_argument("--disable-extensions")  # Disables existing extensions
    chrome_options.add_argument("--disable-gpu")  # Applicable to Windows OS only
    chrome_options.add_argument("--disable-software-rasterizer")

    # Specify the path to chromedriver using Service
    service = Service(executable_path="/usr/lib/chromium-browser/chromedriver")

    # Initialize the Chrome driver
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def get_main_forums(driver: webdriver.Firefox, url: str):
    def accept_cookies(driver: webdriver.Firefox):
        accept_cookies_button = driver.find_element(
            by="xpath", value="//button[@id='didomi-notice-agree-button']"
        )
        accept_cookies_button.click()

    def get_sub_forums(driver: webdriver.Firefox, headline: dict) -> list[dict]:
        sf = []
        url = headline["url"]
        headline_id = url.split("-")[-1].split("/")[0]

        id_name = f"bbp-forum-{headline_id}"

        sub_forms = driver.find_elements(by="xpath", value=f"//tr[@id='{id_name}']")

        for elem in sub_forms:
            sub_forum_url = elem.find_element(by="xpath", value=".//a").get_attribute(
                "href"
            )
            sub_forum_text = elem.find_element(by="xpath", value=".//a").text
            number_of_articles = elem.find_element(
                by="xpath", value="//*[contains(@class, 'u-light')]"
            ).text

            sf.append(
                {
                    "category": headline["title"],
                    "forum": sub_forum_text,
                    "link": sub_forum_url,
                    "posts": number_of_articles,
                }
            )

        return sf

    driver.get(url)
    # time.sleep(0.1)
    accept_cookies(driver)
    # time.sleep(0.1)
    headlines = [
        {
            "url": "https://med.over.net/forum/kategorija/zdravje-3574397/",
            "title": "Zdravje",
        }
    ]

    res = []
    for headline in headlines:
        res.extend(get_sub_forums(driver, headline))

    return res


def get_subforum_subforum(
    driver: webdriver.Firefox,
    url="https://med.over.net/forum/kategorija/dusevno-zdravje-in-odnosi/druzina-3574102/",
):
    def get_tables(driver: webdriver.Firefox):
        try:
            data = []
            tables = driver.find_elements(
                by="xpath", value="//div[@class='table__outer table__outer--margin']"
            )

            for table in tables:
                table_elements = table.find_elements(
                    by="xpath", value=".//tr[@class='table-forum__row']"
                )

                buttons = table.find_elements(
                    by="xpath", value=".//a[@class='button button--link']"
                )

                for button in buttons:
                    data.append(button.get_attribute("href"))
            return data
        except Exception as e:
            print(f"Error in get_tables: {e}")
            return []

    try:
        driver.get(url)
        # time.sleep(0.1)

        data = get_tables(driver)

        return data
    except Exception as e:
        print(f"Error occurred in get_subforum_subforum: {e}")
        return []


def get_forums_from_url(
    driver: webdriver.Firefox,
    url="https://med.over.net/forum/kategorija/zdravje/bolezni-srca-in-ozilja/kardiologija-31/",
):
    try:
        driver.get(url)
        # time.sleep(0.1)

        res_dict = []

        def parse_numbers(string):
            numbers = re.findall(r"\d+", string)
            numbers = [int(num) for num in numbers]
            return numbers

        def get_number_of_pages(driver: webdriver.Firefox):
            try:
                pages = driver.find_element(
                    by="xpath", value="//div[@class='bbp-pagination-links']"
                )
                numbers = parse_numbers(pages.text)
                return numbers[-1]
            except Exception as e:
                print(f"Error in get_number_of_pages: {e}")
                return 0

        num_pgs = get_number_of_pages(driver)
        if num_pgs == 0:
            return []

        # for page in range(1, num_pgs + 1):
        for page in range(1, 2):
            driver.get(f"{url}page/{page}/")
            # time.sleep(0.1)

            try:
                forums_on_site = driver.find_elements(
                    by="xpath",
                    value="//td[contains(@class, 'table-forum__td--title')]//a[contains(@class, 'js-checkVisited')]",
                )
                for forum in forums_on_site:
                    res_dict.append(
                        {"text": forum.text, "link": forum.get_attribute("href")}
                    )
            except Exception as e:
                print(f"Error parsing forums on site: {e}")
                continue

        return res_dict
    except Exception as e:
        print(f"Error occurred: {e}")
        return []  # Return empty list if any error occurs


def get_forum_data(
    driver, url="https://med.over.net/forum/tema/terapija-za-pritisk-22533676/"
):
    try:
        driver.get(url)
        # Set up an explicit wait to handle loading times more gracefully
        wait = WebDriverWait(driver, 10)

        # Wait for the inner page to load properly before proceeding
        inner_page = wait.until(
            EC.presence_of_element_located(
                (By.XPATH, "//article[@class='page__inner']")
            )
        )

        users = inner_page.find_elements(
            By.XPATH, "//span[@class='title title--xsmall title--author']"
        )
        dates = inner_page.find_elements(
            By.XPATH, "//span[@class='text text--small u-light']"
        )
        contents = inner_page.find_elements(
            By.XPATH, "//div[@class='forum-post__content']"
        )

        post_views = inner_page.find_element(
            By.XPATH, "//span[@class='post-views-count']"
        )

        forum_data = []
        for user, date, content in zip(users, dates, contents):
            forum_data.append(
                {
                    "user": user.text,
                    "date": date.text,
                    "content": content.text,
                    "views": post_views.text,
                    "user_html": user.get_attribute("outerHTML"),
                }
            )

        return forum_data
    except TimeoutException:
        print("Error: The page load timed out.")
    except NoSuchElementException:
        print("Error: Some elements were not found on the page.")
    except WebDriverException as e:
        print(f"WebDriver Error: {e}")
    except Exception as e:
        print(f"Unhandled exception occurred: {e}")

    return []  # Return empty data if any error occurs


def parse_end_to_end(driver: webdriver.Firefox, url: str):
    main_forms = get_main_forums(driver, url)

    for main_form in main_forms:
        sub_forums = get_subforum_subforum(driver, main_form["link"])

        for subform in sub_forums:
            forums = get_forums_from_url(driver, subform)
            res_data = []

            for forum_idx_ in tqdm(range(len(forums))):
                forum = forums[forum_idx_]

                forum_data = get_forum_data(driver, forum["link"])

                res_data.append(
                    {
                        "category": main_form["category"],
                        "forum": main_form["forum"],
                        "posts": main_form["posts"],
                        "subforum": subform,
                        "forum_name": forum["text"],
                        "forum_link": forum["link"],
                        "forum_data": forum_data,
                    }
                )
            sf_no_slashes = subform.replace("/", "-")
            os.makedirs(f"data/{sf_no_slashes}/", exist_ok=True)
            with open(f"data/{sf_no_slashes}/data.json", "w") as f:
                json.dump(res_data, f, indent=4)


def main():
    global MAIN_URL
    driver = setup_firefox_headless()
    parse_end_to_end(driver, MAIN_URL)
    driver.close()


if __name__ == "__main__":
    main()
