from selenium import webdriver
import time
import os
import requests
import pandas as pd

MAIN_URL = "https://med.over.net/forum"


def setup_firefox_headless() -> webdriver.Firefox:
    firefox_options = webdriver.FirefoxOptions()
    # firefox_options.add_argument("--headless")
    driver = webdriver.Firefox(options=firefox_options)

    return driver

# TASK1
def get_main_forums(driver: webdriver.Firefox, url: str):
    def accept_cookies(driver: webdriver.Firefox):
        accept_cookies_button = driver.find_element(
            by="xpath", value="//button[@id='didomi-notice-agree-button']"
        )
        accept_cookies_button.click()

    def get_headlines(driver: webdriver.Firefox) -> list[dict]:
        headlines = driver.find_elements(
            by="xpath", value="//div[@class='headline-table']"
        )
        hls = []

        for headline in headlines:
            # Find element with class title title--normal
            title = headline.find_element(
                by="xpath", value=".//div[@class='title title--normal']"
            )
            url = title.find_element(
                by="xpath", value="//a[@class='button']"
            ).get_attribute("href")

            hls.append({"url": url, "title": title.text})

        return hls

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
    time.sleep(2)
    accept_cookies(driver)
    time.sleep(2)
    headlines = get_headlines(driver)

    res = []
    for headline in headlines:
        res.extend(get_sub_forums(driver, headline))

    return res

## TASK 2
def get_subforum_subforum(
    driver: webdriver.Firefox,
    url="https://med.over.net/forum/kategorija/dusevno-zdravje-in-odnosi/druzina-3574102/",
):
    def get_tables(driver: webdriver.Firefox):
        data = []
        tables = driver.find_elements(
            by="xpath", value="//div[@class='table__outer table__outer--margin']"
        )

        for table in tables:
            table_elements = table.find_elements(
                by="xpath", value="//tr[@class='table-forum__row']"
            )
            for elem in table_elements:
                elem_name = elem.find_element(by="xpath", value=".//a").text
                elem_url = elem.find_element(by="xpath", value=".//a").get_attribute(
                    "href"
                )
                # Get the second element from table data
                elem_answers = elem.find_elements(by="xpath", value=".//td")[1].text
                elem_views = elem.find_elements(by="xpath", value=".//td")[2].text
                elem_last_post = elem.find_elements(by="xpath", value=".//td")[3].text

                data.append(
                    {
                        "topic": elem_name,
                        "answers": elem_answers,
                        "views": elem_views,
                        "link": elem_url,
                        "last-activity": elem_last_post,
                    }
                )
            return data

    driver.get(url)
    time.sleep(2)

    data = get_tables(driver)
    return data

## TASK 3
def get_forum_data(
    driver: webdriver.Firefox,
    url="https://med.over.net/forum/tema/dodatek-za-nego-otroka-ki-potrebuje-posebno-nego-in-varstvo-nego-2069589/",
):
    driver.get(url)
    time.sleep(2)

    inner_page = driver.find_element(
        by="xpath", value="//article[@class='page__inner']"
    )

    user = inner_page.find_element(
        by="xpath", value="//span[@class='title title--xsmall title--author']"
    ).text
    date = inner_page.find_element(
        by="xpath", value="//span[@class='text text--small u-light']"
    ).text
    content = inner_page.find_element(
        by="xpath", value="//div[@class='forum-post__content']"
    ).text.strip()

    return {"user": user, "time": date, "content": content}

## TASK 4
def parse_end_to_end(driver: webdriver.Firefox, url: str):
    main_forms = get_main_forums(driver, url)

    res_data = []

    for main_form in main_forms:
        sub_forums = get_subforum_subforum(driver, main_form["link"])

        for sub_forum in sub_forums[:5]:
            fd = get_forum_data(driver, sub_forum["link"])

            res = {
                "category": main_form["category"],
                "forum": main_form["forum"],
                "posts-number": main_form["posts"],
                "topic": sub_forum["answers"],
                "topic-views": sub_forum["views"],
                "topic-last-activity": sub_forum["last-activity"],
                "user": fd["user"],
                "user-post-time": fd["time"],
                "user-content": fd["content"],
            }

            res_data.append(res)

    # Write to pandas dataframe
    df = pd.DataFrame(res_data)
    df.to_csv("data.csv", index=False)


def main():
    global MAIN_URL
    driver = setup_firefox_headless()
    parse_end_to_end(driver, MAIN_URL)

    driver.close()


if __name__ == "__main__":
    main()
