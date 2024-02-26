from bs4 import BeautifulSoup

if __name__ == "__main__":
    test_str = '<div> <p> Some text <a id="href_end">Anchor 1</a></p> <p>Other text <a id="another_id">Anchor 2</a></p> </div>'
    soup = BeautifulSoup(test_str, "html.parser")

    curr = soup.find('p')
    while curr:
        print(f"Curr: {curr}")
        curr = curr.find_next()