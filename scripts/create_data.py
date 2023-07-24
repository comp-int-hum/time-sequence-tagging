from bs4 import BeautifulSoup
import argparse
import tarfile
import json
import re
# import xml.etree.ElementTree as ET

def get_form(fp):
    soup = BeautifulSoup(fp, features = "xml")
    # only considering body (front, body, back)'
    # note: there may be more body elements embedded; ex: a letter included within a novel
    body = soup.find('body')
    form = None
    if body:
        form = body.get("type")
        if not form:
            # --Sanity Check--:
            # first_div = body.find("div")
            # first_div_type = body.find("div", attrs={"type":True})
            # if first_div != first_div_type:
            #     print(f"First div: {first_div}")
            #     print(f"Sanity check: {sanity_check}")
            div = body.find("div", attrs={"type":True})
            if div:
                form = div.type

    return form
        
        # # TODO: consider better solution to connecting words between pages``
        # mw_tags = soup.find_all('mw')
        # for mw_tag in mw_tags:
        #     mw_tag.extract()

def get_metadata(fp):
    soup = BeautifulSoup(fp, features="xml")
    author = soup.author.persName.string
    title = soup.title
    edition = soup.edition
    imprint_year = soup.imprint.date
    publisher = soup.publisher.findall('persName')
    print(f"Title: {title} author: {author}")
    assert(edition and imprint_year and publisher)





def segment_paragraphs(soup): # passing in bs4 soup object
    paragraph_dict = {} # key=paragraph num, value=paragraph text

    if soup is not None:
        chp_num = 0
        for paragraph in soup.find_all('p'):
            paragraph_text = paragraph.get_text().strip()
            paragraph_text = re.sub(r'\s+', ' ', paragraph_text)
            data_dict[chp_num] = paragraph_text
            chp_num+=1

    return paragraph_dict

def segment_chapters(soup):

    paragraph_dict = {} # key=paragraph num, value=paragraph text


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", dest="outputs", help="Output files")
    parser.add_argument("--data_path", dest="data_path", nargs="+", help="path to data files")
    parser.add_argument("--granularity", dest="granularity", choices=["chapter", "paragraph"], help="chapter or paragraph")
    args, rest = parser.parse_known_args()

    if tarfile.is_tarfile(args.data_path[0]):
        print("Found tarfile")
        for tar_name in args.data_path:
            tar = tarfile.open(tar_name)
            for member in tar.getmembers():
                if not member.isdir():
                    file = tar.extractfile(member)
                    assert(file)
                    get_metadata(file)

    else:
        print("Using regular files")
        with open(args.data_path[0], 'r') as fp:
            soup = BeautifulSoup(fp, features="xml") # file, parser
            body = soup.find('body')
            print(type(body))
    
    data_dict = {} # Used for json dump at end

    print(f"Data path: {args.data_path[0]}")
    
    
    # TODO: hink more about resolution to this
    
            # data_dict[str(i)] = paragraph_text
            # print(data_dict[str(i)])
            # i+=1

    
    # with open(args.outputs, "w") as output:
    #     json.dump(data_dict, output)

    
    # text_content = soup.get_text()
    # print(text_content)
    # for result in soup.find_all('p'):
    #     print(result)


    ## Alternative for Testing

    # data_dict = {}

    # for tar_name in args.data_path:
    #     tar = tarfile.open(tar_name)
    #     for member in tar.getmembers():
    #         namespaces = {} # name-space for current xml file
    #         file = tar.extractfile(member)
    #         tree = ET.parse(file)
    #         root = tree.getroot() # get root of xml element tree
    #         namespaces["def"] = root.tag.split("}")[0][1:] # add default namespace based on root, which is in the format {uri}tag
    #         print(namespaces)
    #         titles = root.findall('.//def:p', namespaces=namespaces)
    #         for title in titles:
    #             print(title.text)
            


