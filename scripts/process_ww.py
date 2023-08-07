from bs4 import BeautifulSoup
import argparse
import tarfile
import json
import re
# import xml.etree.ElementTree as ET

def get_form(soup):
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
                form = div.get("type")
                
    return form
        
    

def get_metadata(soup):

    data = {} 
    data["title"] = soup.title.string
    data["author"] = soup.author.persName.string if soup.author else None
    data["edition"] = soup.edition.string if soup.edition else None
    imprint = soup.imprint
    data["pub_info"] = None
    if imprint:
        pub_info = {}
        pub_info["publisher"] = [pers.string for pers in imprint.publisher.find_all('persName')] if imprint.publisher else []
        pub_info["pub_place"] = imprint.pubPlace.string
        pub_info["imprint_year"] = imprint.date.string
        data["pub_info"] = pub_info
    data["form"] = get_form(soup)
    return data





def segment_paragraphs(soup):

    paragraph_dict = {} # key=paragraph num, value=paragraph text

    body = soup.find('body')

    # Removing marked words of type "catch" and "pageNum"
    mw_tags = body.find_all('mw', type="catch") + body.find_all('mw', type='pageNum')
    for mw_tag in mw_tags:
        mw_tag.extract()


    if body is not None:
        for pnum, paragraph in enumerate(body.find_all('p')):
            paragraph_text = paragraph.get_text().strip()
            paragraph_text = re.sub(r'\s+', ' ', paragraph_text)
            paragraph_dict[pnum] = paragraph_text  

    return paragraph_dict

def segment_chapters(soup):

    chapter_dict = {}

    body = soup.find('body')

    # Removing marked words of type "catch" and "pageNum"
    mw_tags = body.find_all('mw', type="catch") + body.find_all('mw', type='pageNum')
    for mw_tag in mw_tags:
        mw_tag.extract()

    if body is not None:
        for cnum, div in enumerate(body.find_all('div', {'type':'chapter'})):
            paragraph_dict = {}
            for pnum, paragraph in enumerate(div.find_all('p')):
                paragraph_text = paragraph.get_text().strip()
                paragraph_text = re.sub(r'\s+', ' ', paragraph_text)
                paragraph_dict[pnum] = paragraph_text
            chapter_dict["ch" + str(cnum)] = paragraph_dict
    return chapter_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", dest="output", help="Output files")
    parser.add_argument("--data_path", dest="data_path", nargs="+", help="path to data files")
    parser.add_argument("--granularity", dest="granularity", choices=["chapter", "paragraph"], help="chapter or paragraph")
    args, rest = parser.parse_known_args()

    result = []
    if tarfile.is_tarfile(args.data_path[0]):
        print("Found tarfile")
        for tar_name in args.data_path:
            tar = tarfile.open(tar_name)
            for member in tar.getmembers():
                if not member.isdir():
                    fp = tar.extractfile(member)
                    soup = BeautifulSoup(fp, features="xml")
                    print(f"Name: {member.name}, form: {get_form(soup)}")
                    data = get_metadata(soup)
                    if data["form"] == "chapter": # non-ideal hard coding
                        if args.granularity == "paragraph":
                           data["segments"] = segment_paragraphs(soup)
                        else:
                           data["segments"] = segment_chapters(soup)
                        result.append(data)

    else:
        print("Using regular files")
        print(args.data_path)
        for curr_file in args.data_path:
            with open(curr_file, 'r') as fp:
                soup = BeautifulSoup(fp, features="xml") # file, parser
                print(f"Name: {curr_file}, form: {get_form(soup)}")


    print(f"RESULT LENGTH: {len(result)}")
    
    with open(args.output, "w") as output:
        json.dump(result, output)

    
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
            


