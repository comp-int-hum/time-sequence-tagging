import argparse
import re
import os
import gzip
import csv
import logging
import json


logger = logging.getLogger("compute_dataset_statistics")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Name of input file")
    parser.add_argument("--output", dest="output", help="Name of output file")
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
    
    logger.info("Scanning dataset")
    titles = {}
    authors = {}
    years = {}
    loc_codes = {}
    types = {}
    subjects = {}
    shelves = {}
    
    # Language Type Title Authors;[] Subjects; LoCC; Bookshelves;
    with gzip.open(args.input, "rt") as ifd:
        for line in ifd:
            year = None
            j = json.loads(line)
            for author in j["Authors"].split(";"):
                author = author.strip()
                m = re.match(r"^(?P<name>.+?)(((?P<birth>\d+)\??)?-((?P<death>\d+)\??)?)?(?:\s+\[(?P<role>.*?)\])?$", author)
                if m.group("name") and not m.group("role") and (m.group("birth") or m.group("death")):
                    name = m.group("name").strip()
                    if m.group("birth"):
                        birth = int(m.group("birth"))
                        if m.group("death"):
                            death = int(m.group("death"))
                            year = birth + (death - birth) / 2
                        else:
                            year = birth + 30
                    elif m.group("death"):
                        year = death - 30
                    authors[name] = authors.get(name, 0) + 1
            for subject in j["Subjects"].split(";"):
                subject = subject.strip()
                subjects[subject] = subjects.get(subject, 0) + 1
            for loc_code in j["LoCC"].split(";"):
                loc_code = loc_code.strip()
                loc_codes[loc_code] = loc_codes.get(loc_code, 0) + 1
            for shelf in j["Bookshelves"].split(";"):
                shelf = shelf.strip()
                shelves[shelf] = shelves.get(shelf, 0) + 1
            titles[j["Title"]] = titles.get(j["Title"], 0) + 1
            types[j["Type"]] = types.get(j["Type"], 0) + 1
            year = int(year) if year else year
            years[year] = years.get(year, 0) + 1

    with open(args.output, "wt") as ofd:
        for name, counts in [
                ("Authors", authors),
                ("Years", list(sorted(years))),
                ("Types", types),
                ("LoC Codes", loc_codes),
                ("Subjects", subjects),
                ("Shelves", shelves),
                #("Titles", titles),
        ]:
            ofd.write(name + "\n")
            if isinstance(counts, list):
                pass
            else:
                pass
            ofd.write("\n\n")
