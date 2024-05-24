# Input: chapter_dict representing one chapter: key=paragraph_num, value = string paragraph_content
# Output: list of all sentences in the chapter
# def get_chapter_sentences(chapter_dict):
#     all_sent = []
#     paragraph_sentences = get_paragraphs(chapter_dict) # dictionary
#     for sentences in paragraph_sentences.values():
#         all_sent.extend(sentences)
#     return all_sent

# def get_passage(pg_text, incorrect_text):
#     ch_names = incorrect_text["chapters"]
    
#     first_start, first_end = incorrect_text["first_ch"]
    
#     if not ch_names[0] in pg_text["segments"]:
#         return ""
#     first_chapter = get_chapter_sentences(pg_text["segments"][ch_names[0]])
#     first_passage = first_chapter[first_start : first_end]

#     if len(ch_names) == 2:
#         if not ch_names[1] in pg_text["segments"]:
#             return ""
#         second_chapter = get_chapter_sentences(pg_text["segments"][ch_names[1]])
#         second_start, second_end = incorrect_text["second_ch"]
#         second_passage = second_chapter[second_start : second_end]
#         return {"Title": pg_text["title"], "Author": pg_text["author"], "POSITIVE" : " [SEP] ".join(first_passage) + " **** CHAPTER BOUNDARY **** " + " [SEP] ".join(second_passage)}
#     return {"Title": pg_text["title"], "Author": pg_text["author"], "NEGATIVE": " [SEP] ".join(first_passage)}