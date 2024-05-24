# Takes in list of tuples (chapter_name, chapter) and converts to sequence [(ch_name, pnum)] [(label, sent)]
# def convert_to_sequence(chapters, paragraph):
#     sequence_list = []
#     id_list = []
#     for (ch_name, ch) in chapters:
#         ch_len = len(ch)
#         for pnum in range(ch_len):
#             ptag = 1 if (pnum == 0 or pnum == (ch_len-1)) else 0
#             p_name = "p" + str(pnum)
#             par = ch[p_name]
#             if paragraph:
#                 sequence_list.append((ptag, average_embeddings(par)))
#                 id_list.append((ch_name, p_name))
#             else:
#                 par_len = len(par)
#                 for i, sent in enumerate(par):
#                     stag = 1 if ptag and (i == 0 or i == (par_len - 1)) else 0
#                     sequence_list.append((stag, sent))
#                     id_list.append((ch_name, p_name, i))
                
#     return id_list, sequence_list