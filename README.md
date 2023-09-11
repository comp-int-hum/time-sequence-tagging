To run:
	pip install -r requirements.txt
	scons -Q

Current pipeline:
- Collect data from Project Gutenberg and/or Women Writers Project
- Sample data from PG and WW. Create a file of data_size.
- Feed this file into encoder
- Shuffle and split encoded data into train and test. If experimenting with same domain (i.e. train with PG, test with PG), one file will be passed into shuffle_data. Otherwise, for cross-domain, two files will be passed in. The combined train + test size will be at most max_size
- Train model based on train/test files

Note about data parsing for Women Writers Project:
- Only considering chapters/paragraphs under the outermost <body> element (all docs have a front, body, and back)
- Only consider texts that have type = "chapter" in the outermost <body> element or the next div