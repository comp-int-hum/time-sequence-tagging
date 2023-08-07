To run:
	pip install -r requirements.txt
	scons -Q


Note: Code quality of SConstruct is rather horrific at the moment. I will fix it soon.

Some choices:
- Only considering chapters/paragraphs under the outermost <body> element (all docs have a front, body, and back)
- Only consider texts that have type = "chapter" in the outermost <body> element or the next div