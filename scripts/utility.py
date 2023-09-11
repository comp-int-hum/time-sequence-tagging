import os

def make_dirs(filepath):
	directory = os.path.dirname(filepath)
	if not os.path.exists(directory):
		os.makedirs(directory)