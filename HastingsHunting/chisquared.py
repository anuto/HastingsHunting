# adapted from https://programminghistorian.org/en/lessons/introduction-to-stylometry-with-python
# code implementing the algorithm especially

import nltk
import os

########## THINGS TO PLAY AROUND WITH ##########
# using the top N most frequent words
N = 500

# the 2 authors we are comparing to the disputed author
AUTHOR_1 = 'anon'
AUTHOR_2 = 'kama'

# the disputed author
DISPUTED = 'gehi'

# True: we consider 'The' different from 'the'
# False: we consider 'The' and 'the' the same word
CASE_SENSITIVE = False

##################################################
# tyler's naming convention, for readability
AUTHOR_CODE = {
	'beha': 'Beatrice Hastings',
	'kama': 'Katharine Mansfield',
	'alor': 'Alfred Orage',
	'anon': 'Anonymous',
	'tkl': 'T.K.L. (Hastings)',
	'almo': 'Alice Morning',
	'gehi': 'George Hirst'
}

# vars for use below
authors_plus_disputed = [AUTHOR_1, AUTHOR_2, DISPUTED]
authors = [AUTHOR_1, AUTHOR_2]

###################################################

def main():
	author_corps = get_author_corps()

	# to keep our records clean =)
	print("chi-squared of distance to ", AUTHOR_CODE[DISPUTED])
	print("N: ", N)
	print("Case sensitive? ", CASE_SENSITIVE)

	author_tokens = tokenize(author_corps)

	# Calculate chisquared for each of the two candidate authors
	for author in authors:
		calculate_chi_squared(author, author_tokens)

def tokenize(author_corps):
	author_tokens = {}
	
	for author in authors_plus_disputed:
		tokens = nltk.word_tokenize(author_corps[author])
		# filter out tokens with no alphabetic characters
		# a token is uninterrupted characters. Tokens are separated by whitespace.
		author_tokens[author] = ([token for token in tokens 
												if any(c.isalpha() for c in token)])

	if CASE_SENSITIVE:
		# Lowercase the tokens so that the same word, capitalized or not, 
		# counts as one word. Although, perhaps, using capitalization might give us other
		# information!
		for author in authors_plus_disputed:
			author_tokens[author] = (
				[token.lower() for token in author_tokens[author]])

	return author_tokens

def get_author_corps():
	author_corps = {}
	for file in os.listdir("./data"):
		if (file.endswith(".txt")):
			# following tyler's naming convention
			author_name = file.split('.')[1]
			# strips windows line endings
			file_contents = open("./data/" + file, encoding="utf8", errors='ignore').read()
			if (author_name not in author_corps):
				author_corps[author_name] = []
			author_corps[author_name].append(file_contents)

	for auth in author_corps:
		author_corps[auth] = '\n'.join(author_corps[auth])

	return author_corps	

def calculate_chi_squared(author, author_tokens):
	# First, build a joint corpus and identify the N most frequent words in it
	joint_corpus = (author_tokens[author] + author_tokens[DISPUTED])
	joint_freq_dist = nltk.FreqDist(joint_corpus)
	most_common = list(joint_freq_dist.most_common(N))

	# What proportion of the joint corpus is made up 
	# of the candidate author's tokens?
	author_share = (len(author_tokens[author]) / len(joint_corpus))

	# Now, let's look at the 500 most common words in the candidate 
	# author's corpus and compare the number of times they can be observed 
	# to what would be expected if the author's papers 
	# and the Disputed papers were both random samples from the same distribution.
	chisquared = 0
	for word,joint_count in most_common:
		# How often do we really see this common word?
		author_count = author_tokens[author].count(word)
		disputed_count = author_tokens[DISPUTED].count(word)

		# How often should we see it?
		expected_author_count = joint_count * author_share
		expected_disputed_count = joint_count * (1-author_share)

		# Add the word's contribution to the chi-squared statistic
		chisquared += ((author_count-expected_author_count) * 
			(author_count-expected_author_count) / expected_author_count)

		chisquared += ((disputed_count-expected_disputed_count) *
			(disputed_count-expected_disputed_count)
			/ expected_disputed_count)

	print("The Chi-squared statistic for candidate", AUTHOR_CODE[author], "is", chisquared)

if __name__ == '__main__':
	main()