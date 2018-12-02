# adapted from https://programminghistorian.org/en/lessons/introduction-to-stylometry-with-python
# code implementing the algorithm especially. 
import nltk
import math
import os

########## THINGS TO PLAY AROUND WITH ##########
# using the top N most frequent words

# False: we consider 'The' different from 'the'
# True: we consider 'The' and 'the' the same word
CASE_INSENSITIVE = False
DISPUTED = 'anon'

# If true, prints the 'word, count' of the top N words
PRINT_TOP_N_WORDS = False

# coming soon...weights!

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
###################################################
def main():
	# for N in range(1, 501, 50):
	N = 60
	author_corps = get_author_corps()

	# to keep our records clean =)
	print("deltas of distance to ", AUTHOR_CODE[DISPUTED])
	print("N: ", N)
	print("Case sensitive? ", CASE_INSENSITIVE)

	author_tokens = tokenize(author_corps)
	disputed_tokens = author_tokens.pop(DISPUTED)
	delta(N, author_tokens, disputed_tokens)

	print()

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

	# TODO: take this out later, it's cutting out the disproportionate amount of orage
	author_corps['alor'] = author_corps['alor'][:int((0.5* len(author_corps['alor'])))]
	return author_corps	

def tokenize(author_corps):
	author_tokens = {}
	
	for author in author_corps:
		tokens = nltk.word_tokenize(author_corps[author])
		# filter out tokens with no alphabetic characters
		# a token is uninterrupted characters. Tokens are separated by whitespace.
		author_tokens[author] = ([token for token in tokens if any(c.isalpha() for c in token)])

	if CASE_INSENSITIVE:
		# Lowercase the tokens so that the same word, capitalized or not, 
		# counts as one word. Although, perhaps, using capitalization might give us other
		# information!
		for author in author_corps:
			author_tokens[author] = (
				[token.lower() for token in author_tokens[author]])

	return author_tokens

def delta(N, author_tokens, testcase_tokens):
	# Combine every paper except our test case into a single corpus
	whole_corpus = []
	for author in author_tokens:
		whole_corpus += author_tokens[author]

	# Get a frequency distribution
	whole_corpus_freq_dist = list(nltk.FreqDist(whole_corpus).most_common(N))

	# print some info about how much the top N words show up
	captured = 0
	i = 1
	for k, v in whole_corpus_freq_dist:
		if PRINT_TOP_N_WORDS:
			print(i, ": ", k, " ", v)
			i += 1
		captured += v
	print("whole corpus: ", len(whole_corpus))
	print("percentage captured: ", ((captured / len(whole_corpus)) * 100), "%")

	features = [word for word,freq in whole_corpus_freq_dist]
	feature_freqs = {}

	for author in author_tokens:
		# A dictionary for each candidate's features
		feature_freqs[author] = {} 

		# A helper value containing the number of tokens in the author's subcorpus
		overall = len(author_tokens[author])

		# Calculate each feature's presence in the subcorpus
		for feature in features:
			presence = author_tokens[author].count(feature)
			feature_freqs[author][feature] = presence / overall

	# The data structure into which we will be storing the "corpus standard" statistics
	corpus_features = {}

	# For each feature...
	for feature in features:
		# Create a sub-dictionary that will contain the feature's mean 
		# and standard deviation
		corpus_features[feature] = {}

		# Calculate the mean of the frequencies expressed in the subcorpora
		feature_average = 0
		for author in author_tokens:
			feature_average += feature_freqs[author][feature]
		feature_average /= len(author_tokens)
		corpus_features[feature]["Mean"] = feature_average

		# Calculate the standard deviation using the basic formula for a sample
		feature_stdev = 0
		for author in author_tokens:
			diff = feature_freqs[author][feature] - corpus_features[feature]["Mean"]
			feature_stdev += diff*diff
		feature_stdev /= (len(author_tokens) - 1)
		feature_stdev = math.sqrt(feature_stdev)
		corpus_features[feature]["StdDev"] = feature_stdev

	feature_zscores = {}
	for author in author_tokens:
		feature_zscores[author] = {}
		for feature in features:
			# Z-score definition = (value - mean) / stddev
			# We use intermediate variables to make the code easier to read
			feature_val = feature_freqs[author][feature]
			feature_mean = corpus_features[feature]["Mean"]
			feature_stdev = corpus_features[feature]["StdDev"]
			feature_zscores[author][feature] = ((feature_val-feature_mean) / feature_stdev)

	
	# Calculate the test case's features
	overall = len(testcase_tokens)
	testcase_freqs = {}
	for feature in features:
		presence = testcase_tokens.count(feature)
		testcase_freqs[feature] = presence / overall

	# Calculate the test case's feature z-scores
	testcase_zscores = {}
	for feature in features:
		feature_val = testcase_freqs[feature]
		feature_mean = corpus_features[feature]["Mean"]
		feature_stdev = corpus_features[feature]["StdDev"]
		testcase_zscores[feature] = (feature_val - feature_mean) / feature_stdev
		# print("Test case z-score for feature", feature, "is", testcase_zscores[feature])

	# calculate delta
	for author in author_tokens:
		delta = 0
		for feature in features:
			delta += math.fabs((testcase_zscores[feature] - feature_zscores[author][feature]))
		delta /= len(features)
		print( "Delta score for candidate", AUTHOR_CODE[author], "is", delta )

	print()
	# calculate cosine distance
	for author in author_tokens:
		numerator = 0
		denominator_a = 0
		denominator_b = 0
		num_features = len(features)
		for feature in features:
			a_i = testcase_zscores[feature] 
			b_i = feature_zscores[author][feature]
			numerator += a_i * b_i
			denominator_a += a_i ** 2
			denominator_b += b_i ** 2
		cosine_similarity = numerator / ((denominator_b ** (1/2)) * (denominator_a ** (1/2)))
		print("cosine similarity for candidate ", AUTHOR_CODE[author], " is ", cosine_similarity)


if __name__ == '__main__':
	main()