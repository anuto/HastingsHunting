# libraries
import nltk			# for language processing
import matplotlib	# for graphing
import math			# for sqrt functions

##### only one of these should be set to True. #####
MENDENHALL = False # stylometric "fingerprint"
KILGARIFF = False  # find chi-squared distance between <possible author> and
				   # the unknown doc. Minimize across all possible authors.
BURROWS = True     # find the delta between an arbitrary number of authors,
				   # and an unknown document

# who wrote which papers.
papers = {
	'Madison': [10, 14, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
	'Hamilton': [1, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 21, 22, 23, 24, 
				25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 59, 60,
				61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 
				78, 79, 80, 81, 82, 83, 84, 85],
	'Jay': [2, 3, 4, 5],
	'Shared': [18, 19, 20],
	'Disputed': [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 62, 63],
	'TestCase': [64]
}

if MENDENHALL:
	authors = ("Hamilton", "Madison", "Disputed", "Jay", "Shared")

if KILGARIFF:
	source_authors = ("Hamilton", "Madison", "Disputed")
	authors = ("Hamilton", "Madison")
	##### consider the top N frequent words. Changeable. #####
	N = 500

if BURROWS:
	authors = ("Hamilton", "Madison", "Jay", "Disputed", "Shared")
	##### consider the top N frequent words. Changeable. #####
	N = 500

# take in a list of numbers, return a yuge string of all the texts
# in ./data/federalist_#.txt. File bodies separated by a linebreak.
def read_files_into_string(filenames):
	strings = []
	for filename in filenames:
		with open(f'./data/federalist_{filename}.txt') as f:
			strings.append(f.read())
	return '\n'.join(strings)

# makes graphs of 'stylistic fingerprints' of the distribution of
# word lengths. 
def mendenhall(federalist_by_author):
	federalist_by_author_tokens = {}
	federalist_by_author_length_distributions = {}
	for author in authors:
		tokens = nltk.word_tokenize(federalist_by_author[author])

		# Filter out punctuation // it's also filtering tokens with no alphabetic character (14, 1)
		federalist_by_author_tokens[author] = ([token for token in tokens 
												if any(c.isalpha() for c in token)])
		# Get a distribution of token lengths
		token_lengths = [len(token) for token in federalist_by_author_tokens[author]]
		federalist_by_author_length_distributions[author] = nltk.FreqDist(token_lengths)
		federalist_by_author_length_distributions[author].plot(15,title=author)  

def kilgariff(federalist_by_author):
	federalist_by_author_tokens = {}
	for author in source_authors:
		tokens = nltk.word_tokenize(federalist_by_author[author])

		# Filter out punctuation // it's also filtering tokens with no alphabetic character (14, 1)
		federalist_by_author_tokens[author] = ([token for token in tokens 
												if any(c.isalpha() for c in token)])

	# Lowercase the tokens so that the same word, capitalized or not, 
	# counts as one word
	for author in authors:
		federalist_by_author_tokens[author] = (
			[token.lower() for token in federalist_by_author_tokens[author]])
	federalist_by_author_tokens["Disputed"] = (
		[token.lower() for token in federalist_by_author_tokens["Disputed"]])

	# Calculate chisquared for each of the two candidate authors
	for author in authors:
		# First, build a joint corpus and identify the N most frequent words in it
		joint_corpus = (federalist_by_author_tokens[author] +
			federalist_by_author_tokens["Disputed"])
		joint_freq_dist = nltk.FreqDist(joint_corpus)
		most_common = list(joint_freq_dist.most_common(N))

		# What proportion of the joint corpus is made up 
		# of the candidate author's tokens?
		author_share = (len(federalist_by_author_tokens[author]) / len(joint_corpus))

		# Now, let's look at the 500 most common words in the candidate 
		# author's corpus and compare the number of times they can be observed 
		# to what would be expected if the author's papers 
		# and the Disputed papers were both random samples from the same distribution.
		chisquared = 0
		for word,joint_count in most_common:
			# How often do we really see this common word?
			author_count = federalist_by_author_tokens[author].count(word)
			disputed_count = federalist_by_author_tokens["Disputed"].count(word)

			# How often should we see it?
			expected_author_count = joint_count * author_share
			expected_disputed_count = joint_count * (1-author_share)

			# Add the word's contribution to the chi-squared statistic
			chisquared += ((author_count-expected_author_count) * 
				(author_count-expected_author_count) / expected_author_count)

			chisquared += ((disputed_count-expected_disputed_count) *
				(disputed_count-expected_disputed_count)
				/ expected_disputed_count)

		print("The Chi-squared statistic for candidate", author, "is", chisquared)

def burrows(federalist_by_author):
	print("N: ", N)
	federalist_by_author_tokens = {}
	for author in authors:
		tokens = nltk.word_tokenize(federalist_by_author[author])

		# Filter out punctuation // it's also filtering tokens with no alphabetic character (14, 1)
		federalist_by_author_tokens[author] = ([token for token in tokens 
												if any(c.isalpha() for c in token)])
	for author in authors:
		federalist_by_author_tokens[author] = (
			[token.lower() for token in federalist_by_author_tokens[author]])

	# Combine every paper except our test case into a single corpus
	whole_corpus = []
	for author in authors:
		whole_corpus += federalist_by_author_tokens[author]
		# Get a frequency distribution
	whole_corpus_freq_dist = list(nltk.FreqDist(whole_corpus).most_common(N))
	whole_corpus_freq_dist[ :10 ]
	captured = 0
	for k, v in whole_corpus_freq_dist:
		# print(k, " ", v)
		captured += v
	print("whole corpus: ", len(whole_corpus))
	print("percentage captured: ", ((captured / len(whole_corpus)) * 100))

	# The main data structure
	features = [word for word,freq in whole_corpus_freq_dist]
	feature_freqs = {}

	for author in authors:
		# A dictionary for each candidate's features
		feature_freqs[author] = {} 

		# A helper value containing the number of tokens in the author's subcorpus
		overall = len(federalist_by_author_tokens[author])

		# Calculate each feature's presence in the subcorpus
		for feature in features:
			presence = federalist_by_author_tokens[author].count(feature)
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
		for author in authors:
			feature_average += feature_freqs[author][feature]
		feature_average /= len(authors)
		corpus_features[feature]["Mean"] = feature_average

		# Calculate the standard deviation using the basic formula for a sample
		feature_stdev = 0
		for author in authors:
			diff = feature_freqs[author][feature] - corpus_features[feature]["Mean"]
			feature_stdev += diff*diff
		feature_stdev /= (len(authors) - 1)
		feature_stdev = math.sqrt(feature_stdev)
		corpus_features[feature]["StdDev"] = feature_stdev

	feature_zscores = {}
	for author in authors:
		feature_zscores[author] = {}
		for feature in features:
			# Z-score definition = (value - mean) / stddev
			# We use intermediate variables to make the code easier to read
			feature_val = feature_freqs[author][feature]
			feature_mean = corpus_features[feature]["Mean"]
			feature_stdev = corpus_features[feature]["StdDev"]
			feature_zscores[author][feature] = ((feature_val-feature_mean) / feature_stdev)

	# Tokenize the test case
	testcase_tokens = nltk.word_tokenize(federalist_by_author["TestCase"])

	# Filter out punctuation and lowercase the tokens
	testcase_tokens = [token.lower() for token in testcase_tokens if any(c.isalpha() for c in token)]

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

	# for author in authors:
	# 	delta = 0
	# 	for feature in features:
	# 		delta += math.fabs((testcase_zscores[feature] - feature_zscores[author][feature]))
	# 	delta /= len(features)
	# 	print( "Delta score for candidate", author, "is", delta )

		# calculate cosine distance
	for author in authors:
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
		cosine_similarity = numerator / (math.sqrt(denominator_b) * math.sqrt(denominator_a))
		print("cosine similarity for candidate ", author,  " is ", cosine_similarity)


def main():
	federalist_by_author = {}  
	for author, files in papers.items():
		federalist_by_author[author] = read_files_into_string(files)

	##### verify working #####
	# for author in papers:
	# 	print(author)
	# 	print(federalist_by_author[author][:20])
	# 	print()

	##### mendenhall #####
	# Transform the authors' corpora into lists of word tokens
	if MENDENHALL:
		mendenhall(federalist_by_author)

	##### kilgariff #####
	# note: This assumes that a person’s vocabulary and word usage patterns are relatively constant. 
	# PROBABLY a false assumption.
	if KILGARIFF:
		kilgariff(federalist_by_author)

	##### burrows #####
	# supposedly quite prominent to this day!
	# "The Delta Method is designed to compare an anonymous text (or set of texts) to many 
	# 		different authors’ signatures at the same time." WOOT
	# equal weight to all features to avoid common word overwhelming (an issue with chi squared)
	#	but perhaps we should extra weight something inversely to common words
	if BURROWS:
		burrows(federalist_by_author)


if __name__ == '__main__':
	main()