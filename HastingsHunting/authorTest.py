import nltk
import numpy as np
import math
# from scipy import sparse
# adverbs
# lengths
# frequency
# ngram

N = 500

def main():
	mam = open("manAndManners.txt", "r").read() # probably hastings
	with open("bla.txt", encoding="utf8", errors='ignore') as f:
		lit = f.read()
	ff = open("ff.txt", "r").read() # definitely hastings

	# mam_adverbs = adverbs(mam)
	# lit_adverbs = adverbs(lit)
	# ff_adverbs = adverbs(ff)

	# all_adverbs = get_all_adverbs(mam_adverbs, lit_adverbs, ff_adverbs)

	# mam_vector = make_vector(mam_adverbs, all_adverbs)
	# lit_vector = make_vector(lit_adverbs, all_adverbs)
	# ff_vector = make_vector(ff_adverbs, all_adverbs)

	# distance(mam_vector, ff_vector)
	# distance(mam_vector, lit_vector)

	# chi_squared(mam, ff, 'hastings')
	# chi_squared(mam, lit, 'not hastings')
	print("N: ", N)
	author_docs = {'hastings': ff, 'orage': lit}
	delta(author_docs, mam)

def delta(author_docs, disputed):

	author_tokens = {}
	for author in author_docs.keys():
		tokens = nltk.word_tokenize(author_docs[author])
		# Filter out punctuation // it's also filtering tokens with no alphabetic character (14, 1)
		author_tokens[author] = ([token for token in tokens 
												if any(c.isalpha() for c in token)])
	for author in author_docs.keys():
		author_tokens[author] = (
			[token.lower() for token in author_tokens[author]])

	# Combine every paper except our test case into a single corpus
	whole_corpus = []
	for author in author_docs.keys():
		whole_corpus += author_tokens[author]
		# Get a frequency distribution
	whole_corpus_freq_dist = list(nltk.FreqDist(whole_corpus).most_common(N))
	whole_corpus_freq_dist[ :10 ]
	captured = 0
	for k, v in whole_corpus_freq_dist:
		# print(k, " ", v)
		captured += v
	print("combined corpus size: ", len(whole_corpus))
	print("percentage captured: ", ((captured / len(whole_corpus)) * 100))

	# The main data structure
	features = [word for word,freq in whole_corpus_freq_dist]
	feature_freqs = {}

	for author in author_docs.keys():
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
		for author in author_docs.keys():
			feature_average += feature_freqs[author][feature]
		feature_average /= len(author_docs)
		corpus_features[feature]["Mean"] = feature_average

		# Calculate the standard deviation using the basic formula for a sample
		feature_stdev = 0
		for author in author_docs.keys():
			diff = feature_freqs[author][feature] - corpus_features[feature]["Mean"]
			feature_stdev += diff*diff
		feature_stdev /= (len(author_docs) - 1)
		feature_stdev = math.sqrt(feature_stdev)
		corpus_features[feature]["StdDev"] = feature_stdev

	feature_zscores = {}
	for author in author_docs.keys():
		feature_zscores[author] = {}
		for feature in features:
			# Z-score definition = (value - mean) / stddev
			# We use intermediate variables to make the code easier to read
			feature_val = feature_freqs[author][feature]
			feature_mean = corpus_features[feature]["Mean"]
			feature_stdev = corpus_features[feature]["StdDev"]
			feature_zscores[author][feature] = ((feature_val-feature_mean) / feature_stdev)

	# Tokenize the test case
	testcase_tokens = nltk.word_tokenize(disputed)

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

	for author in author_docs.keys():
		delta = 0
		for feature in features:
			delta += math.fabs((testcase_zscores[feature] - feature_zscores[author][feature]))
		delta /= len(features)
		print( "Delta score for candidate", author, "is", delta )

def chi_squared(disputed, definite, author_name):
	author_tokens = {}
	tokens = nltk.word_tokenize(disputed)

	# Filter out punctuation // it's also filtering tokens with no alphabetic character (14, 1)
	author_tokens["Disputed"] = ([token for token in tokens 
											if any(c.isalpha() for c in token)])

	tokens = nltk.word_tokenize(definite)

	# Filter out punctuation // it's also filtering tokens with no alphabetic character (14, 1)
	author_tokens["Definite"] = ([token for token in tokens 
											if any(c.isalpha() for c in token)])

	# Lowercase the tokens so that the same word, capitalized or not, 
	# counts as one word
	author_tokens["Definite"] = (
		[token.lower() for token in author_tokens["Definite"]])
	author_tokens["Disputed"] = (
		[token.lower() for token in author_tokens["Disputed"]])


	# First, build a joint corpus and identify the N most frequent words in it
	joint_corpus = (author_tokens["Definite"] + author_tokens["Disputed"])
	joint_freq_dist = nltk.FreqDist(joint_corpus)
	most_common = list(joint_freq_dist.most_common(N))

	# What proportion of the joint corpus is made up 
	# of the candidate author's tokens?
	author_share = (len(author_tokens["Definite"]) / len(joint_corpus))

	# Now, let's look at the 500 most common words in the candidate 
	# author's corpus and compare the number of times they can be observed 
	# to what would be expected if the author's papers 
	# and the Disputed papers were both random samples from the same distribution.
	chisquared = 0
	for word,joint_count in most_common:
		# How often do we really see this common word?
		author_count = author_tokens["Definite"].count(word)
		disputed_count = author_tokens["Disputed"].count(word)

		# How often should we see it?
		expected_author_count = joint_count * author_share
		expected_disputed_count = joint_count * (1-author_share)

		# Add the word's contribution to the chi-squared statistic
		chisquared += ((author_count-expected_author_count) * 
			(author_count-expected_author_count) / expected_author_count)

		chisquared += ((disputed_count-expected_disputed_count) *
			(disputed_count-expected_disputed_count)
			/ expected_disputed_count)

	print("The Chi-squared statistic for candidate", author_name, "is", chisquared)

def distance(hasting_vec, other_vec):
	if len(hasting_vec) != len(other_vec):
		"dimension mismatch"
	else:
		euclid_sum = 0
		for index in range(len(hasting_vec)):
			euclid_sum += (hasting_vec[index] - other_vec[index])**2
		euclid_sum = math.sqrt(euclid_sum)
		print(euclid_sum)


def make_vector(adverb_counts, all_adverbs):
	vec = []
	for ad in all_adverbs:
		count = adverb_counts.count(ad)
		vec.append(count)
	vec.append(len(adverb_counts))
	print(vec)
	print()
	return vec

def get_all_adverbs(s1, s2, s3):
	all_adverbs = set([])
	for ad in s1:
		all_adverbs.add(ad)
	for ad in s2:
		all_adverbs.add(ad)
	for ad in s3:
		all_adverbs.add(ad)
	return all_adverbs

def adverbs(work):
	words = nltk.word_tokenize(work)
	adverbs = []
	for word in words:
		if word.endswith("ly"):
			adverbs.append(word)
	return adverbs
	
# mam = nltk.sent_tokenize(mam)
# mamSentences = []
# for s in mam:
# 	mamSentences.append(s)

# lit = nltk.sent_tokenize(lit)
# litSentences = []
# for s in litSentences:
# 	litSentences.append(s)

# ff = nltk.sent_tokenize(ff)
# ffSentences = []
# for s in ffSentences:
# 	ffSentences.append(s)

if __name__ == '__main__':
	main()
