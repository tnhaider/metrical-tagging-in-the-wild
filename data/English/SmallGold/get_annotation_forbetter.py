import sys, re
import joblib
import pyphen
import string
from inout.forbetter.corpus import Corpus
from inout.forbetter.poem import Poem
from inout.utils.helper import *
#from hyphenation.syllabifier import Syllabifier
from somajo import SoMaJo
from nltk import bigrams as bi
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from collections.abc import Iterable   # drop `.abc` with Python 2.7 or lower
import nltk
from nltk import word_tokenize
from nltk import StanfordTagger
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from syllabipy.sonoripy import SonoriPy as sonor
#from hyphenation_english.syllabifier import Syllabifier
from mtl.bilstmcrf.Hyphenator_EN import Hyphenator
#syllabifier = Syllabifier()
syllabifier = Hyphenator()

regex_tokenizer = RegexpTokenizer(r'\w+')

tokenizer = SoMaJo("en_PTB", split_camel_case=True)
#syllabifier = Syllabifier()
dic = pyphen.Pyphen(lang='en_GB')


#for sentence in tokenized:
#	for token in sentence:
#		try:
#			tokens.append(str(token.text))
#			tokens.append(str(token.token_class))
#syllabifier.predict(token)

def is_iterable(obj):
    return isinstance(obj, Iterable)

                
def get_caesura_anno(footmeter):
	feetout = []
	meter = re.sub('\|', '', footmeter)
	bimeter = bi([m for m in re.sub('\|', ':', str(footmeter))])
	for a, b in bimeter:
		if a == ':':
			continue
		elif b == ':':
			feetout.append(':')
		else:
			feetout.append('.')
	if len(feetout) < len(meter):
		feetout.append('.')
	return feetout
		
def nltk_syllabify_nn(token):
	#hyphenated = dic.inserted(token)
	syllabified = syllabifier.predict(token)
	#hyphenated = re.sub('·', '-', syllabified)
	hyphenated = manual_correction(syllabified)
	split = hyphenated.split('·')
	return split

def nltk_syllabify_pyphen(token):
	hyphenated = dic.inserted(token)
	#syllabified = syllabifier.predict(token)
	#hyphenated = re.sub('·', '-', syllabified)
	hyphenated = manual_correction(hyphenated)
	split = hyphenated.split('·')
	return split


def nltk_syllabify_sequence(tokenized):
	outsyllables = []
	for token in tokenized:
		syllables = nltk_syllabify(token)
		for syll in syllables:
			outsyllables.append(syll)
	return outsyllables

def somajo_tokenize(linestring):
	tokens = []
	token_classes = []
	tokenized = tokenizer.tokenize_text([linestring])
	for sentence in tokenized:
		for token in sentence:
			tokens.append(str(token.text))
			token_classes.append(str(token.token_class))
	return tokens, token_classes

def nltk_tokenize(linestring):
        text_tok = nltk.word_tokenize(linestring)
        text_tok = concatenate_words(text_tok)
        return text_tok

def regex_tokenize(linestring):
        text_tok = regex_tokenizer.tokenize(linestring)
        text_tok = concatenate_words(text_tok)
        return text_tok

def nltk_pos_tag(tokenized):
        pos_seq = []
        # print(text_tok)
        pos_tagged = nltk.pos_tag(tokenized)
        # print the list of tuples: (word,word_class)
        #print(pos_tagged)

        # for loop to extract the elements of the tuples in the pos_tagged list
        # print the word and the pos_tag with the underscore as a delimiter
        for word,word_class in pos_tagged:
                #print(word + "_" + word_class)
                pos_seq.append(word_class)
        return pos_seq



def align_syllables(linetext, footmeter, footreal, syllabify='nn', real=False):
	syllable_position = []
	syll_seq = []
	pos_syll = []
	simple_pos_syll = []
	#print(linetext)
	tokens, token_classes = somajo_tokenize(linetext)
	#tokens = nltk_tokenize(linetext)
	#print(tokens)
	pos_seq = nltk_pos_tag(tokens)
	try:
		feet = get_caesura_anno(footmeter)
		#print(footmeter)
		meter = re.sub('\|', '', footmeter)
		meter = re.sub('s', '+', meter)
		meter = re.sub('w', '-', meter)
		meter = re.sub('\^', '-', meter)
		versemeasure = get_versification(meter)
		measure = versemeasure.split('.')[0]
	except TypeError:
		return None
	#clean_syllables = nltk_syllabify_sequence(regex_tokenize(linetext))
	#pos_seq = get_pos_sequence(pos_model, tokens)
	for token, token_c, pos in zip(tokens, token_classes, pos_seq):
		#token_c = ''
		#if token in string.punctuation:
		#	token_c = 'symbol'
		#else: token_c = 'regular'
		try:
			#print(tokenclass)
			#hyphenated = dic.inserted(word)
			#hyphenated = re.sub("-'", "'", hyphenated)
			if syllabify == 'nn':
				syllables = nltk_syllabify_nn(token)
			else:
				syllables = nltk_syllabify_pyphen(token)
			#syllables = syllabified.split('·') 
			#print(syllabified)
			#print(tokenclass)
			if token_c == 'regular':
				syll_position = 0
				for s in syllables:
					if len(s) > 0:
						syll_position += 1
						if len(syllables) == 1:
							syllable_position.append(0)
						else:
							syllable_position.append(syll_position)
						syll_seq.append(s)
						pos_syll.append(pos)
		except IndexError:
				continue

	syll_seq = concatenate_words(syll_seq)

	print('DEBUG', len(meter) == len(syll_seq), len(meter), len(syll_seq), meter, get_versification(meter),  syll_seq, pos_syll, syllable_position, linetext)
	if len(meter) != len(syll_seq) and syllabify=='nn':# and len(real) != len(syllables):
		pass
		print()
		print('NN', len(meter) == len(syll_seq), len(meter), len(syll_seq), meter, get_versification(meter),  syll_seq, pos_syll, syllable_position, linetext)
	if len(meter) != len(syll_seq) and syllabify=='pyphen':# and len(real) != len(syllables):
		pass
		print()
		print('PYH', len(meter) == len(syll_seq), len(meter), len(syll_seq), meter, get_versification(meter),  syll_seq, pos_syll, syllable_position, linetext)
	if len(meter) != len(syll_seq):# and len(real) != len(syllables):
		if real == False and syllabify=='nn':
			return align_syllables(linetext, footreal, footmeter, syllabify='nn', real=True)
		if real == True and syllabify == 'nn':
			return align_syllables(linetext, footmeter, footreal, syllabify='pyphen', real=False)
		if real == False and syllabify=='pyphen':
			return align_syllables(linetext, footreal, footmeter, syllabify='pyphen', real=True)
		#if real == True and syllabify=='pyphen':
		#	return align_syllables(linetext, footmeter, footreal, syllabify='pyphen', real=False)

	if len(meter) == len(syll_seq):
		if syllabify == 'pyphen':
			print('PHY_FIXED')
			pass
		print(versemeasure, linetext)
		return zip(range(1, len(meter)+1), syll_seq, meter, feet, pos_syll, syllable_position, [measure]*len(meter), [versemeasure]*len(meter), [meter]*len(meter))
	#elif len(real) == len(syll_seq):
	#	return zip(range(1, len(real)+1), syll_seq, real, pos_syll, syllable_position, [measure]*len(real), [versemeasure]*len(real), [real]*len(real))
			#return '\t'.join([str(m) for m in i])
			#outfile.write('\n')
	else: return None


	

if __name__ == '__main__':
	feature_lines = []
	counter = 0
	corpuspath = sys.argv[1]
	#posmodelpath = sys.argv[2]
	#outfile = open('german.meter.multi.measure.tsv', 'w')
	#pos_model = joblib.load(posmodelpath)
	c = Corpus(corpuspath)
	poems = c.read_poems()
	#print(poems)
	wrong = 0
	right = 0
	comment = 0
	measures = []
	for poem in poems:
		for stanza in poem.get_stanzas():
			for line in stanza.get_line_objects():
				footreal = line.get_real()
				footmeter = line.get_meter()
				#print(footmeter)
				if not footmeter:
					continue
				#meter = re.sub('\|', '', footmeter)
				#versemeasure = get_versification(meter)
				linetext = line.get_text()
				linetext = re.sub('\n', ' ', linetext)
				#linetext = re.sub('\t', ' ', linetext)
				linetext = linetext.strip()
				linetext = " ".join(linetext.split())
				linetext = normalize_characters(linetext)
				#print(meter, '\t', versemeasure, '\t', linetext)
	

				feature_line = align_syllables(linetext, footmeter, footreal)
	
				if feature_line is not None and is_iterable(feature_line):
					right +=1
					feature_lines.append(feature_line)
				else: wrong +=1

	print('Wrong', wrong, 'Right', right, 'Comment', comment)
	#print(Counter(measures))

	#	counter += 1
	#	#if counter > 5:
	#	#	continue
	#	for stanza in poem.get_stanzas():
	#		for line in stanza.get_line_objects():
	#			feature_line = get_features(line, pos_model)
	#			if feature_line is not None and is_iterable(feature_line):
	#				feature_lines.append(feature_line)
	shuffle(feature_lines)
	y = ['1']*len(feature_lines)
	X_train, X_testval, y_train, y_testval = train_test_split(feature_lines, y, test_size=0.3, random_state=211)
	X_test, X_val, y_test, y_val = train_test_split(X_testval, y_testval, test_size=0.5, random_state=85)
	print('Train Len', len(X_train))
	print('Test Len', len(X_test))
	print('Val Len', len(X_val))
	trainfile = open('forbetter.real.meter.train.3fold.txt', 'w')
	for f_line in X_train:
		#print(f_line)
		for i in f_line:
			#print(i)
			line = '\t'.join([str(m) for m in i])
			#print(line)
			trainfile.write(line)
			trainfile.write('\n')
		trainfile.write('\n')
	trainfile.close()
	testfile = open('forbetter.real.meter.test.3fold.txt', 'w')
	for f_line in X_test:
		for i in f_line:
			line = '\t'.join([str(m) for m in i])
			testfile.write(line)
			testfile.write('\n')
		testfile.write('\n')
	testfile.close()
	valfile = open('forbetter.real.meter.dev.3fold.txt', 'w')
	for f_line in X_val:
		for i in f_line:
			line = '\t'.join([str(m) for m in i])
			valfile.write(line)
			valfile.write('\n')
		valfile.write('\n')
	valfile.close()
