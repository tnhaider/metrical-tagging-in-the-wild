import sys, re
import joblib
import pyphen
from inout.dta.corpus import Corpus
from inout.dta.poem import Poem
from inout.utils.helper import *
from hyphenation.syllabifier import Syllabifier
from mtl.bilstmcrf.Hyphenator_DE import Hyphenator
from somajo import SoMaJo
from nltk import bigrams as bi
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from collections.abc import Iterable   # drop `.abc` with Python 2.7 or lower

tokenizer = SoMaJo("de_CMC", split_camel_case=True)
#syllabifier = Syllabifier()
syllabifier = Hyphenator()
dic = pyphen.Pyphen(lang='de_DE')


#for sentence in tokenized:
#	for token in sentence:
#		try:
#			tokens.append(str(token.text))
#			tokens.append(str(token.token_class))
#syllabifier.predict(token)

def is_iterable(obj):
    return isinstance(obj, Iterable)

def word2features(sentence, index):
        word = sentence[index][0]
        postag = sentence[index][1]
        features = {
        # uebernommen vom DecisionTreeClassifier
                'word': word,
                'position_in_sentence': index,
                'rel_position_in_sentence': index / len(sentence),
                'is_first': index == 0,
                'is_last': index == len(sentence) - 1,
                'is_capitalized': word[0].upper() == word[0],
                'next_capitalized': '' if index == len(sentence) -1 else sentence[index+1][0].upper() == sentence[index+1][0],
                'last_capitalized': '' if index == 0 else sentence[index-1][0].upper() == sentence[index-1][0],
                'is_all_caps': word.upper() == word,
                'is_all_lower': word.lower() == word,
                'prefix-1-low': word[0].lower(),
                'prefix-1': word[0],
                'prefix-2': word[:2],
                'prefix-3': word[:3],
                'prefix-4': word[:4],
                'suffix-1': word[-1],
                'suffix-2': word[-2:],
                'suffix-3': word[-3:],
                'suffix-4': word[-4:],
                'prev_word': '' if index == 0 else sentence[index-1][0],
                'prev_prev_word': '' if index == 0 or index == 1 else sentence[index-2][0],
		'next_word': '' if index == len(sentence) - 1 else sentence[index + 1][0],
                'next_next_word': '' if index == len(sentence) - 1 or index == len(sentence) -2  else sentence[index + 2][0],
                #'prev_tag': '' if index == 0 else sentence[index-1][1],
                #'next_tag': '' if index == len(sentence)-1 else sentence[index+1][1],
                'has_hyphen': '-' in word,
                'is_numeric': word.isdigit(),
                'capitals_inside': word[1:].lower() != word[1:]
        }
        return features


def sent2features(sentence):
        return [word2features(sentence, i) for i in range(len(sentence))]
                
def get_pos_label(label):
        if label.startswith('ADV'):
                return 'ADV'
        elif label.startswith('ADJ'):
                return 'ADJ'
        else:
                return label[:2]

def get_pos_sequence(pos_model, tokenized_line):
        tokenized_line = [(i, '') for i in tokenized_line]
        sent_features = sent2features(tokenized_line)
        pos = pos_model.predict([sent_features])[0]
        return pos

def get_caesura_anno(footmeter):
	feetout = []
	bimeter = bi([m for m in re.sub('\|', ':', str(footmeter))])
	for a, b in bimeter:
		if a == '.':
			continue
		elif b == ':':
			feetout.append(':')
		else:
			feetout.append('.')
	return feetout
		

def get_features(line, pos_model):
	try:
		linetext = line.get_text()
		#print(linetext)
		footmeter = line.get_meter()
		feet = get_caesura_anno(footmeter)
		#print(footmeter)
		meter = re.sub('\|', '', footmeter)
		versemeasure = get_versification(meter)
		measure = versemeasure.split('.')[0]
		#print(meter)
		caesurarhythm = line.get_rhythm()
		caesuras = get_caesura_anno(caesurarhythm)
		rhythm = re.sub('\|', '', caesurarhythm)
		rhythm = re.sub('\(', '', rhythm)
		rhythm = re.sub('\)', '', rhythm)
		rhythm = re.sub('\:', '', rhythm)
		#print(caesurarhythm)
		enjambement = line.get_enjambement_kontext()
		syllable_position = []
		syll_seq = []
		pos_syll = []
		simple_pos_syll = []
		tokenized = tokenizer.tokenize_text([linetext])
		tokens = []
		token_classes = []
	except TypeError:
		return 0
	for sentence in tokenized:
		for token in sentence:
			tokentext = token.text
			tokens.append(tokentext)
			tokenclass = token.token_class
			token_classes.append(tokenclass)
	pos_seq = get_pos_sequence(pos_model, tokens)
	for token, token_c, pos in zip(tokens, token_classes, pos_seq):
		try:
			#print(tokenclass)
			#hyphenated = dic.inserted(word)
			#hyphenated = re.sub("-'", "'", hyphenated)
			syllabified = syllabifier.predict(token)
			syllables = syllabified.split('·') 
			#print(syllabified)
			#print(tokenclass)
			if token_c == 'regular':
				syll_position = 0
				for s in syllables:
					syll_position += 1
					if len(syllables) == 1:
						syllable_position.append(0)
					else:
						syllable_position.append(syll_position)
					syll_seq.append(s)
					pos_syll.append(pos)
					simple_pos_syll.append(simplify_pos_label(pos))
		except IndexError:
				continue


	if len(meter) != len(syll_seq):
		print('MET INFO', meter, versemeasure, rhythm, syll_seq, syllable_position, pos_syll, 'enj'+str(enjambement), '// LEN', str(len(meter)), str(len(syll_seq)), str(len(meter) == len(syll_seq)))

	if len(meter) == len(syll_seq):
		return zip(range(1, len(meter)+1), syll_seq, meter, rhythm, feet, caesuras, simple_pos_syll, pos_syll, syllable_position, [measure]*len(meter), [versemeasure]*len(meter), [meter]*len(meter))
			#return '\t'.join([str(m) for m in i])
			#outfile.write('\n')
	else: return None
	#outfile.write('\n')
	#return 1	
	#if not len(meter) == len(syll_seq):
	#	print('MET MIS', meter, versemeasure, syll_seq, len(meter), len(syll_seq), len(meter) == len(syll_seq))
	#	print('TOKEN: ', tokens)
	#	return 1
	#else: return 0
	#if not len(rhythm) == len(syll_seq):
	#	print('RHY MIS', rhythm, syll_seq, len(rhythm), len(syll_seq), len(rhythm) == len(syll_seq))
	#	return 1
	#else: return 0
	#if '000' in rhythm:
	#	print('000 RHY', rhythm, syll_seq, len(rhythm), len(syll_seq), len(rhythm) == len(syll_seq))
	#	return 1
	#else: return 0
	#if meter.endswith('-+--+-') and meter.count('+') == 6:
	#	print('-++- MET', meter, syll_seq, len(meter), len(syll_seq), len(meter) == len(syll_seq))
	#	return 1
	#else: return 0
	#if '-++-' in meter:
	#	print('-++- MET', meter, versemeasure, syll_seq, len(meter), len(syll_seq), len(meter) == len(syll_seq))
	#	return 1
	#else: return 0
	#print('MET', meter, versemeasure, syll_seq, len(meter), len(syll_seq), len(meter) == len(syll_seq))
	#return 1

if __name__ == '__main__':
	counter = 0
	corpuspath = sys.argv[1]
	posmodelpath = sys.argv[2]
	#outfile = open('german.meter.multi.measure.tsv', 'w')
	pos_model = joblib.load(posmodelpath)
	c = Corpus(corpuspath)
	poems = c.read_poems()
	feature_lines = []
	faulty = 0
	for poem in poems:
		counter += 1
		#if counter > 5:
		#	continue
		for stanza in poem.get_stanzas():
			for line in stanza.get_line_objects():
				feature_line = get_features(line, pos_model)
				if feature_line is not None and is_iterable(feature_line):
					feature_lines.append(feature_line)
				if not feature_line:
					faulty += 1
	shuffle(feature_lines)
	y = ['1']*len(feature_lines)
	X_train, X_testval, y_train, y_testval = train_test_split(feature_lines, y, test_size=0.2, random_state=198)
	X_test, X_val, y_test, y_val = train_test_split(X_testval, y_testval, test_size=0.5, random_state=131)
	print('Total Lines', len(feature_lines) + faulty)
	print('Total Lines Correct', len(feature_lines))
	print('Faulty', faulty)
	print('Train Len', len(X_train))
	print('Test Len', len(X_test))
	print('Val Len', len(X_val))
	trainfile = open('german.meter.train.5fold.txt', 'w')
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
	testfile = open('german.meter.test.5fold.txt', 'w')
	for f_line in X_test:
		for i in f_line:
			line = '\t'.join([str(m) for m in i])
			testfile.write(line)
			testfile.write('\n')
		testfile.write('\n')
	testfile.close()
	valfile = open('german.meter.dev.5fold.txt', 'w')
	for f_line in X_val:
		for i in f_line:
			line = '\t'.join([str(m) for m in i])
			valfile.write(line)
			valfile.write('\n')
		valfile.write('\n')
	valfile.close()


	
