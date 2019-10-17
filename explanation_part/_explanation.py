# explanation.py
# --------------

import os, sys
import string
import nltk

MODULE = 'pattern/pattern'
if MODULE not in sys.path: sys.path.insert(0, os.path.join(os.path.dirname(MODULE)))
from pattern.en import conjugate, tag

CAPTION_PATH = '/archive/MyHome/Programs/git/my_research/XAI_TP/rxicg/results/generated_caption'
RLATION_PATH = '/archive/MyHome/Programs/git/my_research/XAI_TP/rxicg/results/relationships'

Logging = False

# this function is for singular caption and relations
# def readFile():
# 	with open('caption', "r") as ins:
# 		captions = []
# 		for caption in ins:
# 			captions.append(caption)

# 	with open('relation', "r") as ins:
# 		relations = []

# 		for relation in ins:
# 			row = relation.rstrip().split(',')
# 			relations.append(row)

# 	return captions, relations

def readFile(test_file_name):

	caption_file_name = os.path.join(CAPTION_PATH, "{}_result_cp.txt".format(test_file_name))
	relation_file_name = os.path.join(RLATION_PATH, "{}_result_vr.txt".format(test_file_name))

	with open(caption_file_name, "r") as ins:
		captions = []
		while True:
			line = ins.readline()
			if not line: break
			caption = line.split('/')[2]
			caption = caption.replace('\n', '')
			captions.append(caption)

	with open(relation_file_name, "r") as ins:
		relations_set = []
		for relations in ins:
			row = relations.rstrip().split('\\')
			relations_set.append(row)

		# print(relations_set)

		for i in range(len(relations_set)):
			for j in range(len(relations_set[i])):	
				row = relations_set[i][j].rstrip().split('-')
				relations_set[i][j] = row

	return captions, relations_set

"""
Determine whether a relation can be used in the caption.

Relation is as follow:
rel_n = (subject, predicate, object)

	Rule (deprecated)
	if obj1 of rel_n is equal to obj1 of rel_m 
		and 
		obj2 of rel_n and rel_m is not used in a caption
		and the relations are verb -> combine this relation and use the active voice

	if obj1 of rel_n is equal to obj2 of rel_m and 
		both obj1 of rel_n and obj2 of rel_m is not used in a caption
		and the relations are verb -> combine this relation and use the passive voice
"""
def detectObject(pred_caption, pred_relations):
	subject_set = {}
	object_set = {}

	_removeDuplicateRelation(pred_caption, pred_relations)

	for i in range(len(pred_relations)):
		# print(pred_relations[i][0] in subject_set)
		if (pred_relations[i][0] in subject_set) is False:
			subject_set[pred_relations[i][0]] = []
			subject_set[pred_relations[i][0]].append(i)
			if Logging:
				print("[detectObject] create '%s' in subject_set and insert '%i'th relation" % (pred_relations[i][0], i))

		else:
			subject_set[pred_relations[i][0]].append(i)
			if Logging:
				print("[detectObject] insert '%i'th relation in object_Set" % (i))

		if (pred_relations[i][2] in object_set) is False:
			object_set[pred_relations[i][2]] = []
			object_set[pred_relations[i][2]].append(i)
			if Logging:
				print("[detectObject] create '%s' in object_set and insert '%i'th relation" % (pred_relations[i][0], i))

		else:
			object_set[pred_relations[i][2]].append(i)
			if Logging:
				print("[detectObject] insert '%i'th relation in object_set" % (i))


	if Logging:
		print("[detectObject] the subject set is " + str(subject_set))
	# print(object_set)

	return subject_set, object_set

"""
Search and remove redundant relations in the caption
"""
def _removeDuplicateRelation(pred_caption, pred_relations):
	relstring = []
	# caption_person = ["man", "woman", "people"]

	# Remove articles in the predicted caption
	copied_caption = pred_caption.lower()
	copied_caption = copied_caption.replace("a ", "")
	copied_caption = copied_caption.replace("the ", "")

	# Replace a term coressponding to the person in the relations
	copied_caption = copied_caption.replace("woman", "person")
	copied_caption = copied_caption.replace("man", "person")
	copied_caption = copied_caption.replace("people", "person")

	if Logging:
		print("[_removeDuplicateRelation] copied caption: '%s'" % (copied_caption))

	for rel in pred_relations:
		pred_rel = rel[0] + ' ' + rel[1] + ' ' + rel[2]
		if Logging:
			print("[_removeDuplicateRelation] generated relation: '%s'" % (pred_rel))
		relstring.append(pred_rel)

	for i in range(len(relstring)):
		if relstring[i] in pred_caption:
			if Logging:
				print("[_removeDuplicateRelation] Removed '%d'th relation" % (i))
			pred_relations.pop(i)

	return 


"""
input: relations
output: pos_list (list of POS of relations)
		3-dimension
		first: order of 

Search the predicates using part of speech (POS) tagging.

The predicates in the relations can be a set of the plural words.
In this case, we recognizes the part of speech by using only the first and last word of the predicates.
"""
def searchPredicate(pred_relations):
	pos_list = []
	for i in range(len(pred_relations)):
		# pred = []
		pred = pred_relations[i][1]
		# pred.append(pred_relations[i][1])
		pred = pred.split()
		# print("[_searchPredicate] gerated relation: '%s'" % ((pred[-1]))
		last_pred = []
		if len(pred) == 1:
			last_pred.append(pred[-1])
		else:
			last_pred.append(pred[0])
			last_pred.append(pred[-1])
		pos_list.append(nltk.tag.pos_tag(last_pred))
		# pos_list.append(nltk.tag.pos_tag(pred))

	if Logging:
		# print("[_searchPredicate] gerated relation: '%s'" % (pred_relations))
		print("[searchPredicate] the tagged list is '%s'" % (pos_list))

	return pos_list

"""
input: subject_set, object_set, and pos_list, pred_relations
output: pred_rel (a set of phase)

Construct phase by using subject_set, object_set, and pos_list.
Check the pos is VB (verb), VBP or VBZ (present tense verb), or VBD.
If so, it needs to conjugate the verb.
To conjugate the verb, we use the open source: http://github.com/clips/pattern


POS tag list:
	CC	coordinating conjunction
	CD	cardinal digit
	DT	determiner
	EX	existential there (like: "there is" ... think of it like "there exists")
	FW	foreign word
	IN	preposition/subordinating conjunction
	JJ	adjective	'big'
	JJR	adjective, comparative	'bigger'
	JJS	adjective, superlative	'biggest'
	LS	list marker	1)
	MD	modal	could, will
	NN	noun, singular 'desk'
	NNS	noun plural	'desks'
	NNP	proper noun, singular	'Harrison'
	NNPS	proper noun, plural	'Americans'
	PDT	predeterminer	'all the kids'
	POS	possessive ending	parent's
	PRP	personal pronoun	I, he, she
	PRP$	possessive pronoun	my, his, hers
	RB	adverb	very, silently,
	RBR	adverb, comparative	better
	RBS	adverb, superlative	best
	RP	particle	give up
	TO	to	go 'to' the store.
	UH	interjection	errrrrrrrm
	VB	verb, base form	take
	VBD	verb, past tense	took
	VBG	verb, gerund/present participle	taking
	VBN	verb, past participle	taken
	VBP	verb, sing. present, non-3d	take
	VBZ	verb, 3rd person sing. present	takes
	WDT	wh-determiner	which
	WP	wh-pronoun	who, what
	WP$	possessive wh-pronoun	whose
	WRB	wh-abverb	where, when
"""
def constructPhase(subject_set, object_set, pos_list, pred_relations):
	# print("conjugate => go")
	# print(conjugate("goes", "ppart"))
	# part or ppart
	pred_rel = []
	for i in range(len(pos_list)):
		if pos_list[i][0][1] in ['VB', 'VBZ', 'VBP', 'VBZ']:
			# if Logging:
	 	# 		print("[constructPhase] '%s'" % (pos[i][0][0]))
		 	pred_relations[i][1] = conjugate(pos_list[i][0][0], "part")
		 	# print(pred_relations[i][1])

		pred_rel.append(pred_relations[i][0] + ' ' + pred_relations[i][1] + ' ' + pred_relations[i][2])

	if Logging:
		print("[constructPhase] total phrase is '%s'" % (pred_rel))

	return pred_rel


"""
input: pred_caption, subject_set, object_set, pred_relations
output: restructed caption, used relations
"""

def restructCaption(pred_caption, subject_set, object_set, pred_relations):
	total_rel = [0 for i in range(len(subject_set.keys()))]

	skey = list(subject_set.keys())

	for_tagging_caption = pred_caption.lower()
	splitted_caption = for_tagging_caption.split()
	next_tag_of_subject_set = {}
	verb_noun_tag = ["VB", "VBP", "VBZ", "VBG", "VBD", "VBN", "NN"]
	person_tag = ["woman", "man", "people"]
	for subject in skey:
		for i in range(len(splitted_caption)):
			if subject == splitted_caption[i] or (subject == "person" and splitted_caption[i] in person_tag):
				next_word = i + 1

				if next_word < len(splitted_caption):
					if Logging:
						print("[restructCaption] next_word is '%s'" % (splitted_caption[next_word]))

					for word, pos in tag(splitted_caption[next_word]):
						if Logging:
							print("[restructCaption] the pos of next word is '%s'" % (pos))

						if pos in verb_noun_tag:
							next_tag_of_subject_set[subject] = True
							if Logging:
								print("[restructCaption] next_tag(VERB OR NOUN?) is appended: '%s'" % (subject + ' : ' + splitted_caption[next_word]))
						else:
							next_tag_of_subject_set[subject] = False

					break

				else:
					next_tag_of_subject_set[subject] = False
			else:
				next_tag_of_subject_set[subject] = False


	for i in range(len(skey)):
		if Logging:
			print("[restructCaption] the subject key is '%s'" % (skey[i]))

		for j in range(len(subject_set[skey[i]])):
			if Logging:
				print("[restructCaption] next_tag is '%s'" % (next_tag_of_subject_set[skey[i]]))
				
			if j == 0 :
				# total_rel[i] = pred_relations[subject_set[skey[i]][j]][0] + ' ' + pred_relations[subject_set[skey[i]][j]][1] + ' ' + pred_relations[subject_set[skey[i]][j]][2]
				total_rel[i] = pred_relations[subject_set[skey[i]][j]][1] + ' ' + pred_relations[subject_set[skey[i]][j]][2]

				if len(subject_set[skey[i]]) == 1 and next_tag_of_subject_set[skey[i]]:
					total_rel[i] = total_rel[i] + ' and'
				continue

			total_rel[i] = total_rel[i] + ' and ' + pred_relations[subject_set[skey[i]][j]][1] + ' ' + pred_relations[subject_set[skey[i]][j]][2]
			if next_tag_of_subject_set[skey[i]]:
					total_rel[i] = total_rel[i] + ' and'

	if Logging:
		print("[restructCaption] total phrase is '%s'" % (total_rel))

	# Insert part
	final_caption = pred_caption
	pop_relations_index = []
	for i in range(len(skey)):
		# final_caption = final_caption.replace(skey[i], total_rel[i])
		if skey[i] == "person":
			if "man" in final_caption:
				final_caption = final_caption.replace("man", "man" + ' ' + total_rel[i])
			elif "woman" in final_caption:
				final_caption = final_caption.replace("woman", "woman" + ' ' + total_rel[i])
			elif "people" in final_caption:
				final_caption = final_caption.replace("people", "people" + ' ' + total_rel[i])
			else:
				index_list = subject_set[skey[i]]
				# print(index_list)
				for j in range(len(index_list)):
					pop_relations_index.append(index_list[j])
		else:
			if skey[i] in final_caption:
				final_caption = final_caption.replace(skey[i], skey[i] + ' ' + total_rel[i])
			else:
				index_list = subject_set[skey[i]]
				# print(index_list)
				for j in range(len(index_list)):
					pop_relations_index.append(index_list[j])
				# print(pop_relations_index)

	used_relations = []
	for i in range(len(pred_relations)):
		if i in pop_relations_index:
			continue
		else:
			used_relations.append(pred_relations[i])

	if Logging:
		print("[restructCaption] final caption is '%s'" % (final_caption))
		print("[restructCaption] final used relations are '%s'" % (used_relations))

	# print(used_relations)

	return final_caption, used_relations


"""
input: Logging (default: False)
output: final caption (restructed caption, type: List)
		relation set (type: 3-dimension list)
"""
def explanation(doLogging = False, test_file_name = None, captions = None, relations_set = None):
	global Logging
	Logging = doLogging
	final_caption = []
	fianl_used_relations = []


	if test_file_name is None:
		if captions is not None and relations_set is not None:
			for i, rel in enumerate(relations_set):
				relations_set[i] = rel.replace('-', ',')
		else:
			print("[Error] Not found file.")
	else:
		captions, relations_set = readFile(test_file_name)

	print(captions)
	print(relations_set)

	print('\n --------------------------')

	for i in range(len(captions)):
		# print(i)
		s, o = detectObject(captions[i], relations_set[i])
		p = searchPredicate(relations_set[i])

		constructPhase(s, o, p, relations_set[i])
		final, used_relations = restructCaption(captions[i], s, o, relations_set[i])

		if Logging:
			print("%dth caption" %(i))
			print(final)
			print(used_relations)

		final_caption.append(final)
		fianl_used_relations.append(used_relations)

	return final_caption, fianl_used_relations


