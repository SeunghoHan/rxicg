# explanation.py
# --------------
from __future__ import division

import os, sys
import string
import nltk
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.patches as patches
import random

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

class Explainer():
	def __init__(self):
		return


	def readFile(self, test_file_name):

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
	def detectObject(self, pred_caption, pred_relations):
		subject_set = {}
		object_set = {}

		self._removeDuplicateRelation(pred_caption, pred_relations)

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
	def _removeDuplicateRelation(self, pred_caption, pred_relations):
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
	def searchPredicate(self, pred_relations):
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
	def constructPhase(self, subject_set, object_set, pos_list, pred_relations):
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

	def restructCaption(self, pred_caption, subject_set, object_set, pred_relations):
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
		splitted_final_caption = final_caption.split()
		pop_relations_index = []
		for i in range(len(skey)):
			# final_caption = final_caption.replace(skey[i], total_rel[i])
			if skey[i] == "person":
				if "man" in splitted_final_caption:
					final_caption = final_caption.replace("man", "man" + ' ' + total_rel[i])
				elif "woman" in splitted_final_caption:
					final_caption = final_caption.replace("woman", "woman" + ' ' + total_rel[i])
				elif "people" in splitted_final_caption:
					final_caption = final_caption.replace("people", "people" + ' ' + total_rel[i])
				else:
					index_list = subject_set[skey[i]]
					# print(index_list)
					for j in range(len(index_list)):
						pop_relations_index.append(index_list[j])
			else:
				if skey[i] in splitted_final_caption:
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
	def explanation(self, doLogging = False, test_file_name = None, captions = None, relations_set = None):
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
				return [], []
		else:
			captions, relations_set = self.readFile(test_file_name)

		for i in range(len(captions)):
			# print(i)
			s, o = self.detectObject(captions[i], relations_set[i])
			p = self.searchPredicate(relations_set[i])

			self.constructPhase(s, o, p, relations_set[i])
			final, used_relations = self.restructCaption(captions[i], s, o, relations_set[i])

			if Logging:
				print("%dth caption" %(i))
				print(final)
				print(used_relations)

			final_caption.append(final)
			fianl_used_relations.append(used_relations)

		return final_caption, fianl_used_relations

	def _random_colors(self, N, bright=True):
		hsv = [(float(i / N), 1, 1.0) for i in range(N)]
		colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
		random.shuffle(colors)

		return colors


	def generate_final_output(self, original_cap, caption, relationships, cls_boxes, cls_names, img_path):
		b_boxes = []
		phrase = []

		for i in range(len(relationships)):
			rel_for_cap = relationships[i]

			if len(rel_for_cap) == 0:
				b_boxes.append(None)
				phrase.append(None)
			else:
				br_box = []
				pr_phrase = []
				for rel in rel_for_cap:
					ob1 = rel[0]
					pred = rel[1]
					ob2 = rel[2]


					ob1_idx = -1
					ob2_idx = -1
					if ob1 in cls_names: ob1_idx = cls_names.index(ob1)
					if ob2 in cls_names: ob2_idx = cls_names.index(ob2)

					if ob1_idx != -1 and ob2_idx != -1:
						ob1_x1, ob1_y1, ob1_x2, ob1_y2 = cls_boxes[ob1_idx]
						ob2_x1, ob2_y1, ob2_x2, ob2_y2 = cls_boxes[ob2_idx]

						rel_x1 = min([ob1_x1, ob2_x1])
						rel_y1 = min([ob1_y1, ob2_y1])
						rel_x2 = max([ob1_x2, ob2_x2])
						rel_y2 = max([ob1_y2, ob2_y2])

						br_box.append([rel_x1, rel_y1, rel_x2, rel_y2])
						pr_phrase.append("{} {} {}".format(ob1, pred, ob2))

				b_boxes.append(br_box)
				phrase.append(pr_phrase)

		ax = None



		
		if not ax:
			fig, ax = plt.subplots(5, figsize=(40, 40))

		img = imread(img_path)

		final_img  = img.astype(np.uint32).copy()
		interval = 30

		for i, cap in enumerate(caption):
			height, width = img.shape[:2]
			ax[i].set_ylim(height + 10, -10)
			ax[i].set_xlim(-10, width + 10)
			ax[i].axis('off')

			if b_boxes[i] is None:
				x1, y1, x2, y2 = [0, 0, 0, 0]
				ax[i].set_title("Generated caption (Not rich caption): {}".format(caption[i]))
			else:
				bb_boxes = b_boxes[i]
				bb_phrase = phrase[i]
				region_num = len(bb_boxes)
				colors = self._random_colors(region_num)   

				for j, box in enumerate(bb_boxes): 
					color = colors[j]
					x1, y1, x2, y2 = box
					p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, alpha=1, linestyle="solid", edgecolor=color, facecolor='none')

					ax[i].add_patch(p)  
					mid_x1 = x1 + (x2 - x1)/2
					mid_y1 = y1 + (y2 - y1)/2
					p_arr = patches.FancyArrow(mid_x1, mid_y1, (width+5)-mid_x1, interval-mid_y1, width=1.5, linewidth=0.5, alpha=1, linestyle="-", color=color)

					ax[i].add_patch(p_arr)

					label = "{}".format(bb_phrase[j])
					ax[i].text(width+20, interval, label, color=color, size=16, backgroundcolor="none")


					interval += 20
					title = "Origianl caption: {}".format(original_cap[i][0])
					title += '\n'
					title += "Generated rich caption: {}".format(caption[i])
					ax[i].set_title(title, size = 18)

			ax[i].imshow(final_img.astype(np.uint8), interpolation='nearest') 