아 main.py
# -------

import explanation

if __name__ == '__main__':

	# captions, relations = explanation.readFile()

	# print('caption: ')
	# print(captions)
	# print('relation:')
	# print(relations)


	# s, o = explanation.detectObject(captions, relations)
	# p = explanation.searchPredicate(relations)

	# explanation.constructPhase(s, o, p, relations)
	# final = explanation.restructCaption(captions, s, o, relations)

	# # 	print("%dth caption" %(i))
	# 	print(final)

	# captions, relations_set = explanation.readFile()

	# # print('caption: ')
	# # print(captions[0])
	# # print('relation:')
	# # print(relations_set[0])


	# for i in range(len(captions)):
	# 	# print(i)
	# 	s, o = explanation.detectObject(captions[i], relations_set[i])
	# 	p = explanation.searchPredicate(relations_set[i])

	# 	explanation.constructPhase(s, o, p, relations_set[i])
	# 	final = explanation.restructCaption(captions[i], s, o, relations_set[i])

	# 	print("%dth caption" %(i))
	# 	print(final)


	caption, relations = explanation.explanation(doLogging = False)
	print(caption)
	print(relations)