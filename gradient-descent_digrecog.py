from PIL import Image
from matplotlib import pyplot as plt
from numpy import *

import numpy as np
import csv
import idx2numpy
import pandas as pd

def turnNegative(features):
	featureListNegative = np.zeros((57, ), dtype=float32)
	for i in range(0, 57):
		if features[i] == 1:
			featureListNegative[i] = 0
		else:
			featureListNegative[i] = 1

	return featureListNegative



def imageProcess(image, isTesting):
	img = image.resize((28, 28))
	gray = img.convert('L')

	pixels = array(gray).flatten()
	pixels = pixels.astype('float32')

	arr = pixels.reshape(28, 28)
	arr = 1/(1 + np.exp(-arr + 127))
	arr = arr * 255

	image = Image.fromarray(arr)
	
	if isTesting:
		plt.imshow(image)
		plt.show()

	arr /= 255
	arr = arr.astype(int)

	arr = arr.reshape(784,)

	for i in range(0, 783):
		if arr[i] == 0:
			arr[i] = -1

	arr = arr.reshape(28, 28)

	######################################## LAYER 1: Convolutional Layer ########################################

	print("########## Layer 1 ##########")

	# Calculate the probability of vertical lines

	print ("Vertical")

	probVert = np.empty([27, 27], dtype=int)

	for i in range(0, 27):
		for j in range(0, 27):
			vert1 = 1*arr[i][j] + -1*arr[i][j+1] + 1*arr[i+1][j] + -1*arr[i+1][j+1]
			vert2 = -1*arr[i][j] + 1*arr[i][j+1] + -1*arr[i+1][j] + 1*arr[i+1][j+1]
			probVert[i][j] = max(vert1, vert2) / 4

	if isTesting:
		print(probVert)

	# Calculate the probability of horizontal lines

	print ("Horizontal")

	probHoriz = np.empty([27, 27], dtype=int)

	for i in range(0, 27):
		for j in range(0, 27):
			horiz1 = -1*arr[i][j] + -1*arr[i][j+1] + 1*arr[i+1][j] + 1*arr[i+1][j+1]
			horiz2 = 1*arr[i][j] + 1*arr[i][j+1] + -1*arr[i+1][j] + -1*arr[i+1][j+1]
			probHoriz[i][j] = max(horiz1, horiz2) / 4

	if isTesting:
		print(probHoriz)

	# Calculate the probability of up slants lines

	print ("Up Slants")

	probUpSlant = np.empty([27, 27], dtype=int)

	for i in range(0, 27):
		for j in range(0, 27):
			upSlant1 = 1*arr[i][j] + -1*arr[i][j+1] + -1*arr[i+1][j] + 1*arr[i+1][j+1]
			upSlant2 = 1*arr[i][j] + -1*arr[i][j+1] + -1*arr[i+1][j] + -1*arr[i+1][j+1]
			upSlant3 = -1*arr[i][j] + -1*arr[i][j+1] + -1*arr[i+1][j] + 1*arr[i+1][j+1]
			probUpSlant[i][j] = max(upSlant1, upSlant2, upSlant3) / 4

	if isTesting:
		print(probUpSlant)

	# Calculate the probability of down slants lines

	print ("Down Slants")

	probDownSlant = np.empty([27, 27], dtype=int)

	for i in range(0, 27):
		for j in range(0, 27):
			downSlant1 = -1*arr[i][j] + 1*arr[i][j+1] + 1*arr[i+1][j] + -1*arr[i+1][j+1]
			downSlant2 = -1*arr[i][j] + -1*arr[i][j+1] + 1*arr[i+1][j] + -1*arr[i+1][j+1]
			downSlant3 = -1*arr[i][j] + 1*arr[i][j+1] + -1*arr[i+1][j] + -1*arr[i+1][j+1]
			probDownSlant[i][j] = max(downSlant1, downSlant2, downSlant3) / 4

	if isTesting:
		print(probDownSlant)

	######################################## DOWN SAMPLING ########################################

	probVertZero = np.zeros((28, 28), dtype=int)
	probHorizZero = np.zeros((28, 28), dtype=int)
	probUpSlantZero = np.zeros((28, 28), dtype=int)
	probDownSlantZero = np.zeros((28, 28), dtype=int)

	for i in range(0, 26):
		for j in range(0, 26):
			probVertZero[i][j] = probVert[i][j]
			probHorizZero[i][j] = probHoriz[i][j]
			probUpSlantZero[i][j] = probUpSlant[i][j]
			probDownSlantZero[i][j] = probDownSlant[i][j]

	matDownSample = np.zeros((7, 7), dtype=int)

	for i in range(0, 7):
		for j in range(0, 7):
			vertMatSum = probVertZero[i*4][j*4] + probVertZero[i*4][j*4 + 1] + probVertZero[i*4][j*4 + 2] + probVertZero[i*4][j*4 + 3] \
					+ probVertZero[i*4 + 1][j*4] + probVertZero[i*4 + 1][j*4 + 1] + probVertZero[i*4 + 1][j*4 + 2] + probVertZero[i*4 + 1][j*4 + 3] \
					+ probVertZero[i*4 + 2][j*4] + probVertZero[i*4 + 2][j*4 + 1] + probVertZero[i*4 + 2][j*4 + 2] + probVertZero[i*4 + 2][j*4 + 3] \
					+ probVertZero[i*4 + 3][j*4] + probVertZero[i*4 + 3][j*4 + 1] + probVertZero[i*4 + 3][j*4 + 2] + probVertZero[i*4 + 3][j*4 + 3]

			horizMatSum = probHorizZero[i*4][j*4] + probHorizZero[i*4][j*4 + 1] + probHorizZero[i*4][j*4 + 2] + probHorizZero[i*4][j*4 + 3] \
					+ probHorizZero[i*4 + 1][j*4] + probHorizZero[i*4 + 1][j*4 + 1] + probHorizZero[i*4 + 1][j*4 + 2] + probHorizZero[i*4 + 1][j*4 + 3] \
					+ probHorizZero[i*4 + 2][j*4] + probHorizZero[i*4 + 2][j*4 + 1] + probHorizZero[i*4 + 2][j*4 + 2] + probHorizZero[i*4 + 2][j*4 + 3] \
					+ probHorizZero[i*4 + 3][j*4] + probHorizZero[i*4 + 3][j*4 + 1] + probHorizZero[i*4 + 3][j*4 + 2] + probHorizZero[i*4 + 3][j*4 + 3]

			upSlantMatSum = probUpSlantZero[i*4][j*4] + probUpSlantZero[i*4][j*4 + 1] + probUpSlantZero[i*4][j*4 + 2] + probUpSlantZero[i*4][j*4 + 3] \
					+ probUpSlantZero[i*4 + 1][j*4] + probUpSlantZero[i*4 + 1][j*4 + 1] + probUpSlantZero[i*4 + 1][j*4 + 2] + probUpSlantZero[i*4 + 1][j*4 + 3] \
					+ probUpSlantZero[i*4 + 2][j*4] + probUpSlantZero[i*4 + 2][j*4 + 1] + probUpSlantZero[i*4 + 2][j*4 + 2] + probUpSlantZero[i*4 + 2][j*4 + 3] \
					+ probUpSlantZero[i*4 + 3][j*4] + probUpSlantZero[i*4 + 3][j*4 + 1] + probUpSlantZero[i*4 + 3][j*4 + 2] + probUpSlantZero[i*4 + 3][j*4 + 3]

			downSlantMatSum = probDownSlantZero[i*4][j*4] + probDownSlantZero[i*4][j*4 + 1] + probDownSlantZero[i*4][j*4 + 2] + probDownSlantZero[i*4][j*4 + 3] \
					+ probDownSlantZero[i*4 + 1][j*4] + probDownSlantZero[i*4 + 1][j*4 + 1] + probDownSlantZero[i*4 + 1][j*4 + 2] + probDownSlantZero[i*4 + 1][j*4 + 3] \
					+ probDownSlantZero[i*4 + 2][j*4] + probDownSlantZero[i*4 + 2][j*4 + 1] + probDownSlantZero[i*4 + 2][j*4 + 2] + probDownSlantZero[i*4 + 2][j*4 + 3] \
					+ probDownSlantZero[i*4 + 3][j*4] + probDownSlantZero[i*4 + 3][j*4 + 1] + probDownSlantZero[i*4 + 3][j*4 + 2] + probDownSlantZero[i*4 + 3][j*4 + 3]

			maxMatSum = float(max(vertMatSum, horizMatSum, upSlantMatSum, downSlantMatSum))

			if maxMatSum == vertMatSum:
				maxMatSum = maxMatSum / 16.0
				if maxMatSum >= 0.05:
					matDownSample[i][j] = 1
				else:
					matDownSample[i][j] = 0
			elif maxMatSum == horizMatSum:
				maxMatSum = maxMatSum / 16
				if maxMatSum >= 0.05:
					matDownSample[i][j] = 2
				else:
					matDownSample[i][j] = 0
			elif maxMatSum == upSlantMatSum:
				maxMatSum = maxMatSum / 16
				if maxMatSum >= 0.05:
					matDownSample[i][j] = 3
				else:
					matDownSample[i][j] = 0
			elif maxMatSum == downSlantMatSum:
				maxMatSum = maxMatSum / 16
				if maxMatSum >= 0.05:
					matDownSample[i][j] = 4
				else:
					matDownSample[i][j] = 0

	if isTesting:
		print(matDownSample)

	######################################## LAYER 2 ########################################

	features = np.zeros((57,), dtype=int)

	# Left Top Arc - Quad 1

	for i in range(0, 2):
		for j in range(0, 2):
			if matDownSample[i+1][j] == 3 and matDownSample[i][j+1] == 2:
				features[0] = 1
			elif (matDownSample[i][j] == 3 and matDownSample[i][j+1] == 2) or (matDownSample[i+1][j] == 3 and matDownSample[i+1][j+1] == 2):
				features[0] = 1
			elif matDownSample[i+1][j] == 1 and matDownSample[i][j+1] == 3:
				features[0] = 1
			elif (matDownSample[i][j] == 1 and matDownSample[i][j+1] == 3) or (matDownSample[i+1][j] == 1 and matDownSample[i+1][j+1] == 3):
				features[0] = 1
			elif matDownSample[i][j] == 1 and matDownSample[i][j+1] == 2:
				features[0] = 1

	if isTesting:
		print("Left Top Arc - Quad 1: " + str(features[0]))

	# Left Top Arc - Quad 7

	for i in range(4, 6):
		for j in range(0, 2):
			if matDownSample[i+1][j] == 3 and matDownSample[i][j+1] == 2:
				features[1] = 1
			elif (matDownSample[i][j] == 3 and matDownSample[i][j+1] == 2) or (matDownSample[i+1][j] == 3 and matDownSample[i+1][j+1] == 2):
				features[1] = 1
			elif matDownSample[i+1][j] == 1 and matDownSample[i][j+1] == 3:
				features[1] = 1
			elif (matDownSample[i][j] == 1 and matDownSample[i][j+1] == 3) or (matDownSample[i+1][j] == 1 and matDownSample[i+1][j+1] == 3):
				features[1] = 1

	if isTesting:
		print("Left Top Arc - Quad 7: " + str(features[1]))


	# Right Top Arc - Quad 3

	for i in range(0, 2):
		for j in range(4, 6):
			if matDownSample[i][j] == 2 and matDownSample[i+1][j+1] == 4:
				features[2] = 1
			elif (matDownSample[i][j] == 2 and matDownSample[i][j+1] == 4) or (matDownSample[i+1][j] == 2 and matDownSample[i+1][j+1] == 4):
				features[2] = 1
			elif matDownSample[i][j] == 4 and matDownSample[i+1][j+1] == 1:
				features[2] = 1
			elif (matDownSample[i][j] == 4 and matDownSample[i][j+1] == 1) or (matDownSample[i+1][j] == 4 and matDownSample[i+1][j+1] == 1):
				features[2] = 1

	if isTesting:
		print("Right Top Arc - Quad 3: " + str(features[2]))

	# Right Top Arc - Quad 9

	for i in range(4, 6):
		for j in range(4, 6):
			if matDownSample[i][j] == 2 and matDownSample[i+1][j+1] == 4:
				features[3] = 1
			elif (matDownSample[i][j] == 2 and matDownSample[i][j+1] == 4) or (matDownSample[i+1][j] == 2 and matDownSample[i+1][j+1] == 4):
				features[3] = 1
			elif matDownSample[i][j] == 4 and matDownSample[i+1][j+1] == 1:
				features[3] = 1
			elif (matDownSample[i][j] == 4 and matDownSample[i][j+1] == 1) or (matDownSample[i+1][j] == 4 and matDownSample[i+1][j+1] == 1):
				features[3] = 1

	if isTesting:
		print("Right Top Arc - Quad 9: " + str(features[3]))

	# Left Bottom Arc - Quad 1

	for i in range(0, 2):
		for j in range(0, 2):
			if matDownSample[i+1][j] == 1 and matDownSample[i][j+1] == 3:
				features[4] = 1
			elif (matDownSample[i][j] == 1 and matDownSample[i][j+1] == 3) or (matDownSample[i+1][j] == 1 and matDownSample[i+1][j+1] == 3):
				features[4] = 1
			elif matDownSample[i+1][j] == 4 and matDownSample[i][j+1] == 2:
				features[4] = 1
			elif (matDownSample[i][j] == 4 and matDownSample[i][j+1] == 2) or (matDownSample[i+1][j] == 4 and matDownSample[i+1][j+1] == 2):
				features[4] = 1

	if isTesting:
		print("Left Bottom Arc - Quad 1: " + str(features[4]))

	# Left Bottom Arc - Quad 7

	for i in range(4, 6):
		for j in range(0, 2):
			if matDownSample[i+1][j] == 1 and matDownSample[i][j+1] == 3:
				features[5] = 1
			elif (matDownSample[i][j] == 1 and matDownSample[i][j+1] == 3) or (matDownSample[i+1][j] == 1 and matDownSample[i+1][j+1] == 3):
				features[5] = 1
			elif matDownSample[i+1][j] == 4 and matDownSample[i][j+1] == 2:
				features[5] = 1
			elif (matDownSample[i][j] == 4 and matDownSample[i][j+1] == 2) or (matDownSample[i+1][j] == 4 and matDownSample[i+1][j+1] == 2):
				features[5] = 1

	if isTesting:
		print("Left Bottom Arc - Quad 7: " + str(features[5]))

	# Right Bottom Arc - Quad 3

	for i in range(0, 2):
		for j in range(4, 6):
			if matDownSample[i+1][j] == 2 and matDownSample[i][j+1] == 3:
				features[6] = 1
			elif (matDownSample[i][j] == 2 and matDownSample[i][j+1] == 3) or (matDownSample[i+1][j] == 2 and matDownSample[i+1][j+1] == 3):
				features[6] = 1
			elif matDownSample[i+1][j] == 3 and matDownSample[i][j+1] == 1:
				features[6] = 1
			elif (matDownSample[i][j] == 3 and matDownSample[i][j+1] == 1) or (matDownSample[i+1][j] == 3 and matDownSample[i+1][j+1] == 1):
				features[6] = 1

	if isTesting:
		print("Right Bottom Arc - Quad 3: " + str(features[6]))

	# Right Bottom Arc - Quad 9

	for i in range(4, 6):
		for j in range(4, 6):
			if matDownSample[i+1][j] == 2 and matDownSample[i][j+1] == 3:
				features[7] = 1
			elif (matDownSample[i][j] == 2 and matDownSample[i][j+1] == 3) or (matDownSample[i+1][j] == 2 and matDownSample[i+1][j+1] == 3):
				features[7] = 1
			elif matDownSample[i+1][j] == 3 and matDownSample[i][j+1] == 1:
				features[7] = 1
			elif (matDownSample[i][j] == 3 and matDownSample[i][j+1] == 1) or (matDownSample[i+1][j] == 3 and matDownSample[i+1][j+1] == 1):
				features[7] = 1

	if isTesting:
		print("Right Bottom Arc - Quad 9: " + str(features[7]))

	# Horizontal - Quad 1

	for i in range(0, 2):
		for j in range(0, 2):
			if (matDownSample[i][j] == 2 and matDownSample[i][j+1] == 2) or (matDownSample[i+1][j] == 2 and matDownSample[i+1][j+1] == 2):
				features[8] = 1

	if isTesting:
		print("Horizontal - Quad 1: " + str(features[8]))

	# Horizontal - Quad 2

	for i in range(0, 2):
		for j in range(2, 4):
			if (matDownSample[i][j] == 2 and matDownSample[i][j+1] == 2) or (matDownSample[i+1][j] == 2 and matDownSample[i+1][j+1] == 2):
				features[9] = 1

	if isTesting:
		print("Horizontal - Quad 2: " + str(features[9]))

	# Horizontal - Quad 3

	for i in range(0, 2):
		for j in range(4, 6):
			if (matDownSample[i][j] == 2 and matDownSample[i][j+1] == 2) or (matDownSample[i+1][j] == 2 and matDownSample[i+1][j+1] == 2):
				features[10] = 1

	if isTesting:
		print("Horizontal - Quad 3: " + str(features[10]))

	# Horizontal - Quad 4

	for i in range(2, 4):
		for j in range(0, 2):
			if (matDownSample[i][j] == 2 and matDownSample[i][j+1] == 2) or (matDownSample[i+1][j] == 2 and matDownSample[i+1][j+1] == 2):
				features[11] = 1

	if isTesting:
		print("Horizontal - Quad 4: " + str(features[11]))

	# Horizontal - Quad 5

	for i in range(2, 4):
		for j in range(2, 4):
			if (matDownSample[i][j] == 2 and matDownSample[i][j+1] == 2) or (matDownSample[i+1][j] == 2 and matDownSample[i+1][j+1] == 2):
				features[12] = 1

	if isTesting:
		print("Horizontal - Quad 5: " + str(features[12]))

	# Horizontal - Quad 6

	for i in range(2, 4):
		for j in range(4, 6):
			if (matDownSample[i][j] == 2 and matDownSample[i][j+1] == 2) or (matDownSample[i+1][j] == 2 and matDownSample[i+1][j+1] == 2):
				features[13] = 1

	if isTesting:
		print("Horizontal - Quad 6: " + str(features[13]))

	# Horizontal - Quad 7

	for i in range(4, 6):
		for j in range(0, 2):
			if (matDownSample[i][j] == 2 and matDownSample[i][j+1] == 2) or (matDownSample[i+1][j] == 2 and matDownSample[i+1][j+1] == 2):
				features[14] = 1

	if isTesting:
		print("Horizontal - Quad 7: " + str(features[14]))

	# Horizontal - Quad 8

	for i in range(4, 6):
		for j in range(2, 4):
			if (matDownSample[i][j] == 2 and matDownSample[i][j+1] == 2) or (matDownSample[i+1][j] == 2 and matDownSample[i+1][j+1] == 2):
				features[15] = 1

	if isTesting:
		print("Horizontal - Quad 8: " + str(features[15]))

	# Horizontal - Quad 9

	for i in range(4, 6):
		for j in range(4, 6):
			if (matDownSample[i][j] == 2 and matDownSample[i][j+1] == 2) or (matDownSample[i+1][j] == 2 and matDownSample[i+1][j+1] == 2):
				features[16] = 1

	if isTesting:
		print("Horizontal - Quad 9: " + str(features[16]))

	#Vertical - Quad 1

	for i in range(0, 2):
		for j in range(0, 2):
			if (matDownSample[i][j] == 1 and matDownSample[i+1][j] == 1) or (matDownSample[i][j+1] == 1 and matDownSample[i+1][j+1]):
				features[17] = 1

	if isTesting:
		print("Vertical - Quad 1: " + str(features[17]))

	#Vertical - Quad 2

	for i in range(0, 2):
		for j in range(2, 4):
			if (matDownSample[i][j] == 1 and matDownSample[i+1][j] == 1) or (matDownSample[i][j+1] == 1 and matDownSample[i+1][j+1]):
				features[18] = 1

	if isTesting:
		print("Vertical - Quad 2: " + str(features[18]))

	#Vertical - Quad 3

	for i in range(0, 2):
		for j in range(4, 6):
			if (matDownSample[i][j] == 1 and matDownSample[i+1][j] == 1) or (matDownSample[i][j+1] == 1 and matDownSample[i+1][j+1]):
				features[19] = 1

	if isTesting:
		print("Vertical - Quad 3: " + str(features[19]))

	#Vertical - Quad 4

	for i in range(2, 4):
		for j in range(0, 2):
			if (matDownSample[i][j] == 1 and matDownSample[i+1][j] == 1) or (matDownSample[i][j+1] == 1 and matDownSample[i+1][j+1]):
				features[20] = 1

	if isTesting:
		print("Vertical - Quad 4: " + str(features[20]))

	#Vertical - Quad 5

	for i in range(2, 4):
		for j in range(2, 4):
			if (matDownSample[i][j] == 1 and matDownSample[i+1][j] == 1) or (matDownSample[i][j+1] == 1 and matDownSample[i+1][j+1]):
				features[21] = 1

	if isTesting:
		print("Vertical - Quad 5: " + str(features[21]))

	#Vertical - Quad 6

	for i in range(2, 4):
		for j in range(4, 6):
			if (matDownSample[i][j] == 1 and matDownSample[i+1][j] == 1) or (matDownSample[i][j+1] == 1 and matDownSample[i+1][j+1]):
				features[22] = 1

	if isTesting:
		print("Vertical - Quad 6: " + str(features[22]))

	#Vertical - Quad 7

	for i in range(4, 6):
		for j in range(0, 2):
			if (matDownSample[i][j] == 1 and matDownSample[i+1][j] == 1) or (matDownSample[i][j+1] == 1 and matDownSample[i+1][j+1]):
				features[23] = 1

	if isTesting:
		print("Vertical - Quad 7: " + str(features[23]))

	#Vertical - Quad 8

	for i in range(4, 6):
		for j in range(2, 4):
			if (matDownSample[i][j] == 1 and matDownSample[i+1][j] == 1) or (matDownSample[i][j+1] == 1 and matDownSample[i+1][j+1]):
				features[24] = 1

	if isTesting:
		print("Vertical - Quad 8: " + str(features[24]))

	#Vertical - Quad 9

	for i in range(4, 6):
		for j in range(4, 6):
			if (matDownSample[i][j] == 1 and matDownSample[i+1][j] == 1) or (matDownSample[i][j+1] == 1 and matDownSample[i+1][j+1]):
				features[25] = 1

	if isTesting:
		print("Vertical - Quad 9: " + str(features[25]))

	# Up Slant - Quad 1

	for i in range(0, 2):
		for j in range(0, 2):
			if matDownSample[i+1][j] == 3 and matDownSample[i][j+1] == 3:
				features[26] = 1

	if isTesting:
		print("Up Slant - Quad 1: " + str(features[26]))

	# Up Slant - Quad 2

	for i in range(0, 2):
		for j in range(2, 4):
			if matDownSample[i+1][j] == 3 and matDownSample[i][j+1] == 3:
				features[27] = 1

	if isTesting:
		print("Up Slant - Quad 2: " + str(features[27]))

	# Up Slant - Quad 3

	for i in range(0, 2):
		for j in range(4, 6):
			if matDownSample[i+1][j] == 3 and matDownSample[i][j+1] == 3:
				features[28] = 1

	if isTesting:
		print("Up Slant - Quad 3: " + str(features[28]))

	# Up Slant - Quad 4

	for i in range(2, 4):
		for j in range(0, 2):
			if matDownSample[i+1][j] == 3 and matDownSample[i][j+1] == 3:
				features[29] = 1

	if isTesting:
		print("Up Slant - Quad 4: " + str(features[29]))

	# Up Slant - Quad 5

	for i in range(2, 4):
		for j in range(2, 4):
			if matDownSample[i+1][j] == 3 and matDownSample[i][j+1] == 3:
				features[30] = 1

	if isTesting:
		print("Up Slant - Quad 5: " + str(features[30]))

	# Up Slant - Quad 6

	for i in range(2, 4):
		for j in range(4, 6):
			if matDownSample[i+1][j] == 3 and matDownSample[i][j+1] == 3:
				features[31] = 1

	if isTesting:
		print("Up Slant - Quad 6: " + str(features[31]))

	# Up Slant - Quad 7

	for i in range(4, 6):
		for j in range(0, 2):
			if matDownSample[i+1][j] == 3 and matDownSample[i][j+1] == 3:
				features[32] = 1

	if isTesting:
		print("Up Slant - Quad 7: " + str(features[32]))

	# Up Slant - Quad 8

	for i in range(4, 6):
		for j in range(2, 4):
			if matDownSample[i+1][j] == 3 and matDownSample[i][j+1] == 3:
				features[33] = 1

	if isTesting:
		print("Up Slant - Quad 8: " + str(features[33]))

	# Up Slant - Quad 9

	for i in range(4, 6):
		for j in range(4, 6):
			if matDownSample[i+1][j] == 3 and matDownSample[i][j+1] == 3:
				features[34] = 1

	if isTesting:
		print("Up Slant - Quad 9: " + str(features[34]))

	# Down Slant - Quad 1

	for i in range(0, 2):
		for j in range(0, 2):
			if matDownSample[i][j] == 4 and matDownSample[i+1][j+1] == 4:
				features[35] = 1

	if isTesting:
		print("Down Slant - Quad 1: " + str(features[35]))

	# Down Slant - Quad 2

	for i in range(0, 2):
		for j in range(2, 4):
			if matDownSample[i][j] == 4 and matDownSample[i+1][j+1] == 4:
				features[36] = 1

	if isTesting:
		print("Down Slant - Quad 2: " + str(features[36]))

	# Down Slant - Quad 3

	for i in range(0, 2):
		for j in range(4, 6):
			if matDownSample[i][j] == 4 and matDownSample[i+1][j+1] == 4:
				features[37] = 1

	if isTesting:
		print("Down Slant - Quad 3: " + str(features[37]))

	# Down Slant - Quad 4

	for i in range(2, 4):
		for j in range(0, 2):
			if matDownSample[i][j] == 4 and matDownSample[i+1][j+1] == 4:
				features[38] = 1

	if isTesting:
		print("Down Slant - Quad 4: " + str(features[38]))

	# Down Slant - Quad 5

	for i in range(2, 4):
		for j in range(2, 4):
			if matDownSample[i][j] == 4 and matDownSample[i+1][j+1] == 4:
				features[39] = 1

	if isTesting:
		print("Down Slant - Quad 5: " + str(features[39]))

	# Down Slant - Quad 6

	for i in range(2, 4):
		for j in range(4, 6):
			if matDownSample[i][j] == 4 and matDownSample[i+1][j+1] == 4:
				features[40] = 1

	if isTesting:
		print("Down Slant - Quad 6: " + str(features[40]))

	# Down Slant - Quad 7

	for i in range(4, 6):
		for j in range(0, 2):
			if matDownSample[i][j] == 4 and matDownSample[i+1][j+1] == 4:
				features[41] = 1

	if isTesting:
		print("Down Slant - Quad 7: " + str(features[41]))

	# Down Slant - Quad 8

	for i in range(4, 6):
		for j in range(2, 4):
			if matDownSample[i][j] == 4 and matDownSample[i+1][j+1] == 4:
				features[42] = 1

	if isTesting:
		print("Down Slant - Quad 8: " + str(features[42]))

	# Down Slant - Quad 9

	for i in range(4, 6):
		for j in range(4, 6):
			if matDownSample[i][j] == 4 and matDownSample[i+1][j+1] == 4:
				features[43] = 1

	if isTesting:
		print("Down Slant - Quad 9: " + str(features[43]))

	# Left Top Arc - Quad 4

	for i in range(2, 4):
		for j in range(0, 2):
			if matDownSample[i+1][j] == 3 and matDownSample[i][j+1] == 2:
				features[44] = 1
			elif (matDownSample[i][j] == 3 and matDownSample[i][j+1] == 2) or (matDownSample[i+1][j] == 3 and matDownSample[i+1][j+1] == 2):
				features[44] = 1
			elif matDownSample[i+1][j] == 1 and matDownSample[i][j+1] == 3:
				features[44] = 1
			elif (matDownSample[i][j] == 1 and matDownSample[i][j+1] == 3) or (matDownSample[i+1][j] == 1 and matDownSample[i+1][j+1] == 3):
				features[44] = 1
			elif matDownSample[i][j] == 1 and matDownSample[i][j+1] == 2:
				features[44] = 1

	if isTesting:
		print("Left Top Arc - Quad 4: " + str(features[44]))

	# Left Top Arc - Quad 5

	for i in range(2, 4):
		for j in range(2, 4):
			if matDownSample[i+1][j] == 3 and matDownSample[i][j+1] == 2:
				features[45] = 1
			elif (matDownSample[i][j] == 3 and matDownSample[i][j+1] == 2) or (matDownSample[i+1][j] == 3 and matDownSample[i+1][j+1] == 2):
				features[45] = 1
			elif matDownSample[i+1][j] == 1 and matDownSample[i][j+1] == 3:
				features[45] = 1
			elif (matDownSample[i][j] == 1 and matDownSample[i][j+1] == 3) or (matDownSample[i+1][j] == 1 and matDownSample[i+1][j+1] == 3):
				features[45] = 1

	if isTesting:
		print("Left Top Arc - Quad 5: " + str(features[45]))

	# Left Top Arc - Quad 2

	for i in range(0, 2):
		for j in range(2, 4):
			if matDownSample[i+1][j] == 3 and matDownSample[i][j+1] == 2:
				features[46] = 1
			elif (matDownSample[i][j] == 3 and matDownSample[i][j+1] == 2) or (matDownSample[i+1][j] == 3 and matDownSample[i+1][j+1] == 2):
				features[46] = 1
			elif matDownSample[i+1][j] == 1 and matDownSample[i][j+1] == 3:
				features[46] = 1
			elif (matDownSample[i][j] == 1 and matDownSample[i][j+1] == 3) or (matDownSample[i+1][j] == 1 and matDownSample[i+1][j+1] == 3):
				features[46] = 1

	if isTesting:
		print("Left Top Arc - Quad 2: " + str(features[46]))

	# Right Top Arc - Quad 2

	for i in range(0, 2):
		for j in range(2, 4):
			if matDownSample[i][j] == 2 and matDownSample[i+1][j+1] == 4:
				features[47] = 1
			elif (matDownSample[i][j] == 2 and matDownSample[i][j+1] == 4) or (matDownSample[i+1][j] == 2 and matDownSample[i+1][j+1] == 4):
				features[47] = 1
			elif matDownSample[i][j] == 4 and matDownSample[i+1][j+1] == 1:
				features[47] = 1
			elif (matDownSample[i][j] == 4 and matDownSample[i][j+1] == 1) or (matDownSample[i+1][j] == 4 and matDownSample[i+1][j+1] == 1):
				features[47] = 1

	if isTesting:
		print("Right Top Arc - Quad 2: " + str(features[47]))

	# Right Top Arc - Quad 5

	for i in range(2, 4):
		for j in range(2, 4):
			if matDownSample[i][j] == 2 and matDownSample[i+1][j+1] == 4:
				features[48] = 1
			elif (matDownSample[i][j] == 2 and matDownSample[i][j+1] == 4) or (matDownSample[i+1][j] == 2 and matDownSample[i+1][j+1] == 4):
				features[48] = 1
			elif matDownSample[i][j] == 4 and matDownSample[i+1][j+1] == 1:
				features[48] = 1
			elif (matDownSample[i][j] == 4 and matDownSample[i][j+1] == 1) or (matDownSample[i+1][j] == 4 and matDownSample[i+1][j+1] == 1):
				features[48] = 1

	if isTesting:
		print("Right Top Arc - Quad 5: " + str(features[48]))

	# Right Top Arc - Quad 6

	for i in range(2, 4):
		for j in range(4, 6):
			if matDownSample[i][j] == 2 and matDownSample[i+1][j+1] == 4:
				features[49] = 1
			elif (matDownSample[i][j] == 2 and matDownSample[i][j+1] == 4) or (matDownSample[i+1][j] == 2 and matDownSample[i+1][j+1] == 4):
				features[49] = 1
			elif matDownSample[i][j] == 4 and matDownSample[i+1][j+1] == 1:
				features[49] = 1
			elif (matDownSample[i][j] == 4 and matDownSample[i][j+1] == 1) or (matDownSample[i+1][j] == 4 and matDownSample[i+1][j+1] == 1):
				features[49] = 1

	if isTesting:
		print("Right Top Arc - Quad 6: " + str(features[49]))

	# Left Bottom Arc - Quad 4

	for i in range(2, 4):
		for j in range(0, 2):
			if matDownSample[i+1][j] == 1 and matDownSample[i][j+1] == 3:
				features[50] = 1
			elif (matDownSample[i][j] == 1 and matDownSample[i][j+1] == 3) or (matDownSample[i+1][j] == 1 and matDownSample[i+1][j+1] == 3):
				features[50] = 1
			elif matDownSample[i+1][j] == 4 and matDownSample[i][j+1] == 2:
				features[50] = 1
			elif (matDownSample[i][j] == 4 and matDownSample[i][j+1] == 2) or (matDownSample[i+1][j] == 4 and matDownSample[i+1][j+1] == 2):
				features[50] = 1

	if isTesting:
		print("Left Bottom Arc - Quad 4: " + str(features[50]))

	# Left Bottom Arc - Quad 8

	for i in range(4, 6):
		for j in range(2, 4):
			if matDownSample[i+1][j] == 1 and matDownSample[i][j+1] == 3:
				features[51] = 1
			elif (matDownSample[i][j] == 1 and matDownSample[i][j+1] == 3) or (matDownSample[i+1][j] == 1 and matDownSample[i+1][j+1] == 3):
				features[51] = 1
			elif matDownSample[i+1][j] == 4 and matDownSample[i][j+1] == 2:
				features[51] = 1
			elif (matDownSample[i][j] == 4 and matDownSample[i][j+1] == 2) or (matDownSample[i+1][j] == 4 and matDownSample[i+1][j+1] == 2):
				features[51] = 1

	if isTesting:
		print("Left Bottom Arc - Quad 8: " + str(features[51]))

	# Right Bottom Arc - Quad 5

	for i in range(2, 4):
		for j in range(2, 4):
			if matDownSample[i+1][j] == 2 and matDownSample[i][j+1] == 3:
				features[52] = 1
			elif (matDownSample[i][j] == 2 and matDownSample[i][j+1] == 3) or (matDownSample[i+1][j] == 2 and matDownSample[i+1][j+1] == 3):
				features[52] = 1
			elif matDownSample[i+1][j] == 3 and matDownSample[i][j+1] == 1:
				features[52] = 1
			elif (matDownSample[i][j] == 3 and matDownSample[i][j+1] == 1) or (matDownSample[i+1][j] == 3 and matDownSample[i+1][j+1] == 1):
				features[52] = 1

	if isTesting:
		print("Right Bottom Arc - Quad 5: " + str(features[52]))

	# Right Bottom Arc - Quad 6

	for i in range(2, 4):
		for j in range(4, 6):
			if matDownSample[i+1][j] == 2 and matDownSample[i][j+1] == 3:
				features[53] = 1
			elif (matDownSample[i][j] == 2 and matDownSample[i][j+1] == 3) or (matDownSample[i+1][j] == 2 and matDownSample[i+1][j+1] == 3):
				features[53] = 1
			elif matDownSample[i+1][j] == 3 and matDownSample[i][j+1] == 1:
				features[53] = 1
			elif (matDownSample[i][j] == 3 and matDownSample[i][j+1] == 1) or (matDownSample[i+1][j] == 3 and matDownSample[i+1][j+1] == 1):
				features[53] = 1

	if isTesting:
		print("Right Bottom Arc - Quad 6: " + str(features[53]))

	# Right Bottom Arc - Quad 8

	for i in range(4, 6):
		for j in range(2, 4):
			if matDownSample[i+1][j] == 2 and matDownSample[i][j+1] == 3:
				features[54] = 1
			elif (matDownSample[i][j] == 2 and matDownSample[i][j+1] == 3) or (matDownSample[i+1][j] == 2 and matDownSample[i+1][j+1] == 3):
				features[54] = 1
			elif matDownSample[i+1][j] == 3 and matDownSample[i][j+1] == 1:
				features[54] = 1
			elif (matDownSample[i][j] == 3 and matDownSample[i][j+1] == 1) or (matDownSample[i+1][j] == 3 and matDownSample[i+1][j+1] == 1):
				features[54] = 1

	if isTesting:
		print("Right Bottom Arc - Quad 8: " + str(features[54]))

	# Top Loop

	if features[0] == 1 and features[2] == 1 and features[4] == 1 and features[6] == 1:
		features[55] = 1

	if features[0] == 1 and features[2] == 1 and features[50] == 1 and features[52] == 1:
		features[55] = 1

	if isTesting:
		print("Top Loop: " + str(features[55]))

	# Bottom Loop

	if (features[1] == 1 or features[44] == 1) and (features[3] == 1 or features[49] == 1) and features[5] == 1 and features[7] == 1:
		features[56] = 1

	if isTesting:
		print("Bottom Loop: " + str(features[56]))

	return features

############################ Train the Neural Network ###################################

imageOne = Image.open('/Users/Bala/Desktop/Handwritten-Digits-for-Testing/boxedone.png')
imageTwo = Image.open('/Users/Bala/Desktop/Handwritten-Digits-for-Testing/boxedtwo.png')
imageThree = Image.open('/Users/Bala/Desktop/Handwritten-Digits-for-Testing/boxedthree.png')
imageFour = Image.open('/Users/Bala/Desktop/Handwritten-Digits-for-Testing/boxedfour.png')
imageFive = Image.open('/Users/Bala/Desktop/Handwritten-Digits-for-Testing/boxedfive.png')
imageSix = Image.open('/Users/Bala/Desktop/Handwritten-Digits-for-Testing/boxedsix.png')
imageSeven = Image.open('/Users/Bala/Desktop/Handwritten-Digits-for-Testing/boxedseven.png')
imageEight = Image.open('/Users/Bala/Desktop/Handwritten-Digits-for-Testing/boxedeight.png')
imageNine = Image.open('/Users/Bala/Desktop/Handwritten-Digits-for-Testing/boxednine.png')
featureListOne = imageProcess(imageOne, False)
featureListTwo = imageProcess(imageTwo, False)
featureListThree = imageProcess(imageThree, False)
featureListFour = imageProcess(imageFour, False)
featureListFive = imageProcess(imageFive, False)
featureListSix = imageProcess(imageSix, False)
featureListSeven = imageProcess(imageSeven, False)
featureListEight = imageProcess(imageEight, False)
featureListNine = imageProcess(imageNine, False)

oneLabel = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
twoLabel = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
threeLabel = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
fourLabel = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
fiveLabel = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
sixLabel = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
sevenLabel = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
eightLabel = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
nineLabel = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

weightMat = np.zeros((57, 10), dtype=float32)
weightMat.fill(0.5)

print("loss for 1: Positive")
for i in range(0, 40):
	digits = np.dot(featureListOne, weightMat)
	expDigits = np.exp(digits)
	totalexp = np.sum(expDigits)
	softmaxProbs = expDigits / totalexp
	softmaxProbs = np.around(softmaxProbs, decimals=3)
	# Cross-Entropy Loss
	loss = 0
	for j in range (0, 10):
		loss = loss + oneLabel[j] * np.log(softmaxProbs[j])

	print(-1*loss)

	for a in range(0, 10):
		for b in range(0, 57):
			#W_ij = W_ij - alpha*(S_i - KroneckerDelta_it)*x_j
			weightMat[b][a] = weightMat[b][a] - 0.01*((softmaxProbs[a] - oneLabel[a])*featureListOne[b])

print(weightMat)

print("loss for 2: Positive")
for i in range(0, 40):
	digits = np.dot(featureListTwo, weightMat)
	expDigits = np.exp(digits)
	totalexp = np.sum(expDigits)
	softmaxProbs = expDigits / totalexp
	softmaxProbs = np.around(softmaxProbs, decimals=3)
	# Cross-Entropy Loss
	loss = 0
	for j in range (0, 10):
		loss = loss + twoLabel[j] * np.log(softmaxProbs[j])

	print(-1*loss)

	for a in range(0, 10):
		for b in range(0, 57):
			#W_ij = W_ij - alpha*(S_i - KroneckerDelta_it)*x_j
			weightMat[b][a] = weightMat[b][a] - 0.01*((softmaxProbs[a] - twoLabel[a])*featureListTwo[b])

print(weightMat)

print("loss for 3: Positive")
for i in range(0, 40):
	digits = np.dot(featureListThree, weightMat)
	expDigits = np.exp(digits)
	totalexp = np.sum(expDigits)
	softmaxProbs = expDigits / totalexp
	softmaxProbs = np.around(softmaxProbs, decimals=3)
	# Cross-Entropy Loss
	loss = 0
	for j in range (0, 10):
		loss = loss + threeLabel[j] * np.log(softmaxProbs[j])

	print(-1*loss)

	for a in range(0, 10):
		for b in range(0, 57):
			#W_ij = W_ij - alpha*(S_i - KroneckerDelta_it)*x_j
			weightMat[b][a] = weightMat[b][a] - 0.01*((softmaxProbs[a] - threeLabel[a])*featureListThree[b])

print("loss for 4: Positive")
for i in range(0, 40):
	digits = np.dot(featureListFour, weightMat)
	expDigits = np.exp(digits)
	totalexp = np.sum(expDigits)
	softmaxProbs = expDigits / totalexp
	softmaxProbs = np.around(softmaxProbs, decimals=3)
	# Cross-Entropy Loss
	loss = 0
	for j in range (0, 10):
		loss = loss + fourLabel[j] * np.log(softmaxProbs[j])

	print(-1*loss)

	for a in range(0, 10):
		for b in range(0, 57):
			#W_ij = W_ij - alpha*(S_i - KroneckerDelta_it)*x_j
			weightMat[b][a] = weightMat[b][a] - 0.01*((softmaxProbs[a] - fourLabel[a])*featureListFour[b])

print("loss for 5: Positive")
for i in range(0, 40):
	digits = np.dot(featureListFive, weightMat)
	expDigits = np.exp(digits)
	totalexp = np.sum(expDigits)
	softmaxProbs = expDigits / totalexp
	softmaxProbs = np.around(softmaxProbs, decimals=3)
	# Cross-Entropy Loss
	loss = 0
	for j in range (0, 10):
		loss = loss + fiveLabel[j] * np.log(softmaxProbs[j])

	print(-1*loss)

	for a in range(0, 10):
		for b in range(0, 57):
			#W_ij = W_ij - alpha*(S_i - KroneckerDelta_it)*x_j
			weightMat[b][a] = weightMat[b][a] - 0.01*((softmaxProbs[a] - fiveLabel[a])*featureListFive[b])

print("loss for 6: Positive")
for i in range(0, 40):
	digits = np.dot(featureListSix, weightMat)
	expDigits = np.exp(digits)
	totalexp = np.sum(expDigits)
	softmaxProbs = expDigits / totalexp
	softmaxProbs = np.around(softmaxProbs, decimals=3)
	# Cross-Entropy Loss
	loss = 0
	for j in range (0, 10):
		loss = loss + sixLabel[j] * np.log(softmaxProbs[j])

	print(-1*loss)

	for a in range(0, 10):
		for b in range(0, 57):
			#W_ij = W_ij - alpha*(S_i - KroneckerDelta_it)*x_j
			weightMat[b][a] = weightMat[b][a] - 0.01*((softmaxProbs[a] - sixLabel[a])*featureListSix[b])

print("loss for 7: Positive")
for i in range(0, 40):
	digits = np.dot(featureListSeven, weightMat)
	expDigits = np.exp(digits)
	totalexp = np.sum(expDigits)
	softmaxProbs = expDigits / totalexp
	softmaxProbs = np.around(softmaxProbs, decimals=3)
	# Cross-Entropy Loss
	loss = 0
	for j in range (0, 10):
		loss = loss + sevenLabel[j] * np.log(softmaxProbs[j])

	print(-1*loss)

	for a in range(0, 10):
		for b in range(0, 57):
			#W_ij = W_ij - alpha*(S_i - KroneckerDelta_it)*x_j
			weightMat[b][a] = weightMat[b][a] - 0.01*((softmaxProbs[a] - sevenLabel[a])*featureListSeven[b])

print("loss for 8: Positive")
for i in range(0, 40):
	digits = np.dot(featureListEight, weightMat)
	expDigits = np.exp(digits)
	totalexp = np.sum(expDigits)
	softmaxProbs = expDigits / totalexp
	softmaxProbs = np.around(softmaxProbs, decimals=3)
	# Cross-Entropy Loss
	loss = 0
	for j in range (0, 10):
		loss = loss + eightLabel[j] * np.log(softmaxProbs[j])

	print(-1*loss)

	for a in range(0, 10):
		for b in range(0, 57):
			#W_ij = W_ij - alpha*(S_i - KroneckerDelta_it)*x_j
			weightMat[b][a] = weightMat[b][a] - 0.01*((softmaxProbs[a] - eightLabel[a])*featureListEight[b])

print("loss for 9: Positive")
for i in range(0, 40):
	digits = np.dot(featureListNine, weightMat)
	expDigits = np.exp(digits)
	totalexp = np.sum(expDigits)
	softmaxProbs = expDigits / totalexp
	softmaxProbs = np.around(softmaxProbs, decimals=3)
	# Cross-Entropy Loss
	loss = 0
	for j in range (0, 10):
		loss = loss + nineLabel[j] * np.log(softmaxProbs[j])

	print(-1*loss)

	for a in range(0, 10):
		for b in range(0, 57):
			#W_ij = W_ij - alpha*(S_i - KroneckerDelta_it)*x_j
			weightMat[b][a] = weightMat[b][a] - 0.01*((softmaxProbs[a] - nineLabel[a])*featureListNine[b])

print(weightMat)

############################## Testing Digits ##############################

imageTest = Image.open('/Users/Bala/Desktop/Handwritten-Digits-for-Testing/testTwo.jpg')

featureListTest = imageProcess(imageTest, True)

print(featureListTest)

expDigitsTest = np.zeros((10,), dtype=float32)
softmaxProbsTest = np.zeros((10,), dtype=float32)

digitsTest = np.dot(featureListTest, weightMat)
print(digitsTest)

expDigitsTest = np.exp(digitsTest)
print(expDigitsTest)

totalexpTest = np.sum(expDigitsTest)
print(totalexpTest)

softmaxProbsTest = expDigitsTest / totalexpTest

softmaxProbsTest = np.around(softmaxProbsTest, decimals=3)
print(softmaxProbsTest)

maxProb = 0.0
maxIndex = -1

for i in range(0, 10):
	if softmaxProbsTest[i] > maxProb:
		maxProb = softmaxProbsTest[i]
		maxIndex = i

print(maxIndex) 

print(weightMat)
