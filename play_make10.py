import sys
import mss
import numpy as np
import time
import cv2
import pyautogui as pag
import itertools
from sklearn.neighbors import KNeighborsClassifier


def grab_screen(bbox):
	'''Grab the specified area on the screen'''
	with mss.mss() as sct:
		left, top, width, height = bbox
		grab_area = {'left': left, 'top': top, 'width': width, 'height': height}
		img = sct.grab(grab_area)
		return np.array(img)[:, :, :3]


def process_img(img):
	'''Extract ROIs from cards'''
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, img_bin = cv2.threshold(img_gray, 210, 255, cv2.THRESH_BINARY)
	_, cnts, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = list(filter(lambda x: 20 < cv2.boundingRect(x)[-1] < 70, cnts))
	cnts = sorted(cnts, key=lambda x: (cv2.boundingRect(x)[1] // 10, cv2.boundingRect(x)[0]))
	rois = []
	card_locs = []
	for cnt in cnts:
		x, y, w, h = cv2.boundingRect(cnt)
		card_locs.append([int(x + w/2), int(y + h/2)])
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
		roi = cv2.resize(img_bin[y:y+h, x:x+w], (10, 10)).reshape(1, 100)
		rois.append(roi)
	cv2.imwrite('detection_result.jpg', img)
	card_locs = np.array(card_locs)
	rois = np.array(rois).squeeze()
	return rois, card_locs


def build_knn():
	'''Build a knn classifier'''
	samples = np.loadtxt('train_dataset/train_samples.data', np.uint8)
	labels = np.loadtxt('train_dataset/train_labels.data', np.float32)
	knn = KNeighborsClassifier(n_neighbors=3)
	knn.fit(samples, labels)
	return knn


def find_sum10(nums):
	'''Find a combination of integers which sums to 10'''
	result = [seq for i in range(len(nums), 0, -1) for seq in itertools.combinations(nums, i) if sum(seq) == 10]
	return result


def main():
	area_to_grab = (124, 460, 400, 240)
	knn = build_knn()
	score = 0
	while True:
		img = grab_screen(area_to_grab)
		rois, card_locs = process_img(img.copy())
		nums = knn.predict(rois).astype(int)  # classify numbers on each card
		while len(card_locs) > 0:
			idxs_btm = (np.abs(card_locs[:, 1] - card_locs[-1, 1]) < 5)  # index of the bottom cards
			card_locs_btm = card_locs[idxs_btm].tolist()
			nums_btm = nums[idxs_btm].tolist()
			comb = find_sum10(nums_btm)
			if not comb:
				print('Done!')
				print(f'Your score: {score}')
				sys.exit()
			print(f'Cards: {str(tuple(nums_btm)):13}', 'Tap:', comb[0])
			for num in comb[0]:
				idx_tap = nums_btm.index(num)  # index of the card to tap
				x, y = (card_locs_btm.pop(idx_tap))
				nums_btm.pop(idx_tap)
				pag.click(area_to_grab[0] + x, area_to_grab[1] + area_to_grab[3] - 50)  # tap a card
			score += 1
			card_locs = card_locs[~idxs_btm]  # remove the bottom cards
			nums = nums[~idxs_btm]
			time.sleep(0.085)  # wait for the bottom cards to disappear
		time.sleep(0.36)


if __name__ == '__main__':
	main()
