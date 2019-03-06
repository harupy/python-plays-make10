# Python plays Make10

A python script to play Make10 in [Brain Wars](https://play.google.com/store/apps/details?id=jp.co.translimit.brainwars&hl=en)

## Game Rule
The rule of this game is quite simple. Just tap the cards to make a 10.

## How this works
1. Apply contour detection and crop out the portion of each card which contains number
2. Pass the cropped images to a KNN-classifier and recognize the numbers on each card
3. Find a combination of cards which adds up to 10 and tap them

## Requirements
- numpy
- sklearn
- opencv
- mms
- pyautogui


## Demo
<img src="https://user-images.githubusercontent.com/17039389/53886816-34e76d00-4064-11e9-8ad9-7359c1a94a1a.gif" width=80%>
