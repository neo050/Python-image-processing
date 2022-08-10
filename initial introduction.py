#pillow מניפולציות על תמונות ועיבוד תמונה אפשר להשתמש בה לשנות תמונה ופילטרים בסיסיים
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from PIL.Image import Transpose
from skimage import io,img_as_float,img_as_ubyte
import cv2
import glob
from PIL import Image

import czifile

img = Image.open("imgs\\Neoray.jpg")
print (type(img))
# זה לא נופיי איזור אם נרצה לעשות מתמטיקה נצטרך להמיר את זה לנומפיי איזור
img.show()
print(img.getcolors().__str__())
print(img.format)# אמור להראות לנו הודעה שזה פורמט JPG

# איך להמיר תמונה לנומפיי מערך
img1 = np.asarray(img)
print(type(img1))
###############################
#matplotlib
#pyplot

img = mpimg.imread("imgs\\Neoray.jpg")
print(type(img))
#print(img)
print(img.shape)# 1063 שורות 827 עמודות ו3 צבעים
plt.imshow(img)
plt.colorbar()
image = io.imread("imgs\\Neoray.jpg").astype(np.float64)
print(type(image))
print(9)
print(image)
#plt.imshow(image)
image = io.imread("imgs\\Neoray.jpg")
image_float = img_as_float(image)
print("##### image_float \n",image_float)
img =cv2.imread("imgs\\Neoray.jpg")
plt.imshow(img)
###########################################################################

grey_img=cv2.imread("imgs\\Neoray.jpg",0)
color_img=cv2.imread("imgs\\Neoray.jpg",1)
cv2.imshow("Grey Image",grey_img)
cv2.imshow("color Image",color_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#img = czifile.imread("C:\\Users\\neora\\Videos\\WhatsApp Video 2022-03-20 at 19.31.40.mp4")
print(img.shape)

path ="imgs/*"
for file in glob.glob(path):
    print(file)
    a=cv2.imread(file)
    print(a)
    c=cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
    cv2.imshow("Color Image",c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img1 = Image.open("imgs/Neoray.jpg")
print(img1.size)
cropped_img =img1.crop((0,0,30,30))
cropped_img .save("imgs/cropped_img.jpg")
img2=Image.open("imgs/monky.jpg")
print(img2.size)
img2.thumbnail((150,200))
img1_copy=img1.copy()
img1_copy.paste(img2,(50,50))
img1_copy.save("imgs/pasted_image.jpg")
img=Image.open("imgs/monky.jpg")
img90=img.rotate(90,expand=True)
img90.save("imgs/img_rotate.jpg")

img_fliplr=img.transpose(Transpose.FLIP_LEFT_RIGHT)
img_fliplr.save("imgs/FLIP_LEFT_RIGHT.jpg")

img_fliptb=img.transpose(Transpose.FLIP_TOP_BOTTOM)
img_fliptb.save("imgs/FLIP_TOP_BOTTOM.jpg")
grey_img=img.convert("L")
grey_img.save("imgs/grey_monky.jpg")
# לחפש בגוגל  https://pillow.readthedocs.io/en/stable/reference/Image.html

