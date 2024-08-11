import requests
import base64
from io import BytesIO
from PIL import Image
import json


url = "http://localhost:8080/detect_base64"     # url of a web-server
PATH = "images/vid_4_26460.jpg"                 # path to an image


# Read the base64 string from a txt file
# f = open(file='image.txt') # read base64 string from file
# data = f.read()
# f.close()

# Get base64 string directly from an image 
data_img = Image.open(PATH)                                 # read an image
buff = BytesIO()                            
data_img.save(buff, format='JPEG')                          # save an image to a buffer
data = base64.b64encode(buff.getvalue()).decode('utf-8')    # encode and get a string to transfer

data = {'image_file' : data}
response = requests.post(url=url, json=data).json()         # send request to the server

b = response['base64_string']

# print(f'{response}') # print response
with open('response_base64.txt', 'w') as f:                 # write a base64 response of a mask to a separate file
    f.write(b)

b_decoded = BytesIO(base64.b64decode(b))                    # decode response to image bytes

b_img = Image.open(b_decoded)                               # get an Image object from image bytes

b_img.show() # show resulting mask

