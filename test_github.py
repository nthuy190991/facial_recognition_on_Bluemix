# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 12:41:47 2016

@author: thnguyen
"""

import requests
import json
import httplib
import urllib
import urllib2
import cv2
import numpy as np
from binascii import *
from processRequest import processRequest
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

# username = 'nthuy190991'
# password = 'Thanhhuy123'

username = 'thanhhuynguyenorange'
password = 'GGQN0871abc'

_key_face   = '5d99eec09a7e4b2a916eba7f75671600' # primary key
_url        = 'https://api.projectoxford.ai/face/v1.0/'

def faceDetect(imageData):
    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = _key_face
    headers['Content-Type'] = 'application/octet-stream'
    data = imageData
    json = None

    # Face detection parameters
    params = { 'returnFaceAttributes': 'age,gender,glasses',
               'returnFaceLandmarks': 'true'}
    url_face = _url + 'detect'
    result, status_code = processRequest('post', url_face, json, data, headers, params, 10 )
    return result




def read_image(image_path):
    with open(image_path, 'rb') as f:
        data = f.read()
        f.close()
    return data
    
def write_image(image_path, data):
    with open(image_path, 'wb') as f:
        f.write(data)
        f.close()
    
# simple wrapper function to encode the username & pass
def encodeUserData(user, password):
    return "Basic " + (user + ":" + password).encode("base64").rstrip()
    
# Authentification
# requests.get('https://api.github.com/users/nthuy190991', auth=('nthuy190991', 'Thanhhuy123'))       



params = {'ref':'master'}
# r = requests.get('https://api.github.com/repos/nthuy190991/facial_recognition_on_Bluemix/readme', params=params,  auth=('nthuy190991', 'Thanhhuy123'))
# print r.content

res_list_img = requests.get('https://api.github.com/repos/nthuy190991/facial_recognition_on_Bluemix/contents/face_database_for_oxford', params=params,  auth=('nthuy190991', 'Thanhhuy123'))

# res_list_img = requests.request('GET','https://api.github.com/repos/nthuy190991/facial_recognition_on_Bluemix/contents/face_database_for_oxford', auth=('nthuy190991', 'Thanhhuy123'))

res_list_img = eval(res_list_img.content)
# print res_list_img

list_images = []
for img in res_list_img:
    list_images.append(img['name'])
print list_images  

# for image in list_images:
#     res_img = requests.get('https://api.github.com/repos/nthuy190991/facial_recognition_on_Bluemix/contents/face_database_for_oxford/'+image)
#     res_img = eval(res_img.content)
#     data = res_img['content']
# print res_img

#    binary_data     = a2b_base64(data)
#    data8uint       = np.fromstring(binary_data, np.uint8) # Convert bytestream to an unsigned int array
#    frame   = cv2.imdecode(data8uint, cv2.IMREAD_COLOR)
#    cv2.imwrite('a.jpg', frame)



    
# r = requests.get('https://api.github.com/user', auth=('nthuy190991', 'Thanhhuy123'))
# auth = r.content
# auth = auth.replace('null','None')
# auth = auth.replace('false','False')
# auth = auth.replace('true','True')

# auth = eval(auth)

# print 'Authentification:'
# print auth




# u='nthuy190991'
# p='Thanhhuy123'
# url='https://api.github.com/users/nthuy190991'

# # create the request object and set some headers
# req = urllib2.Request(url)
# req.add_header('Accept', 'application/json')
# req.add_header("Content-type", "application/x-www-form-urlencoded")
# req.add_header('Authorization', encodeUserData(u, p))
# # make the request and print the results
# res = urllib2.urlopen(req)

# print 'Authentification::'
# print res.read()




# print 'Put a new file...'

# image_path = 'face_database_for_oxford/catherie.1.png'
# data = read_image(image_path)
# # print b2a_base64(data)


# headers = {
#     'Content-Type': 'application/json'
# }
# params = {
#     "message": "abc",
#     "committer": {
#     "name": "Nguyen Thanh Huy",
#     "email": "nthuy190991@gmail.com"
#   },
#   "content": b2a_base64(data)
# }
# url = 'https://api.github.com/repos/nthuy190991/facial_recognition_on_Bluemix/contents/face_database_bin/catherie.1.bin'

# resquest = requests.put(url, headers=headers, json=params, auth=(username, password))
# print resquest, resquest.content




# response = requests.get('https://api.github.com/repos/nthuy190991/facial_recognition_on_Bluemix/contents/face_database_for_oxford/huy.0.jpg', auth=(username, password))

# result = response.json() if response.content else None

# data_read = result['content']
# binary_data = a2b_base64(data_read)

# # Method 1
# data8uint = np.fromstring(binary_data, np.uint8) # Convert bytestream to an unsigned int array
# frame     = cv2.imdecode(data8uint, cv2.IMREAD_COLOR)
# print len(frame), len(frame[0]), len(frame[0][0])
# cv2.imwrite('a.jpg', frame)

# # # Method 2
# # data8uint = np.fromstring(binary_data, np.uint8) # Convert bytestream to an unsigned int array
# # img = cv2.cvtColor( cv2.imdecode( data8uint, cv2.IMREAD_COLOR ), cv2.COLOR_BGR2RGB )
# # plt.imsave('aa.jpg', imgRGB)


# # Method 3
# data_read = b2a_base64(binary_data)
# fh = open("imageToSave.png", "wb")
# fh.write(data_read.decode('base64'))
# fh.close()


# res = faceDetect(binary_data)
# print res










# response = requests.get('https://api.github.com/repos/nthuy190991/facial_recognition_on_Bluemix/contents/face_database_for_oxford/emm.2.jpg', auth=(username, password))
# result = response.json() if response.content else None
# sha = result['sha']

# headers = {
#     'Content-Type': 'application/json'
# }
# params = {"message": "delete emm.2.jpg",
#             "committer": {
#             "name": "thanhhuynguyenorange",
#             "email": "thanhhuy.nguyen@orange.com"
#           },
#         "sha": sha}
# raa = requests.delete("https://api.github.com/repos/nthuy190991/facial_recognition_on_Bluemix/contents/face_database_for_oxford/emm.2.jpg", headers=headers,json=params, auth=(username, password))
# print raa, raa.content





# data_read = b2a_base64(binary_data)
# fh = open("imageToSave.jpg", "wb")
# fh.write(data_read.decode('base64'))
# fh.close()

# plt.figure()
# img = mpimg.imread("imageToSave.jpg")
# plt.imshow(img)
# plt.show()

# print binary_data
# print img

# time.sleep(2.5)
# plt.close("all")