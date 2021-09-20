'''This loads data in accordance to the standards mentioned in the GW database.
Author: Omar BOUDRAA <o_boudraa@esi.dz>
'''

from glob import glob
import cv2
import os
import encodings
from xml.etree import ElementTree as ET
import numpy as np

from create_punet_label import generate_label
from datetime import datetime

WORD_IMAGE_DIR = 'words/'
XML_DIR = 'xml/'
GW_DIR = 'images/'
transcripts = {}

def append_data(x, y, transcript, data): # need not return anything
    x.append(data[0])
    y.append(data[1])
    transcript.append(data[2])


def load_data():
    time_start = datetime.now()

    #train_rule, valid_rule, test_rule = rule()

    xml_files = glob(XML_DIR+'*.xml')

    x_train = []
    y_train = []
    train_transcript = []
    x_valid = []
    y_valid = []
    valid_transcript = []
    x_test = []
    y_test = []
    test_transcript = []
    global transcripts
    
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        xml_file_name = os.path.splitext(xml_file)[0]
        print(xml_file_name)
        # Change this code to get the corresponsing word image dir
        # image_dir = xml_file.split('/')[-1].split('.')[0].split('-')
        # image_dir = image_dir[0] + '/' + image_dir[0]+'-'+image_dir[1]+ '/'
        image_dir = GW_DIR # + image_dir

        for word_idx, word_elem in enumerate(root.findall('spot')):
        # for word in root.iter('word'):
            img_id = word_elem.attrib['image'].split('.')[0]
            img_name = image_dir+img_id+'.tif'
            img_transcript = word_elem.attrib['word'].lower()
            #img_transcript = word.get('text').lower()

            im = cv2.imread(img_name, 0)
            #height = np.size(im, 0)
            #print(height)
            x = int(word_elem.attrib['x'])
            y = int(word_elem.attrib['y'])
            h = int(word_elem.attrib['h'])
            w = int(word_elem.attrib['w'])
            img = im[y:y+h,x:x+w]
            #height = np.size(img, 0)
            #print(height)
            if img is None: # Some image files are corrupted
                continue

            target = generate_label(img_transcript)
            if sum(target) == 0: # For special characters
                img_transcript = '' # Use a special notation for them

            img = cv2.resize(img, (160, 64))
            img = np.where(img<200, 1, 0)
            img = img[:, :, np.newaxis]

            data = [img, target, img_transcript]

            if xml_file_name == 'xml/gw_cv1_train' or xml_file_name == 'xml/gw_cv2_train':
                append_data(x_train, y_train, train_transcript, data)
            elif xml_file_name == 'xml/gw_cv3_val':
                append_data(x_valid, y_valid, valid_transcript, data)
            elif xml_file_name == 'xml/gw_cv4_test':
                append_data(x_test, y_test, test_transcript, data)

    N = len(x_train) + len(x_valid) + len(x_test)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    train_trainscript = np.array(train_transcript)

    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    valid_transcript = np.array(valid_transcript)

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    test_transcript = np.array(test_transcript)

    print ("Time to fetch data: ", datetime.now() - time_start)

    return (x_train, y_train, train_transcript,
            x_valid, y_valid, valid_transcript,
            x_test, y_test, test_transcript)
