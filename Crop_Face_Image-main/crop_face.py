from mtcnn import MTCNN
import cv2
import os

detector = MTCNN()
def detect_face(image, folder_out, file_name):
    img = cv2.imread(image)
    faces = detector.detect_faces(img)
    for face in faces:
        bounding_box = face['box']
        im  = img[ bounding_box[1]:bounding_box[1]+bounding_box[3],
                 bounding_box[0]:bounding_box[0]+bounding_box[2]]
        file_path = folder_out +"/"+ file_name
        cv2.imwrite(file_path, im)
        
# data_folder_in = "./data_in"
# data_folder_out = "./data_out"

def crop_face_many_folders(data_folder_in, data_folder_out):
    for folder_name in os.listdir(data_folder_in):
        os.mkdir(data_folder_out+"/"+folder_name)
        folder_path = os.path.join(data_folder_in,folder_name)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(data_folder_in,folder_name,image_name)
            folder_out =  data_folder_out + "/" + folder_name
            detect_face(image_path,folder_out,image_name)
            
# image_folder_in = ".\data_in\Amber Heard"
# image_folder_out =".\data_out\Amber"   
        
def crop_face_one_folder(image_folder_in, image_folder_out):
    for image_name in os.listdir(image_folder_in):
            image_path = os.path.join(image_folder_in,image_name)
            detect_face(image_path,image_folder_out,image_name)

 