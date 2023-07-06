from mtcnn import MTCNN
import cv2
from PIL import Image
from face_search import search_face

#Resize Image  
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

# Detect face, search face and display reuslts
detector = MTCNN()
def tagging_image(path): 
    im = Image.open(path)
    image = cv2.imread(path) # Đọc ảnh
    faces = detector.detect_faces(image) # Sử dụng MTCNN để phát hiện khuôn mặt trong ảnh
    print("Results:")
    
    # Duyệt qua tất cả các khuôn mặt được phát hiện
    for face in faces:
        if face['confidence'] > 0.8:
            bounding_box = face['box']     # Lấy tọa độ của khuôn mặt
            # Vẽ hình chữ nhật xung quanh khuôn mặt trên ảnh gốc
            cv2.rectangle(image,(bounding_box[0], bounding_box[1]), 
                        (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]), 
                        (0,204,0),2)
            # cắt ảnh khuôn mặt
            crop_img = im.crop((bounding_box[0], bounding_box[1],
                                bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]))
            # Tìm kiếm
            name = search_face(crop_img)
            cv2.putText(image,name, (bounding_box[0], bounding_box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    print("---------------")
    img = ResizeWithAspectRatio(image, width=640, height=640)
    cv2.imshow("Face Recognition", img)
    cv2.waitKey(0)

# TEST

#image_folder_in = "D:/test"
#for image_name in os.listdir(image_folder_in):
#            image_path = os.path.join(image_folder_in,image_name)
#            tagging_image(image_path)

tagging_image("D:/OneDrive - Hanoi University of Science and Technology/Desktop/test1.jpg")
       
#tagging_image("D:/OneDrive - Hanoi University of Science and Technology/Desktop/test2.webp")

#tagging_image("D:/OneDrive - Hanoi University of Science and Technology/Desktop/test3.jpg")


#tagging_image("D:/test/63.jpg")
