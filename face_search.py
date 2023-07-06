import faiss
import torch
import os
import pandas as pd
import clip
import cv2
from mtcnn import MTCNN

detector = MTCNN()

def extract_face_features(image_path):
    if not os.path.exists(image_path):
        print("Không tìm thấy ảnh.")
        return None
    
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print("Không thể đọc ảnh.")
        return None
    
    # Sử dụng MTCNN để phát hiện khuôn mặt trong ảnh
    faces = detector.detect_faces(image)
    
    if len(faces) > 0:
        # Chỉ lấy khuôn mặt đầu tiên trong ảnh
        face = faces[0]
        x, y, w, h = face['box']
        
        # Cắt khuôn mặt từ ảnh và chuyển đổi thành đúng kích thước (vd: 128x128)
        face_roi = cv2.resize(image[y:y+h, x:x+w], (128, 128))
        
        # Chuyển đổi khuôn mặt thành vector đặc trưng
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_features = face_gray.flatten()
       
        return face_features
    
    # Nếu không tìm thấy khuôn mặt, trả về None
    return None


# Read file KNN index 
df = pd.read_parquet("D:\OneDrive - Hanoi University of Science and Technology\Desktop\AI\Face_Recognition_Use_AutoFaiss-main\data\embedding_folder\metadata\metadata_0.parquet")
image_list = df["image_path"].tolist()
ind = faiss.read_index("D:\OneDrive - Hanoi University of Science and Technology\Desktop\AI\Face_Recognition_Use_AutoFaiss-main\data\knn.index")

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess =  clip.load("ViT-B/32", device=device)

"""
# Đường dẫn đến thư mục chứa dữ liệu
data_folder = 'D:/OneDrive - Hanoi University of Science and Technology/Desktop/AI/train'

# Tạo đối tượng chỉ mục Faiss
d = 16384  # Số chiều của vector đặc trưng
index = faiss.IndexFlatL2(d)  # Chỉ mục với khoảng cách Euclidean
pca = PCA(n_components=min(d, 1), svd_solver='full')

check = 0
# Duyệt qua tất cả các thư mục con trong thư mục train
for person_folder in os.listdir(data_folder):
    check += 1
    if check > 2:
        break
    person_path = os.path.join(data_folder, person_folder)
    if os.path.isdir(person_path):
        # Duyệt qua tất cả các ảnh khuôn mặt trong thư mục con
        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)
            
            # Đọc ảnh và trích xuất đặc trưng
            face_features = extract_face_features(image_path)  # Trích xuất đặc trưng từ ảnh
    
            # Thêm vector đặc trưng vào chỉ mục
            if face_features is not None:
                print("Số chiều của vector đặc trưng:", face_features.shape[0])
                if face_features.shape[0] == d:
                    index.add(np.array([face_features]))

# Lấy dữ liệu huấn luyện từ chỉ mục
x = index.reconstruct_n(0, index.ntotal)

# Áp dụng PCA cho dữ liệu huấn luyện
if x.shape[0] > 1:
    x_pca = pca.fit_transform(x)
else:
    x_pca = x

# Cập nhật chỉ mục với dữ liệu huấn luyện đã giảm chiều
index.reset()
index.add(x)

# Xây dựng chỉ mục
index.train()
"""

# Search image
def search_face(image):
    # Chuyển đổi ảnh thành tensor và tiền xử lý
    image_tensor = preprocess(image)
    # Trích xuất đặc trưng từ ảnh
    image_features = model.encode_image(torch.unsqueeze(image_tensor.to(device), dim=0))
    # Chuẩn hóa đặc trưng
    image_features /= image_features.norm(dim=-1, keepdim=True)
    # Chuyển đổi tensor thành numpy array
    image_embeddings = image_features.cpu().detach().numpy().astype('float32')
    # Tìm kiếm trong chỉ mục Faiss
    D, I = ind.search(image_embeddings, 1)
    # Kiểm tra độ tương đồng và trả về tên người tương ứng nếu độ tương đồng lớn hơn ngưỡng
    if D[0][0] > 0.8: 
        name = os.path.basename(os.path.dirname(image_list[I[0][0]])) 
        print("Name:",os.path.basename(os.path.dirname(image_list[I[0][0]])))
        print("Similarity:",D[0][0])
        return name

