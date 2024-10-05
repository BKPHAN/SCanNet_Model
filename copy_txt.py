# # Mở tệp văn bản và đọc các dòng
# with open('./DATA_ROOT/train_info.txt', 'r') as file:
#     lines = file.readlines()
#
# # Tạo một danh sách mới để lưu các dòng đã chuyển đổi
# new_lines = []
#
# # Xử lý từng dòng và tạo 4 dòng mới
# for line in lines:
#     line = line.strip()  # Xóa khoảng trắng thừa
#     base_name = line.replace('.png', '')  # Lấy phần tên trước '.png'
#     # Tạo các dòng mới và thêm vào danh sách
#     for i in range(1, 5):
#         new_line = f"{base_name}_part_{i}.png"
#         new_lines.append(new_line)
#
# # Ghi các dòng đã chuyển đổi vào tệp mới
# with open('./train_info.txt', 'w') as file:
#     for line in new_lines:
#         file.write(line + '\n')
#
# print("Quá trình chuyển đổi hoàn tất!")

import os
import shutil

# Đường dẫn đến thư mục chứa các tệp
folder_path = r'C:\Users\DINHPHAN\Desktop\SCanNet\DATA_ROOT\label1_rgb'
train_folder = os.path.join(folder_path, 'train')
val_folder = os.path.join(folder_path, 'val')

# Tạo các thư mục train, val nếu chưa có
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Lấy danh sách tất cả các tệp trong thư mục
all_files = os.listdir(folder_path)
all_files = [f for f in all_files if os.path.isfile(os.path.join(folder_path, f))]  # Chỉ lấy file, bỏ qua folder

# Tách phần còn lại và 160 tệp cuối
remaining_files = all_files[:-160]
last_160_files = all_files[-160:]

# Tính số lượng tệp cho từng phần theo tỷ lệ 1:4
train_count_remaining = int(len(remaining_files) * 0.8)
val_count_remaining = len(remaining_files) - train_count_remaining

train_count_160 = int(len(last_160_files) * 0.8)
val_count_160 = len(last_160_files) - train_count_160

# Phân chia danh sách tệp cho phần còn lại
train_files_remaining = remaining_files[:train_count_remaining]
val_files_remaining = remaining_files[train_count_remaining:]

# Phân chia danh sách tệp cho 160 tệp cuối
train_files_160 = last_160_files[:train_count_160]
val_files_160 = last_160_files[train_count_160:]

# Gộp tệp train và val từ cả 2 phần
train_files = train_files_remaining + train_files_160
val_files = val_files_remaining + val_files_160


# Hàm để sao chép tệp và ghi tên tệp vào file .txt
def copy_files_and_write_txt(file_list, folder, txt_filename):
    with open(txt_filename, 'w') as f:
        for file_name in file_list:
            # Sao chép tệp sang thư mục mới
            shutil.copy(os.path.join(folder_path, file_name), os.path.join(folder, file_name))
            # Ghi tên tệp vào file .txt
            f.write(file_name + '\n')


# Chia tệp và ghi ra file txt
copy_files_and_write_txt(train_files, train_folder, 'train_files.txt')
copy_files_and_write_txt(val_files, val_folder, 'val_files.txt')

print(f"Chia tệp hoàn thành. Số lượng train: {len(train_files)}, val: {len(val_files)}.")
