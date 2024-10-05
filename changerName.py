import os

# Đường dẫn đến thư mục chứa các tệp
folder_path = r'C:\Users\DINHPHAN\Desktop\SCanNet\TEST_DIR\im2'

# Chuỗi cần thay thế
old_date = "20231207"
new_date = "20181008"

# Lấy danh sách tất cả các tệp trong thư mục
for filename in os.listdir(folder_path):
    # Kiểm tra nếu tệp có chứa chuỗi cần thay thế
    if old_date in filename:
        # Tạo tên tệp mới bằng cách thay thế old_date bằng new_date
        new_filename = filename.replace(old_date, new_date)

        # Đường dẫn đầy đủ đến tệp gốc và tệp mới
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)

        # Đổi tên tệp
        os.rename(old_file, new_file)
        print(f"Đã đổi tên: {filename} thành {new_filename}")
