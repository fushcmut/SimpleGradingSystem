import random as rd
import os
import shutil

option = {
    1: "A",
    2: "B",
    3: "C",
    4: "D",
}


NUM_QUESTIONS = 30
NUM_STUDENTS = 5
DIR = os.path.join(os.getcwd(), "input")

try:
    if os.path.exists(DIR):
        shutil.rmtree(DIR)
        print(f"Đã xóa thư mục cũ: {DIR}")
except OSError as e:
    print(f"Cảnh báo: Không thể xóa thư mục cũ {DIR}. Lỗi: {e}")

os.makedirs(DIR)
print(f"Đã tạo lại thư mục: {DIR}")

with open(os.path.join(DIR, f"teacher.txt"), "w", encoding="utf-8") as f:
    for i in range(1, NUM_QUESTIONS + 1):
        f.write(f"{i},{option[rd.randint(1,4)]}\n")         # Mode 2
        # f.write(f"{i}{option[rd.randint(1,4)]}\n")          # Mode 3
        # f.write(f"{option[rd.randint(1,4)]}\n")             # Mode 1

for id in range(1, NUM_STUDENTS + 1):
    path = os.path.join(DIR, f"student_{id:0>3}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for i in range(1, NUM_QUESTIONS + 1):
            f.write(f"{i},{option[rd.randint(1,4)]}\n")     # Mode 2
            # f.write(f"{i}{option[rd.randint(1,4)]}\n")      # Mode 3
            # f.write(f"{option[rd.randint(1,4)]}\n")         # Mode 1