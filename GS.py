# Require library for this project
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
import subprocess
import time

def wrapper(func):
    def inner(*args, **kwargs):
        name = func.__name__
        print(f"{' ' + name + ' started ':-^120}")
        func(*args, **kwargs)
        print(f"{' ' + name + ' done ':-^120}")
    return inner
    

# ------------------------------------------------
# -------------------ANSWER-----------------------
# ------------------------------------------------

class Input(ABC):
    """
    Abstract base class for all input types (e.g., Txt, Image).
    """
    @abstractmethod
    def __init__(self, source: Any) -> None:
        """
        Initialize input object with given source (file path, image object, etc.)
        """
        pass

    @abstractmethod
    def convert(self) -> tuple[np.array, list]:
        """
        Convert the input content into a NumPy array representation (answers only) and return warnning lists.
        """
        pass

    @abstractmethod
    def preprocess(self):
        """
        Clean data, handling errors
        Warning if solvable problems
        """
        pass


class File(Input):
    def __init__(self, path : str, mode : int = 2, contain_header : bool = False):
        """
        3 Mode
        Mode 1 (Data input format: Answer only)
        Mode 2 (Data input format: Question number + sep + Answer) aka CSV file
        Mode 3 (Data input format: Question number + Answer)
        """
        super().__init__(path)
        self.path = path
        self.mode = mode
        self.contain_header = contain_header
        self.raw_data = None
        self.processed_data = None


    def convert(self):
        """
        Returns a tuple of (processed_data: np.array, warnings: list).
        """
        if not self.path.endswith(('.csv', '.txt', '.xlsx')):
             raise ValueError("Unsupported file format. Please use .csv, .txt, or .xlsx file format.")
        
        try:
            # 1. Đọc file
            if self.path.endswith('.csv') or self.path.endswith('.txt'):
                self.raw_data = pd.read_csv(
                    self.path, 
                    header=0 if self.contain_header else None,
                    sep=','
                )
            elif self.path.endswith('.xlsx'):
                self.raw_data = pd.read_excel(
                    self.path, 
                    header=0 if self.contain_header else None
                )
            
            # 2. Gọi preprocess
            warnings = self.preprocess()
                    
            # 3. Return clean data and warnings
            return self.processed_data, warnings

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at path: {self.path}")
        except Exception as e:
            raise Exception(f"{e}")


    def preprocess(self) -> list[int, str]:
        if self.raw_data is None:
            raise RuntimeError("Call convert() first.")

        df = self.raw_data.copy()
        
        # 1. Tách cột cho mode 3
        if self.mode == 3:
            if df.shape[1] != 1:
                raise ValueError("Mode 3 expects exactly 1 column.")

            col = df.iloc[:, 0].astype(str).str.upper().str.strip()

            extracted = col.str.extract(r'(?i)^\s*(\d+)\s*([A-Z])\s*$')

            df['Question'] = extracted[0].fillna('_')
            df['Answer']   = extracted[1].fillna('_')

        elif self.mode == 2:
            if df.shape[1] != 2:
                raise ValueError("Mode 2 expects exactly 2 columns.")
            df.columns = ['Question', 'Answer']

        elif self.mode == 1:
            if df.shape[1] != 1:
                raise ValueError("Mode 1 expects exactly 1 column.")
            df.columns = ['Answer']

        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        # 2. Dò hết cột Question (Mode 2 và 3)
        warnings = []
        if self.mode == 2 or self.mode == 3:
            df['Q_num'] = pd.to_numeric(df['Question'], errors='coerce')
            
            nan_rows = df[df['Q_num'].isna()].index
            if len(nan_rows) > 0:
                df = df.drop(index=nan_rows).reset_index(drop=True)

            # ---- A. 1 Câu hỏi có nhiều Câu trả lời ----
            dup = df['Q_num'].dropna()

            duplicated_q = dup[dup.duplicated()].unique()

            for q in duplicated_q:
                warnings.append(
                    (int(q), "Duplicate answers found.")
                )

                # Thay bằng - và cho drop hết những thằng còn lại
                mask = df['Q_num'] == q
                df.loc[mask, 'Answer'] = "-"

                drop_idx = df[mask].index[1:]
                df = df.drop(index=drop_idx)

            # ---- B. Kiểm tra và bổ sung missing question ----
            dup = df['Q_num']

            q_min, q_max = int(dup.min()), int(dup.max())
            expected = set(range(q_min, q_max + 1))
            existing = set(dup.astype(int))

            missing = sorted(expected - existing)

            for q in missing:
                warnings.append((q, "Missing question auto-filled '_'"))

            if missing:
                missing_rows = pd.DataFrame({
                    'Question': [str(x) for x in missing],
                    'Answer': ['_'] * len(missing),
                    'Q_num': missing
                })
                df = pd.concat([df, missing_rows], ignore_index=True)

            df = df.sort_values('Q_num')
            df = df.drop(columns=['Q_num', 'Question'])

        # 3. Fillna bằng '_' và return
        df = df.fillna('_') 

        for col in df.columns:
            df[col] = df[col].astype(str)

        self.processed_data = df.to_numpy().flatten()

        return warnings

# ------------------------------------------------
# -------------------ANSWER-----------------------
# ------------------------------------------------

# ------------------------------------------------
# -------------------PROCESS----------------------
# ------------------------------------------------

def teacher_input(path: str, mode: int = 2, contain_header: bool = False) -> tuple[np.array, list]:
    if not path:
        raise ValueError("Teacher file path must not be empty.")

    file = File(path, mode=mode, contain_header=contain_header)

    correct, Warnings = file.convert()
    correct = np.array(correct, dtype=str)

    if "_" in correct:
        raise ValueError("Teacher file contains invalid or missing answers ('_').")

    return correct, Warnings

def students_input(
    paths: list[str], 
    questions_count: int,
    mode: int = 2, 
    contain_header: bool = False
) -> tuple[np.array, list]:

    if not paths:
        raise ValueError("No student file paths provided.")

    students_answers = []
    all_warnings = []

    for path in paths:
        file = File(path, mode=mode, contain_header=contain_header)
        arr, warnings = file.convert()
        arr = np.array(arr, dtype=str)
        if warnings:
            all_warnings.append({
                'file': os.path.basename(path),
                "warnings": warnings,
            })

        if len(arr) > questions_count:
            all_warnings.append({
                'file': os.path.basename(path),
                "warnings": [("N/A", f"File contains {len(arr)} answers, but only {questions_count} questions expected. Extra answers were skipped.")],
            })
            arr = arr[:questions_count]

        students_answers.append(arr)

    normalized = []
    for ans in students_answers:
        if len(ans) < questions_count:
            pad = np.full(questions_count - len(ans), "_", dtype=str)
            ans = np.concatenate([ans, pad])
        normalized.append(ans)

    return np.vstack(normalized, dtype='str'), all_warnings

def check_answer(correct_answer : np.array, student_answers : np.array) -> np.array:
    correct_answer = correct_answer.reshape(1, -1)

    skipped_mask = (student_answers == "_")
    result = (student_answers == correct_answer) & (~skipped_mask)
    return result

def to_score_file(result : np.array, students_answer : np.array, penalty : float = 0):
    result = np.array(result, dtype=bool)
    total_questions = result.shape[1]

    correct_count = result.sum(axis=1)
    
    skipped_mask = (students_answer == "_")
    skipped_count = skipped_mask.sum(axis=1)

    wrong_count = total_questions - correct_count - skipped_count

    scores = (correct_count / total_questions) - (wrong_count * penalty / total_questions)

    scores = np.maximum(scores, 0)
    scores = scores * 10 # Thang 10
    return scores

def to_statistic_file(
    results : np.array,
    round_number : int = 1,
    to_txtfile: bool = True,
    output_path: Optional[str] = None,
    draw_plot: bool = False,
    figure_path: Optional[str] = None,
) -> dict[int, float]:
    
    results = np.array(results, dtype=bool)
    S, Q = results.shape

    # 1. % sai
    wrong_count = (~results).sum(axis=0)
    wrong_percent = {q+1: round(wrong_count[q] / S * 100, round_number) for q in range(Q)}

    # 2. Export to txt file
    if to_txtfile:
        if output_path is None:
            save_path = os.path.join(os.getcwd(), "statistics.txt")
        else:
            # nếu là folder → thêm file name
            save_path = output_path
            if os.path.isdir(output_path):
                save_path = os.path.join(output_path, "statistics.txt")
        with open(save_path, "w", encoding="utf-8") as f:
            for q, pct in wrong_percent.items():
                f.write(f"Question {q:3}: {pct:6.2f}% students answer incorrectly.\n")
        # print(f"Statistics saved to {save_path}")

    # 3. Draw bar plot
    if draw_plot:
        plt.figure(figsize=(10, 5))
        questions = list(wrong_percent.keys())
        percents = list(wrong_percent.values())

        plt.bar(questions, percents, color="skyblue", edgecolor="black")
        plt.xlabel("Question")
        plt.ylabel("Wrong Answer (%)")
        plt.xticks(questions)
        plt.yticks(np.arange(0, 101, 10))
        plt.title("Question Mistake Statistics")
        plt.grid(axis='y', linestyle='--', linewidth=0.25, alpha=0.7)

        # Show % trên cột
        for i, pct in enumerate(percents):
            plt.text(
                questions[i], pct + 1,
                f"{pct:.1f}%", ha='center', va='bottom', fontsize=8
            )

        # Save or show
        if figure_path:
            img_path = figure_path
            if os.path.isdir(figure_path):
                img_path = os.path.join(figure_path, "statistics.png")
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            # print(f"Figure saved to {img_path}")
            plt.close()

    return wrong_percent

def get_user_path() -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    root = tk.Tk()
    root.withdraw()  # ẩn window chính

    student_folder_path = filedialog.askdirectory(title="Choose the students' folder") or None
    teacher_path = filedialog.askopenfilename(
        title="Choose the teacher's answer file (.txt)", 
        filetypes=[
            ("Text files", "*.txt"),
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx"),
            ("All supported", "*.txt *.csv *.xlsx")
        ]
    ) or None
    output_path = filedialog.asksaveasfilename(
        title="Choose the folder to save note",
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt")]
    ) or None
    output_img = filedialog.asksaveasfilename(
        title="Choose the folder to save image",
        defaultextension=".png",
        filetypes=[("Img files", "*.png")]
    ) or None

    return student_folder_path, teacher_path, output_path, output_img

# ------------------------------------------------
# -------------------PROCESS----------------------
# ------------------------------------------------

def main():
    
    # Random test
    os.system("cls")
    subprocess.run(["python", "generate.py"])

    # ----------------- Lấy đường dẫn -----------------
    # student_folder, teacher_file, output_txt, output_img = get_user_path()

    dir = os.getcwd()
    student_folder = os.path.join(dir, "input\\")
    teacher_file = os.path.join(dir, "input\\teacher.txt")
    output_txt = os.path.join(dir, "output\\score.txt")
    output_img = os.path.join(dir, "output\\statistic.png")
    

    print("-"*50)
    print(f"{' Scoring Begin ':-^50}")
    print("-"*50)
    print()

    start = time.time()

    # ----------------- Teacher input -----------------
    teacher_answer, warning1 = teacher_input(teacher_file, mode=2, contain_header=False)
    num_questions = len(teacher_answer)

    # ----------------- Student input -----------------
    student_files = sorted(
        [
            os.path.join(student_folder, f)
            for f in os.listdir(student_folder)
            if f.endswith((".txt", ".csv", ".xlsx"))
        ]
    )
    students_answers, warning2 = students_input(student_files, mode=2, questions_count=num_questions, contain_header=False)

    # ----------------- Check answers -----------------
    results = check_answer(teacher_answer, students_answers)

    print(warning1, warning2, sep="\n\n")

    # ----------------- Tính điểm -----------------
    scores = to_score_file(results, penalty=0.1)  # ví dụ penalty 0.1
    # for i, score in enumerate(scores):
    #     print(f"{os.path.basename(student_files[i]):<20} Score: {score:>6.2f}")

    # ----------------- Thống kê -----------------
    to_statistic_file(
        results=results,
        round_number=1,
        to_txtfile=bool(output_txt),
        output_path=output_txt,
        draw_plot=True,
        figure_path=output_img
    )

    end = time.time()

    print()
    print("-"*50)
    print(f"{' Scoring End ':-^50}")
    print("-"*50)

    print(f"\nProcess finished in {end - start:.2f}sec.\n")

if __name__ == "__main__":
    main()