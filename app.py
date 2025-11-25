import os
import shutil
import time
import numpy as np
from flask import Flask, request, render_template, send_file, redirect, url_for
from werkzeug.utils import secure_filename

# Import toàn bộ các hàm xử lý logic từ file GS.py
# Yêu cầu file GS.py phải là phiên bản mới nhất, có hỗ trợ trả về warnings (tuple)
from GS import teacher_input, students_input, check_answer, to_score_file, to_statistic_file

# --- Cấu hình Thư mục ---
UPLOAD_FOLDER = 'uploads' 
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'txt', 'csv', 'xlsx'} 

# Thiết lập Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# --- Utilities ---
def allowed_file(filename):
    """Check valid file."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup(folder):
    """Deletes the entire directory and its contents, then recreates it."""
    try:
        if os.path.exists(folder):
            # Delete the entire directory tree
            shutil.rmtree(folder) 
    except Exception as e:
        print(f'Error deleting folder {folder}: {e}')
    finally:
        # Recreate the directory to ensure the app can save new files
        os.makedirs(folder, exist_ok=True)


# --- Định nghĩa các Routes (Đường dẫn) ---

@app.route('/', methods=['GET'])
def index():
    """Default route, renders the file upload interface."""
    # Ensure folders are clean and exist upon accessing the homepage
    cleanup(app.config['UPLOAD_FOLDER'])
    cleanup(app.config['RESULT_FOLDER'])
    # Pass an empty warnings list to avoid template errors on initial load
    return render_template('index.html', warnings=[]) 

@app.route('/grade_project', methods=['POST'])
def grade_project():
    """Main route to process grading logic."""
    
    # Clean up before starting
    cleanup(app.config['UPLOAD_FOLDER'])
    cleanup(app.config['RESULT_FOLDER'])
    
    # 1. Check Teacher File
    teacher_file_obj = request.files.get('teacher_file')
    if not teacher_file_obj or teacher_file_obj.filename == '':
        return render_template('index.html', error_message="Teacher's answer key file is required.")

    # 2. Check Student Files
    student_file_objects = request.files.getlist('student_files') 
    if not any(f.filename for f in student_file_objects):
        return render_template('index.html', error_message="Student answer files are required.")

    # 3. Get Configuration Parameters
    try:
        mode = int(request.form.get('mode', 1))
        penalty = float(request.form.get('penalty', 0.0))
        if penalty < 0 or penalty > 1:
            raise ValueError("Penalty value must be between 0.0 and 1.0.")
    except Exception as e:
        return render_template('index.html', error_message=f"Invalid Configuration: {e}")

    # List to store original filenames for the final report
    student_file_info = [] 

    try:
        # --- 4. TEMPORARILY SAVE FILES ON SERVER ---
        # Save Teacher File
        teacher_filename = secure_filename(teacher_file_obj.filename)
        teacher_save_path = os.path.join(app.config['UPLOAD_FOLDER'], teacher_filename)
        teacher_file_obj.save(teacher_save_path) 

        # Save Student Files
        student_paths_for_gs = []
        for file_obj in student_file_objects:
            original_filename = file_obj.filename # Save original name
            
            if file_obj.filename and allowed_file(file_obj.filename):
                # REMOVED TIMESTAMP: Save file using its original name (after securing it)
                student_filename_clean = secure_filename(original_filename) 
                student_save_path = os.path.join(app.config['UPLOAD_FOLDER'], student_filename_clean)
                file_obj.save(student_save_path)
                
                # Store path for GS.py and original name for CSV report
                student_paths_for_gs.append(student_save_path)
                student_file_info.append(original_filename) 
        
        if not student_paths_for_gs:
            return render_template('index.html', error_message="No valid student files were found or processed.")

        # --- 5. CALL GS.PY LOGIC (UNPACKING WARNINGS) ---
        
        # Teacher input - UNPACK (array, warnings)
        teacher_answer, t_warnings = teacher_input(teacher_save_path, mode=mode, contain_header=False)
        num_questions = len(teacher_answer)

        # Student input - UNPACK (array, warnings)
        students_answers, s_warnings = students_input(
            student_paths_for_gs, 
            questions_count=num_questions, 
            mode=mode, 
            contain_header=False
        )

        # Check answers
        results = check_answer(teacher_answer, students_answers) 

        # Calculate scores
        scores = to_score_file(results, students_answer=students_answers, penalty=penalty)  

        # --- 6. GENERATE OUTPUT FILES ---
        output_txt_path = os.path.join(app.config['RESULT_FOLDER'], "statistics.txt")
        output_img_path = os.path.join(app.config['RESULT_FOLDER'], "question_mistake_plot.png")
        
        to_statistic_file(
            results=results,
            round_number=2,
            to_txtfile=True,
            output_path=output_txt_path,
            draw_plot=True,
            figure_path=output_img_path
        )
        
        # Save final score CSV
        score_csv_path = os.path.join(app.config['RESULT_FOLDER'], "scores.csv")
        with open(score_csv_path, 'w', encoding='utf-8') as f:
            f.write("File_Name,Score\n")
            for i, score in enumerate(scores):
                # Use the stored ORIGINAL filename
                original_filename = student_file_info[i] 
                f.write(f"{original_filename},{score:.2f}\n")
        
        # --- 7. RETURN DOWNLOAD PAGE AND WARNINGS ---
        # Collect all warnings
        all_warnings = []
        if t_warnings:
            all_warnings.append({
                'file': teacher_filename,
                'warnings': [(q, desc) for q, desc in t_warnings]
            })
        all_warnings.extend(s_warnings)
        
        return render_template('index.html', 
                               success=True, 
                               score_file="scores.csv",
                               stat_txt="statistics.txt",
                               stat_img="question_mistake_plot.png",
                               num_students=len(scores),
                               warnings=all_warnings)


    except ValueError as ve:
        # Handle validation errors from GS.py
        return render_template('index.html', error_message=f"Data Processing Error: {ve}")
    except Exception as e:
        # Handle generic errors
        return render_template('index.html', error_message=f"An Unknown Error Occurred: {e}")
    finally:
        # Clean up uploaded files after processing (success or failure)
        cleanup(app.config['UPLOAD_FOLDER'])

@app.route('/download/<filename>')
def download_file(filename):
    """Route handles downloading the generated result files."""
    
    # Security check to ensure only generated files are downloaded
    if filename not in ["scores.csv", "statistics.txt", "question_mistake_plot.png"]:
        return "Access denied to this file.", 403

    # Send file from the results folder
    return send_file(
        os.path.join(app.config['RESULT_FOLDER'], filename),
        as_attachment=True,
        download_name=filename 
    )

if __name__ == '__main__':
    app.run(debug=True)