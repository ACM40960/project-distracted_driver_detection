from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import pandas as pd
from werkzeug.utils import secure_filename
import nbformat
from nbclient import NotebookClient

# Configuration
UPLOAD_FOLDER = 'static/uploads'
EXCEL_LOG = 'submissions.xlsx'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

app = Flask(__name__)
app.secret_key = 'super_secret_key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Simple employee login credentials
EMPLOYEES = {'ashish': 'ashish', 'nishanth': 'nishanth'}

# Utility: check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Homepage (upload form)
@app.route('/')
def home():
    return render_template('index.html')

# Handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    driver_id = request.form['driver_id']
    file = request.files['video']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Log to Excel
        new_entry = pd.DataFrame([[driver_id, filename]], columns=["DriverID", "Filename"])
        if os.path.exists(EXCEL_LOG):
            df = pd.read_excel(EXCEL_LOG)
            df = pd.concat([df, new_entry], ignore_index=True)
        else:
            df = new_entry
        df.to_excel(EXCEL_LOG, index=False)

        flash('✅ Video uploaded and logged successfully!')
        return redirect(url_for('home'))
    else:
        flash('❌ Invalid video file.')
        return redirect(url_for('home'))

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        passwd = request.form['password']
        if uname in EMPLOYEES and EMPLOYEES[uname] == passwd:
            session['username'] = uname
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        flash('❌ Invalid credentials')
    return render_template('login.html')

# Logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('logged_in', None)
    flash("✅ You have been logged out.")
    return redirect(url_for('home'))

# Employee dashboard
@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if not os.path.exists(EXCEL_LOG):
        return render_template('dashboard.html', records=[])

    df = pd.read_excel(EXCEL_LOG)
    return render_template('dashboard.html', records=df.to_dict(orient='records'))

# Analyze video
@app.route('/analyze/<filename>')
def analyze(filename):
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    notebook_path = os.path.join('notebooks', '04_video_pipeline.ipynb')

    with open(notebook_path, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Inject video path into notebook
    nb.cells.insert(0, nbformat.v4.new_code_cell(f"video_path = r'{video_path}'"))

    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()

    # Parse notebook outputs
    result_summary = []
    result_images = []
    for cell in nb.cells:
        if cell.cell_type == 'code' and cell.outputs:
            for out in cell.outputs:
                if out.output_type == 'stream' and "Offence Report" in out.text:
                    lines = out.text.splitlines()
                    for line in lines:
                        if line.strip().startswith("-"):
                            result_summary.append(line.strip().lstrip("- ").strip())
                        if "snapshot:" in line:
                            parts = line.strip().split("snapshot: ")
                            if len(parts) > 1:
                                result_images.append(parts[1])

    return render_template("result.html", filename=filename, summary=result_summary, snapshots=result_images)

if __name__ == '__main__':
    print("✅ Flask app is about to start...")
    app.run(debug=True)
