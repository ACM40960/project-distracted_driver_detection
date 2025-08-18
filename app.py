from flask import Flask, render_template, request, redirect, url_for, flash, session, get_flashed_messages
import os
import pandas as pd
from werkzeug.utils import secure_filename
import nbformat
from nbclient import NotebookClient
from datetime import datetime

# configuring upload paths and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
EXCEL_LOG = 'submissions.xlsx'
FLAGGED_LOG = 'flagged_drivers.xlsx'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

# initializing Flask app
app = Flask(__name__)
app.secret_key = 'super_secret_key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# storing hardcoded employee credentials
EMPLOYEES = {'ashish': 'ashish', 'nishanth': 'nishanth'}

# checking if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# rendering homepage
@app.route('/')
def home():
    return render_template('index.html')

# handling video submission
@app.route('/submit', methods=['POST'])
def submit():
    driver_id = request.form['driver_id']
    file = request.files['video']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        new_entry = pd.DataFrame([[driver_id, filename]], columns=["DriverID", "Filename"])
        if os.path.exists(EXCEL_LOG):
            df = pd.read_excel(EXCEL_LOG)
            df = pd.concat([df, new_entry], ignore_index=True)
        else:
            df = new_entry
        df.to_excel(EXCEL_LOG, index=False)

        flash('Video uploaded and logged successfully!')
        return redirect(url_for('home'))
    else:
        flash('Invalid video file.')
        return redirect(url_for('home'))

# handling employee login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        passwd = request.form['password']
        if uname in EMPLOYEES and EMPLOYEES[uname] == passwd:
            session['username'] = uname
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        flash('Invalid credentials')
    return render_template('login.html')

# logging out employee
@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('logged_in', None)

    # clearing previous flash messages 
    list(get_flashed_messages())

    flash("You have been logged out.")
    return redirect(url_for('home'))


# displaying dashboard with uploaded videos
@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if not os.path.exists(EXCEL_LOG):
        return render_template('dashboard.html', records=[])
    df = pd.read_excel(EXCEL_LOG)
    return render_template('dashboard.html', records=df.to_dict(orient='records'))

# analyzing selected video using notebook
@app.route('/analyze/<filename>')
def analyze(filename):
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    driver_id = None
    if os.path.exists(EXCEL_LOG):
        df = pd.read_excel(EXCEL_LOG)
        row = df[df['Filename'] == filename]
        if not row.empty:
            driver_id = row.iloc[0]['DriverID']
        # removing analyzed video from submissions log
        df = df[df['Filename'] != filename]
        df.to_excel(EXCEL_LOG, index=False)

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    notebook_path = os.path.join('notebooks', '04_video_pipeline.ipynb')

    # injecting video path into notebook and executing it
    with open(notebook_path, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    nb.cells.insert(0, nbformat.v4.new_code_cell(f"video_path = r'{video_path}'"))
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()

    # extracting results from notebook output
    result_summary = []
    result_images = []
    for cell in nb.cells:
        if cell.cell_type == 'code' and cell.outputs:
            for out in cell.outputs:
                if out.output_type == 'stream' and "Offence Report" in out.text:
                    lines = out.text.splitlines()
                    for line in lines:
                        if line.strip().startswith("-"):
                            text = line.strip().lstrip("- ").strip()
                            if "→ Combined snapshot:" in text:
                                text = text.split("→ Combined snapshot:")[0].strip()
                            result_summary.append(text)
                        if "snapshot:" in line:
                            parts = line.strip().split("snapshot: ")
                            if len(parts) > 1:
                                path = parts[1].replace("\\", "/")
                                if path.startswith("static/"):
                                    path = path[len("static/"):]  # making path relative to /static
                                result_images.append(path)

    # pairing each offence with corresponding image
    pairs = list(zip(result_summary, result_images))

    return render_template("result.html", filename=filename, driver_id=driver_id, pairs=pairs)

# flagging driver decision (yes or no)
@app.route('/flag/<filename>/<choice>/<driver_id>')
def flag(filename, choice, driver_id):
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if choice == 'yes':
        new_flagged = pd.DataFrame([[driver_id, filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]],
                                   columns=["DriverID", "Filename", "FlaggedAt"])
        if os.path.exists(FLAGGED_LOG):
            df_flagged = pd.read_excel(FLAGGED_LOG)
            df_flagged = pd.concat([df_flagged, new_flagged], ignore_index=True)
        else:
            df_flagged = new_flagged
        df_flagged.to_excel(FLAGGED_LOG, index=False)

    flash("Decision recorded successfully.")
    return redirect(url_for('dashboard'))

# displaying flagged drivers page
@app.route('/flagged')
def flagged():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if not os.path.exists(FLAGGED_LOG):
        return render_template('flagged.html', records=[])
    df = pd.read_excel(FLAGGED_LOG)
    return render_template('flagged.html', records=df.to_dict(orient='records'))

# running Flask application
if __name__ == '__main__':
    print("Flask app is about to start...")
    app.run(debug=True)
