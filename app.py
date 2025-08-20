from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
app.secret_key = 'dakshin'

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row  
    return conn

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        account = cursor.fetchone()
        if account:
            flash('Username already exists!', 'error')
        else:
            cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            conn.close()
            return redirect(url_for('login'))
        conn.close()
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['username'] = user['username']
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

def normal_predict(time_alone, stage_fear, social_event, going_outside, drained, friend_circle, post_freq):
    df = pd.read_csv('dataset/personality_dataset.csv')
    df.dropna(inplace=True)

    df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
    df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
    df['Personality'] = df['Personality'].map({'Extrovert': 1, 'Introvert': 0})

    X = df.drop('Personality', axis=1)
    y = df['Personality']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    new_data = [[int(time_alone), stage_fear, int(social_event),
                 int(going_outside), drained, int(friend_circle), int(post_freq)]]

    for i in range(len(new_data[0])):
        if new_data[0][i] == "Yes":
            new_data[0][i] = 1
        elif new_data[0][i] == "No":
            new_data[0][i] = 0

    prediction = model.predict(new_data)
    probability = model.predict_proba(new_data)[0][prediction[0]] * 100

    personality = "Extrovert" if prediction[0] == 1 else "Introvert"

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM suggestions WHERE personality = ?", (personality,))
    tips = cursor.fetchone()
    conn.close()

    return personality, round(probability, 2), tips['tips']

@app.route('/normal', methods=['GET', 'POST'])
def normal():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        time_alone = request.form['time_alone']
        stage_fear = request.form['stage_fear']
        social_event = request.form['social_event']
        going_outside = request.form['going_outside']
        drained = request.form['drained']
        friend_circle = request.form['friend_circle']
        post_freq = request.form['post_freq']

        personality, accuracy, tips = normal_predict(
            time_alone, stage_fear, social_event, going_outside, drained, friend_circle, post_freq
        )

        return render_template('result.html', personality=personality, accuracy=accuracy, tips=tips)

    return render_template('normal.html')

def detailed_predict(inputs):
    df = pd.read_csv('dataset/detailed_personality_dataset.csv')

    df.dropna(subset=['personality_type'], inplace=True)
    df.dropna(inplace=True)
    df['personality_type'] = df['personality_type'].astype(str)

    X = df.drop('personality_type', axis=1)
    y = df['personality_type']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    prediction = model.predict([inputs])
    probability = model.predict_proba([inputs])[0].max() * 100

    personality = prediction[0]

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM suggestions WHERE personality = ?", (personality,))
    tips = cursor.fetchone()
    conn.close()

    return personality, round(probability, 2), tips['tips']

@app.route('/detailed', methods=['GET', 'POST'])
def detailed():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        inputs = [
            int(request.form['social_energy']),
            int(request.form['alone_time_preference']),
            int(request.form['talkativeness']),
            int(request.form['deep_reflection']),
            int(request.form['group_comfort']),
            int(request.form['party_liking']),
            int(request.form['listening_skill']),
            int(request.form['empathy']),
            int(request.form['creativity']),
            int(request.form['organization']),
            int(request.form['leadership']),
            int(request.form['risk_taking']),
            int(request.form['public_speaking_comfort']),
            int(request.form['curiosity']),
            int(request.form['routine_preference']),
            int(request.form['excitement_seeking']),
            int(request.form['friendliness']),
            int(request.form['emotional_stability']),
            int(request.form['planning']),
            int(request.form['spontaneity']),
            int(request.form['adventurousness']),
            int(request.form['reading_habit']),
            int(request.form['sports_interest']),
            int(request.form['online_social_usage']),
            int(request.form['travel_desire']),
            int(request.form['gadget_usage']),
            int(request.form['work_style_collaborative']),
            int(request.form['decision_speed']),
            int(request.form['stress_handling']),
        ]

        personality, accuracy, tips = detailed_predict(inputs)
        return render_template('result.html', personality=personality, accuracy=accuracy, tips=tips)

    return render_template('detailed.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
