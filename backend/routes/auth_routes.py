from flask import (Blueprint,render_template,request,redirect,url_for,flash,session,jsonify)
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from backend.repositories.user_repo import (load_users,save_users)

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        full_name = request.form.get('full_name')
        
        users = load_users()
        if username in users:
            flash('Username already exists')
            return redirect(url_for('auth.register'))
        
        users[username] = {
            'password': generate_password_hash(password),
            'full_name': full_name,
            'created_at': datetime.now().isoformat(),
            'clones': []
        }
        save_users(users)
        
        flash('Registration successful! Please log in.')
        return redirect(url_for('auth.login'))
    
    return render_template('register.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        users = load_users()
        user = users.get(username)
        
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            return redirect(url_for('page.dashboard'))
        
        flash('Invalid credentials')
    
    return render_template('login.html')


@auth_bp.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('page.index'))


@auth_bp.route('/api/auth/register', methods=['POST'])
def api_register():
    data = request.get_json(silent=True) or {}
    full_name = (data.get('name') or '').strip()
    username = (data.get('username') or data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    if not full_name or not username or not password:
        return jsonify({'error': 'Name, username, and password are required'}), 400

    users = load_users()
    if username in users:
        return jsonify({'error': 'User already exists'}), 409

    users[username] = {
        'password': generate_password_hash(password),
        'full_name': full_name,
        'created_at': datetime.now().isoformat(),
        'clones': []
    }
    save_users(users)
    session['username'] = username

    return jsonify({
        'success': True,
        'user': {
            'name': full_name,
            'username': username,
            'email': username
        }
    }), 201


@auth_bp.route('/api/auth/login', methods=['POST'])
def api_login():
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    users = load_users()
    user = users.get(username)

    if not user or not check_password_hash(user['password'], password):
        return jsonify({'error': 'Invalid credentials'}), 401

    session['username'] = username
    return jsonify({
        'success': True,
        'user': {
            'name': user.get('full_name', username),
            'username': username,
            'email': username
        }
    })


@auth_bp.route('/api/auth/me', methods=['GET'])
def api_me():
    username = session.get('username')
    if not username:
        return jsonify({'authenticated': False}), 401

    users = load_users()
    user = users.get(username)
    if not user:
        session.pop('username', None)
        return jsonify({'authenticated': False}), 401

    return jsonify({
        'authenticated': True,
        'user': {
            'name': user.get('full_name', username),
            'username': username,
            'email': username
        }
    })


@auth_bp.route('/api/auth/logout', methods=['POST'])
def api_logout():
    session.pop('username', None)
    return jsonify({'success': True})
