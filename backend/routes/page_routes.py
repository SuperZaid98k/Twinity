from flask import (Blueprint,render_template,session,redirect,url_for,flash)
from backend.repositories.clone_repo import load_clones
from backend.repositories.user_repo import load_users

page_bp = Blueprint('page', __name__)


@page_bp.route('/')
def index():
    clones = load_clones()
    public_clones = {k: v for k, v in clones.items() if v.get('is_public', False)}
    return render_template('index.html', clones=public_clones)


@page_bp.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('auth.login'))
    
    users = load_users()
    user = users.get(session['username'])
    clones = load_clones()
    user_clones = {k: v for k, v in clones.items() if k in user.get('clones', [])}
    
    return render_template('dashboard.html', clones=user_clones)


@page_bp.route('/chat/<clone_id>')
def chat_page(clone_id):
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone:
        flash('Clone not found')
        return redirect(url_for('page.index'))
    
    if not clone.get('is_public') and (
        'username' not in session or clone['owner'] != session['username']
    ):
        flash('This clone is private')
        return redirect(url_for('page.index'))
    
    return render_template('chat.html', clone=clone, clone_id=clone_id)


@page_bp.route('/manage_data/<clone_id>')
def manage_data(clone_id):
    if 'username' not in session:
        return redirect(url_for('auth.login'))
    
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone or clone['owner'] != session['username']:
        flash('Clone not found or access denied')
        return redirect(url_for('page.dashboard'))
    
    return render_template(
        'manage_data.html',
        clone=clone,
        clone_id=clone_id,
        chunks=[]
    )
