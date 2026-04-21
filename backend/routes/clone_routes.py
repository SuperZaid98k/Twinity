from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    jsonify
)

import uuid
from datetime import datetime

from backend.repositories.user_repo import load_users, save_users
from backend.repositories.clone_repo import load_clones, save_clones
from backend.utils.phone_utils import normalize_phone_number
from backend.core.extensions import qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
import shutil
import os
from flask import current_app


clone_bp = Blueprint('clone', __name__)


def _serialize_clone(clone_id, clone):
    return {
        'id': clone_id,
        'name': clone.get('name', ''),
        'description': clone.get('description', ''),
        'is_public': bool(clone.get('is_public', False)),
        'owner': clone.get('owner'),
        'owner_full_name': clone.get('owner_full_name', clone.get('owner', '')),
        'phone_number': clone.get('phone_number'),
        'chunk_count': clone.get('chunk_count', 0),
        'created_at': clone.get('created_at'),
        'last_updated': clone.get('last_updated')
    }


@clone_bp.route('/create_clone', methods=['GET', 'POST'])
def create_clone():
    if 'username' not in session:
        return redirect(url_for('auth.login'))
    
    if request.method == 'POST':
        clone_name = request.form.get('clone_name')
        description = request.form.get('description')
        is_public = request.form.get('is_public') == 'on'
        
        clone_id = str(uuid.uuid4())
        
        users = load_users()
        owner_full_name = users[session['username']].get(
            'full_name',
            session['username']
        )
        
        clones = load_clones()
        clones[clone_id] = {
            'name': clone_name,
            'description': description,
            'is_public': is_public,
            'owner': session['username'],
            'owner_full_name': owner_full_name,
            'created_at': datetime.now().isoformat(),
            'chunk_count': 0
        }
        save_clones(clones)
        
        users[session['username']]['clones'].append(clone_id)
        save_users(users)
        
        flash(f'Clone "{clone_name}" created successfully!')
        return redirect(url_for('clone.edit_clone', clone_id=clone_id))
    
    return render_template('create_clone.html')


@clone_bp.route('/edit_clone/<clone_id>', methods=['GET', 'POST'])
def edit_clone(clone_id):
    if 'username' not in session:
        return redirect(url_for('auth.login'))
    
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone or clone['owner'] != session['username']:
        flash('Clone not found or access denied')
        return redirect(url_for('page.dashboard'))
    
    return render_template(
        'edit_clone.html',
        clone=clone,
        clone_id=clone_id
    )


@clone_bp.route('/enable_whatsapp/<clone_id>', methods=['POST'])
def enable_whatsapp(clone_id):
    if 'username' not in session:
        return redirect(url_for('auth.login'))
    
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone or clone['owner'] != session['username']:
        flash('Not allowed')
        return redirect(
            url_for('clone.edit_clone', clone_id=clone_id)
        )

    enable = bool(request.form.get('enable_whatsapp'))
    phone = request.form.get('phone_number', '').strip()
    
    if enable and phone:
        normalized_phone = normalize_phone_number(phone)
        
        if normalized_phone:
            clone['phone_number'] = normalized_phone
            save_clones(clones)
            flash(
                f'✅ WhatsApp enabled! '
                f'Number saved as: {normalized_phone}'
            )
        else:
            flash(
                '❌ Invalid phone number format. '
                'Use format: +919876543210 or 9876543210'
            )
            return redirect(
                url_for('clone.edit_clone', clone_id=clone_id)
            )
    else:
        clone.pop('phone_number', None)
        save_clones(clones)
        flash('WhatsApp integration disabled.')
    
    return redirect(
        url_for('clone.edit_clone', clone_id=clone_id)
    )


@clone_bp.route('/delete_clone/<clone_id>', methods=['POST'])
def delete_clone(clone_id):
    if 'username' not in session:
        return redirect(url_for('auth.login'))
    
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone or clone['owner'] != session['username']:
        flash('Access denied or clone not found.')
        return redirect(url_for('page.dashboard'))
    
    try:
        qdrant_client.delete(
            collection_name=current_app.config['COLLECTION_NAME'],
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="clone_id",
                            match=MatchValue(value=clone_id)
                        )
                    ]
                )
            )
        )
        
        clone_folder = os.path.join(
            current_app.config['DOCUMENTS_FOLDER'],
            clone_id
        )
        if os.path.exists(clone_folder):
            shutil.rmtree(clone_folder)
        
        del clones[clone_id]
        save_clones(clones)
        
        users = load_users()
        if session['username'] in users:
            if clone_id in users[session['username']].get('clones', []):
                users[session['username']]['clones'].remove(clone_id)
                save_users(users)
                
        flash('Chatbot and its data have been successfully deleted.')
        
    except Exception as e:
        print(f"Deletion error: {e}")
        flash('An error occurred while deleting the data.')
        
    return redirect(url_for('page.dashboard'))


@clone_bp.route('/api/clones', methods=['GET'])
def api_list_my_clones():
    username = session.get('username')
    if not username:
        return jsonify({'error': 'Not authenticated'}), 401

    users = load_users()
    user = users.get(username)
    if not user:
        session.pop('username', None)
        return jsonify({'error': 'Not authenticated'}), 401

    clones = load_clones()
    user_clone_ids = set(user.get('clones', []))
    user_clones = [
        _serialize_clone(clone_id, clone_data)
        for clone_id, clone_data in clones.items()
        if clone_id in user_clone_ids and clone_data.get('owner') == username
    ]
    return jsonify({'clones': user_clones})


@clone_bp.route('/api/clones/public', methods=['GET'])
def api_list_public_clones():
    clones = load_clones()
    public_clones = [
        _serialize_clone(clone_id, clone_data)
        for clone_id, clone_data in clones.items()
        if clone_data.get('is_public', False)
    ]
    return jsonify({'clones': public_clones})


@clone_bp.route('/api/clones', methods=['POST'])
def api_create_clone():
    username = session.get('username')
    if not username:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.get_json(silent=True) or {}
    clone_name = (data.get('name') or '').strip()
    description = (data.get('description') or '').strip()
    is_public = bool(data.get('is_public', False))

    if not clone_name:
        return jsonify({'error': 'Clone name is required'}), 400

    users = load_users()
    user = users.get(username)
    if not user:
        session.pop('username', None)
        return jsonify({'error': 'Not authenticated'}), 401

    clone_id = str(uuid.uuid4())
    owner_full_name = user.get('full_name', username)

    clones = load_clones()
    clones[clone_id] = {
        'name': clone_name,
        'description': description,
        'is_public': is_public,
        'owner': username,
        'owner_full_name': owner_full_name,
        'created_at': datetime.now().isoformat(),
        'chunk_count': 0
    }
    save_clones(clones)

    user.setdefault('clones', []).append(clone_id)
    users[username] = user
    save_users(users)

    return jsonify({
        'success': True,
        'clone': _serialize_clone(clone_id, clones[clone_id])
    }), 201


@clone_bp.route('/api/clones/<clone_id>/visibility', methods=['PATCH'])
def api_update_clone_visibility(clone_id):
    username = session.get('username')
    if not username:
        return jsonify({'error': 'Not authenticated'}), 401

    clones = load_clones()
    clone = clones.get(clone_id)
    if not clone or clone.get('owner') != username:
        return jsonify({'error': 'Clone not found or access denied'}), 404

    data = request.get_json(silent=True) or {}
    if 'is_public' not in data:
        return jsonify({'error': 'is_public is required'}), 400

    clone['is_public'] = bool(data.get('is_public'))
    clone['last_updated'] = datetime.now().isoformat()
    save_clones(clones)

    return jsonify({
        'success': True,
        'clone': _serialize_clone(clone_id, clone)
    })


@clone_bp.route('/api/clones/<clone_id>', methods=['GET'])
def api_get_clone(clone_id):
    username = session.get('username')
    if not username:
        return jsonify({'error': 'Not authenticated'}), 401

    clones = load_clones()
    clone = clones.get(clone_id)
    if not clone or clone.get('owner') != username:
        return jsonify({'error': 'Clone not found or access denied'}), 404

    return jsonify({'clone': _serialize_clone(clone_id, clone)})


@clone_bp.route('/api/clones/<clone_id>/whatsapp', methods=['PATCH'])
def api_update_whatsapp(clone_id):
    username = session.get('username')
    if not username:
        return jsonify({'error': 'Not authenticated'}), 401

    clones = load_clones()
    clone = clones.get(clone_id)
    if not clone or clone.get('owner') != username:
        return jsonify({'error': 'Clone not found or access denied'}), 404

    data = request.get_json(silent=True) or {}
    enabled = bool(data.get('enabled', False))
    phone = (data.get('phone_number') or '').strip()

    if enabled:
        if not phone:
            return jsonify({'error': 'phone_number is required when enabling WhatsApp'}), 400

        normalized_phone = normalize_phone_number(phone)
        if not normalized_phone:
            return jsonify({'error': 'Invalid phone number format'}), 400

        clone['phone_number'] = normalized_phone
    else:
        clone.pop('phone_number', None)

    clone['last_updated'] = datetime.now().isoformat()
    save_clones(clones)

    return jsonify({
        'success': True,
        'clone': _serialize_clone(clone_id, clone)
    })


@clone_bp.route('/api/clones/<clone_id>', methods=['DELETE'])
def api_delete_clone(clone_id):
    username = session.get('username')
    if not username:
        return jsonify({'error': 'Not authenticated'}), 401

    clones = load_clones()
    clone = clones.get(clone_id)
    if not clone or clone.get('owner') != username:
        return jsonify({'error': 'Clone not found or access denied'}), 404

    try:
        qdrant_client.delete(
            collection_name=current_app.config['COLLECTION_NAME'],
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="clone_id",
                            match=MatchValue(value=clone_id)
                        )
                    ]
                )
            )
        )

        clone_folder = os.path.join(
            current_app.config['DOCUMENTS_FOLDER'],
            clone_id
        )
        if os.path.exists(clone_folder):
            shutil.rmtree(clone_folder)

        del clones[clone_id]
        save_clones(clones)

        users = load_users()
        if username in users and clone_id in users[username].get('clones', []):
            users[username]['clones'].remove(clone_id)
            save_users(users)

        return jsonify({'success': True})

    except Exception as e:
        print(f"Deletion error: {e}")
        return jsonify({'error': str(e)}), 500
