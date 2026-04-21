from flask import (Blueprint,request,jsonify,session,redirect,url_for,flash,send_file,current_app)
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from qdrant_client.models import Filter, FieldCondition, MatchValue, HasIdCondition, FilterSelector
from backend.repositories.clone_repo import load_clones, save_clones
from backend.services.document_service import (process_uploaded_file,store_chunks_in_qdrant,save_document,store_document_reference)
from backend.core.extensions import qdrant_client
from backend.document_processor import LLMDocumentProcessor
from backend.core.extensions import document_processor



data_bp = Blueprint('data', __name__)


@data_bp.route('/preview_chunks/<clone_id>', methods=['POST'])
def preview_chunks(clone_id):
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone or clone['owner'] != session['username']:
        return jsonify({'error': 'Access denied'}), 403
    
    chunks_preview = []
    chunks_data = []
    
    if 'text_content' in request.form and request.form['text_content'].strip():
        text_content = request.form['text_content'].strip()
        chunks = document_processor.process_text_input(text_content)
        
        for idx, chunk in enumerate(chunks):
            chunks_preview.append({
                'id': idx,
                'text': chunk['text'],
                'preview': chunk['text'][:150] + ('...' if len(chunk['text']) > 150 else ''),
                'metadata': chunk.get('metadata', {}),
                'selected': True
            })
        
        chunks_data = chunks
        session[f'preview_chunks_{clone_id}'] = {
            'chunks': chunks_data,
            'source': 'text'
        }
    
    elif 'file' in request.files:
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400

        filename = secure_filename(file.filename)
        
        allowed_extensions = {
            'pdf', 'docx', 'doc', 'pptx', 'ppt',
            'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif',
            'txt', 'csv', 'xlsx', 'xls'
        }
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': f'Unsupported file type. Supported: {", ".join(sorted(allowed_extensions))}'
            }), 400
        
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], f'preview_{filename}')
        file.save(filepath)
        
        try:
            chunks = document_processor.process_file(filepath)

            
            for idx, chunk in enumerate(chunks):
                chunks_preview.append({
                    'id': idx,
                    'text': chunk['text'],
                    'preview': chunk['text'][:150] + ('...' if len(chunk['text']) > 150 else ''),
                    'metadata': chunk.get('metadata', {}),
                    'selected': True,
                    'section': chunk.get('metadata', {}).get('section', 'N/A'),
                    'type': chunk.get('metadata', {}).get('type', 'general')
                })
            
            chunks_data = chunks
            session[f'preview_chunks_{clone_id}'] = {
                'chunks': chunks_data,
                'filename': filename,
                'source': 'file'
            }
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    else:
        return jsonify({'error': 'No text or file provided'}), 400
    
    return jsonify({
        'success': True,
        'chunks': chunks_preview,
        'total_chunks': len(chunks_preview)
    })


@data_bp.route('/upload_selected_chunks/<clone_id>', methods=['POST'])
def upload_selected_chunks(clone_id):
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone or clone['owner'] != session['username']:
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.get_json()
    selected_indices = data.get('selected_indices', [])
    
    preview_data = session.get(f'preview_chunks_{clone_id}')
    
    if not preview_data:
        return jsonify({'error': 'No preview data found. Please upload file again.'}), 400
    
    chunks = preview_data.get('chunks', [])
    
    if not chunks:
        return jsonify({'error': 'No chunks data found.'}), 400
    
    selected_chunks = [chunks[i] for i in selected_indices if i < len(chunks)]
    
    if not selected_chunks:
        return jsonify({'error': 'No chunks selected'}), 400
    
    embedded_chunks = document_processor.chunk_to_embeddings(selected_chunks)
    
    chunk_count = store_chunks_in_qdrant(clone_id, embedded_chunks)
    if chunk_count <= 0:
        return jsonify({
            'error': 'Failed to store chunks in vector database. Please verify Qdrant connection.'
        }), 500
    
    clone['chunk_count'] = clone.get('chunk_count', 0) + chunk_count
    clone['last_updated'] = datetime.now().isoformat()
    save_clones(clones)
    
    session.pop(f'preview_chunks_{clone_id}', None)
    
    return jsonify({
        'success': True,
        'chunks_added': chunk_count,
        'total_chunks': clone['chunk_count']
    })


@data_bp.route('/upload_data/<clone_id>', methods=['POST'])
def upload_data(clone_id):
    try:
        if 'username' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        
        clones = load_clones()
        clone = clones.get(clone_id)
        
        if not clone or clone['owner'] != session['username']:
            return jsonify({'error': 'Access denied'}), 403
        
        embedded_chunks = []
        
        if 'text_content' in request.form and request.form['text_content'].strip():
            text_content = request.form['text_content'].strip()
            chunks = document_processor.process_text_input(text_content)
            embedded_chunks = document_processor.chunk_to_embeddings(chunks)
        
        elif 'file' in request.files:
            file = request.files['file']

            if not file.filename:
                return jsonify({'error': 'No file selected'}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            
            try:
                embedded_chunks, _ = process_uploaded_file(filepath, filename)
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
        else:
            return jsonify({'error': 'No text or file provided'}), 400
        
        if embedded_chunks:
            chunk_count = store_chunks_in_qdrant(clone_id, embedded_chunks)
            if chunk_count <= 0:
                return jsonify({
                    'error': 'Failed to store chunks in vector database. Please verify Qdrant connection.'
                }), 500
            clone['chunk_count'] = clone.get('chunk_count', 0) + chunk_count
            clone['last_updated'] = datetime.now().isoformat()
            save_clones(clones)
            
            return jsonify({
                'success': True,
                'chunks_added': chunk_count,
                'total_chunks': clone['chunk_count']
            })
        
        return jsonify({'error': 'Failed to process content'}), 500
    except Exception as e:
        print(f"Upload data error: {e}")
        return jsonify({'error': str(e)}), 500


@data_bp.route('/upload_document/<clone_id>', methods=['POST'])
def upload_document(clone_id):
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone or clone['owner'] != session['username']:
        return jsonify({'error': 'Access denied'}), 403
    
    if 'document' not in request.files or not request.files['document'].filename:
        return jsonify({'error': 'No document provided'}), 400
    
    description = request.form.get('description', '').strip()
    if not description:
        return jsonify({'error': 'Description is required'}), 400
    
    file = request.files['document']
    
    try:
        file_info = save_document(file, clone_id)
        success = store_document_reference(clone_id, description, file_info)
        
        if success:
            clone['chunk_count'] = clone.get('chunk_count', 0) + 1
            clone['last_updated'] = datetime.now().isoformat()
            save_clones(clones)
            
            return jsonify({
                'success': True,
                'message': 'Document uploaded successfully',
                'filename': file_info['filename'],
                'total_chunks': clone['chunk_count']
            })
        else:
            if os.path.exists(file_info['filepath']):
                os.remove(file_info['filepath'])
            return jsonify({'error': 'Failed to store document reference'}), 500
            
    except Exception as e:
        print(f"Document upload error: {e}")
        return jsonify({'error': str(e)}), 500


@data_bp.route('/delete_chunks/<clone_id>', methods=['POST'])
def delete_chunks(clone_id):
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone or clone['owner'] != session['username']:
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.get_json()
    chunk_ids = data.get('chunk_ids', [])
    
    if not chunk_ids:
        return jsonify({'error': 'No chunks selected'}), 400
    
    try:
        normalized_chunk_ids = [str(chunk_id) for chunk_id in chunk_ids if chunk_id]
        if not normalized_chunk_ids:
            return jsonify({'error': 'No valid chunks selected'}), 400

        delete_filter = Filter(
            must=[
                FieldCondition(
                    key="clone_id",
                    match=MatchValue(value=clone_id)
                ),
                HasIdCondition(
                    has_id=normalized_chunk_ids
                )
            ]
        )

        matched_count = qdrant_client.count(
            collection_name=current_app.config['COLLECTION_NAME'],
            count_filter=delete_filter,
            exact=True
        ).count

        qdrant_client.delete(
            collection_name=current_app.config['COLLECTION_NAME'],
            points_selector=FilterSelector(filter=delete_filter)
        )
        
        clone['chunk_count'] = max(
            0,
            clone.get('chunk_count', 0) - matched_count
        )
        save_clones(clones)
        
        return jsonify({
            'success': True,
            'deleted_count': matched_count,
            'remaining_chunks': clone['chunk_count']
        })
    except Exception as e:
        print(f"Error deleting chunks: {e}")
        return jsonify({'error': str(e)}), 500


@data_bp.route('/download_document/<clone_id>/<unique_filename>')
def download_document(clone_id, unique_filename):
    clones = load_clones()
    clone = clones.get(clone_id)
    
    if not clone:
        return jsonify({'error': 'Clone not found'}), 404
    
    if not clone.get('is_public') and (
        'username' not in session or clone['owner'] != session['username']
    ):
        return jsonify({'error': 'Access denied'}), 403
    
    clone_doc_folder = os.path.join(
        current_app.config['DOCUMENTS_FOLDER'],
        clone_id
    )
    filepath = os.path.join(clone_doc_folder, unique_filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    original_filename = '_'.join(unique_filename.split('_')[1:])
    
    return send_file(
        filepath,
        as_attachment=True,
        download_name=original_filename
    )


@data_bp.route('/api/chunks/<clone_id>', methods=['GET'])
def list_chunks(clone_id):
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    clones = load_clones()
    clone = clones.get(clone_id)
    if not clone or clone.get('owner') != session['username']:
        return jsonify({'error': 'Access denied'}), 403

    try:
        limit = int(request.args.get('limit', 100))
    except ValueError:
        limit = 100
    limit = max(1, min(limit, 500))

    try:
        count_filter = Filter(
            must=[
                FieldCondition(
                    key="clone_id",
                    match=MatchValue(value=clone_id)
                )
            ]
        )

        total_count = qdrant_client.count(
            collection_name=current_app.config['COLLECTION_NAME'],
            count_filter=count_filter,
            exact=True
        ).count

        points, _ = qdrant_client.scroll(
            collection_name=current_app.config['COLLECTION_NAME'],
            scroll_filter=count_filter,
            with_payload=True,
            with_vectors=False,
            limit=limit
        )

        chunks = []
        for point in points:
            payload = point.payload or {}
            text = payload.get('text', '')
            chunks.append({
                'id': str(point.id),
                'text': text,
                'preview': text[:200] + ('...' if len(text) > 200 else ''),
                'metadata': {
                    k: v for k, v in payload.items()
                    if k not in ['clone_id', 'text', 'chunk_index']
                }
            })

        if clone.get('chunk_count') != total_count:
            clone['chunk_count'] = total_count
            clone['last_updated'] = datetime.now().isoformat()
            save_clones(clones)

        return jsonify({
            'chunks': chunks,
            'count': total_count,
            'returned': len(chunks)
        })

    except Exception as e:
        print(f"Error listing chunks: {e}")
        return jsonify({'error': str(e)}), 500


@data_bp.route('/api/documents/<clone_id>', methods=['GET'])
def list_documents(clone_id):
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    clones = load_clones()
    clone = clones.get(clone_id)
    if not clone or clone.get('owner') != session['username']:
        return jsonify({'error': 'Access denied'}), 403

    try:
        points, _ = qdrant_client.scroll(
            collection_name=current_app.config['COLLECTION_NAME'],
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="clone_id",
                        match=MatchValue(value=clone_id)
                    )
                ]
            ),
            with_payload=True,
            with_vectors=False,
            limit=500
        )

        documents = []
        for point in points:
            payload = point.payload or {}
            if not payload.get('is_downloadable', False):
                continue
            unique_filename = payload.get('document_unique_filename')
            if not unique_filename:
                continue
            documents.append({
                'id': str(point.id),
                'filename': payload.get('document_filename', unique_filename),
                'unique_filename': unique_filename,
                'description': payload.get('text', ''),
                'file_size': payload.get('file_size'),
                'download_url': f"/download_document/{clone_id}/{unique_filename}"
            })

        return jsonify({'documents': documents})
    except Exception as e:
        print(f"Error listing documents: {e}")
        return jsonify({'error': str(e)}), 500
