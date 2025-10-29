# document_processor.py (NEW - Much Simpler!)
import os
from llama_parse import LlamaParse
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import nest_asyncio
import re

# Required for async operations in Flask
nest_asyncio.apply()

class LLMDocumentProcessor:
    """Use LlamaParse for intelligent document processing"""
    
    def __init__(self, api_key: str, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            verbose=True,
            language="en",
            parsing_instruction="""
Extract and structure the document content intelligently, seperate them so we can split them into chunks and upload them in vector database:

Example-
If the Resume/CV is uploaded structure and seperate the content as:
            contact
            summary
            experience
            education
            skills
            projects
            certifications
            achievements
            references

            
FOR GENERAL DOCUMENTS:
- Maintain heading hierarchy (H1, H2, H3)
- Preserve table structure
- Keep related paragraphs together
- Identify and separate different sections

Output clean, well-structured markdown.
            """
        )
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process ANY document type with LlamaParse
        Supports: PDF, DOCX, DOC, PPTX, JPG, PNG, TIFF, etc.
        """
        try:
            print(f"Processing file with LlamaParse: {file_path}")
            
            # LlamaParse handles ALL file types automatically!
            documents = self.parser.load_data(file_path)
            
            chunks = []
            for doc in documents:
                # Get structured markdown from LlamaParse
                markdown_text = doc.text
                
                # Split by semantic sections (headers)
                sections = self._split_by_markdown_sections(markdown_text)
                
                for section in sections:
                    if section['content'].strip():  # Skip empty sections
                        chunks.append({
                            'text': section['content'],
                            'metadata': {
                                'type': 'llm_parsed',
                                'section': section['header'],
                                'source_file': os.path.basename(file_path)
                            }
                        })
            
            print(f"✓ Extracted {len(chunks)} semantic chunks")
            return chunks
            
        except Exception as e:
            print(f"❌ LlamaParse error: {e}")
            return []
    
    def _split_by_markdown_sections(self, markdown: str) -> List[Dict[str, str]]:
        """Split markdown by headers for semantic chunking"""
        sections = []
        lines = markdown.split('\n')
        
        current_header = "Introduction"
        current_content = []
        
        for line in lines:
            # Detect markdown headers (# or ##)
            if line.strip().startswith('#'):
                # Save previous section
                if current_content:
                    content_text = '\n'.join(current_content).strip()
                    if content_text:  # Only add non-empty sections
                        sections.append({
                            'header': current_header,
                            'content': content_text
                        })
                
                # Start new section
                current_header = line.lstrip('#').strip()
                current_content = [line]  # Include header in content
            else:
                current_content.append(line)
        
        # Add last section
        if current_content:
            content_text = '\n'.join(current_content).strip()
            if content_text:
                sections.append({
                    'header': current_header,
                    'content': content_text
                })
        
        return sections
    
    def process_text_input(self, text: str) -> List[Dict[str, Any]]:
        """Process plain text input (for textarea submissions)"""
        # For text, use simple paragraph-based chunking
        chunks = []
        paragraphs = text.split('\n\n')
        
        for idx, para in enumerate(paragraphs):
            if para.strip():
                chunks.append({
                    'text': para.strip(),
                    'metadata': {
                        'type': 'text_input',
                        'chunk_index': idx
                    }
                })
        
        return chunks
    
    def chunk_to_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert chunks to embeddings"""
        embedded_chunks = []
        
        for chunk in chunks:
            text = chunk['text']
            if not text.strip():
                continue
            
            # Generate embedding
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            
            embedded_chunks.append({
                'text': text,
                'embedding': embedding.tolist(),
                'metadata': chunk.get('metadata', {})
            })
        
        return embedded_chunks
