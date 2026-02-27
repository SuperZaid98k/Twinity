import socket
import time
from datetime import datetime

from backend.core.extensions import qdrant_client
from backend.repositories.clone_repo import load_clones
from backend.services.whatsapp_service import send_whatsapp_message
from backend.config import Config

COLLECTION_NAME = Config.COLLECTION_NAME


def delete_expired_chunks_job():
    """Background job to delete expired chunks with proper error handling"""
    retry_count = 0
    max_retries = 3
    
    while True:
        try:
            # Check if we have internet connectivity
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=5)
            except OSError:
                print("⚠️  No internet connection. Skipping expired chunk cleanup.")
                time.sleep(900)  # Wait 15 min and retry
                continue
            
            now = datetime.now().date()
            
            # Scroll through all points
            scroll_result = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=None,
                with_payload=True,
                with_vectors=False,
                limit=200
            )
            points = scroll_result[0]
            
            expired_ids = []
            chunks_by_clone = {}
            
            for point in points:
                if point.payload is None:
                    continue
                exp_date = point.payload.get('expiry_date')
                cid = point.payload.get('clone_id')
                
                if exp_date and exp_date != "PERMANENT":
                    try:
                        expiry_datetime = datetime.strptime(
                            exp_date,
                            "%Y-%m-%d"
                        ).date()
                        if expiry_datetime < now:
                            expired_ids.append(point.id)
                            if cid not in chunks_by_clone:
                                chunks_by_clone[cid] = []
                            chunks_by_clone[cid].append(
                                point.payload.get('text', '')[:80]
                            )
                    except ValueError as e:
                        print(f"Invalid date format in chunk: {exp_date}")
                        continue
            
            # Delete expired points
            if expired_ids:
                qdrant_client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=expired_ids
                )
                print(f"✅ Deleted {len(expired_ids)} expired chunks.")
                
                # Notify via WhatsApp
                clones = load_clones()
                for clone_id, chunks in chunks_by_clone.items():
                    phone_number = clones.get(clone_id, {}).get('phone_number')
                    if phone_number:
                        try:
                            msg = f"⏳ The following expired items were removed from your chatbot:\n"
                            for text in chunks[:5]:
                                msg += f"- {text}…\n"
                            if len(chunks) > 5:
                                msg += f"\n...and {len(chunks) - 5} more items."
                            send_whatsapp_message(phone_number, msg)
                        except Exception as e:
                            print(f"Error sending WhatsApp notification: {e}")
            else:
                print("✓ No expired chunks found.")
            
            retry_count = 0
            
        except socket.gaierror as e:
            retry_count += 1
            print(f"⚠️  Network error during expired chunk cleanup: {e}")
            print(f"   Retry {retry_count}/{max_retries} in 5 minutes...")
            if retry_count >= max_retries:
                print(f"❌ Failed after {max_retries} retries. Will try again in 15 minutes.")
                retry_count = 0
                time.sleep(900)
            else:
                time.sleep(300)
            continue
            
        except Exception as e:
            print(f"❌ Expired chunk cleanup error: {e}")
            import traceback
            traceback.print_exc()
        
        # Wait 15 minutes before next check
        time.sleep(900)
