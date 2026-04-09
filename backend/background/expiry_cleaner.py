import socket
import time
from contextlib import closing
from datetime import datetime

from backend.config import Config
from backend.core.extensions import qdrant_client
from backend.repositories.clone_repo import load_clones
from backend.services.whatsapp_service import send_whatsapp_message

COLLECTION_NAME = Config.COLLECTION_NAME


def delete_expired_chunks_job():
    """Background job to delete expired chunks with proper error handling."""
    retry_count = 0
    max_retries = 3

    while True:
        try:
            try:
                with closing(socket.create_connection(("8.8.8.8", 53), timeout=5)):
                    pass
            except OSError:
                print("No internet connection. Skipping expired chunk cleanup.")
                time.sleep(900)  # Wait 15 min and retry
                continue

            now = datetime.now().date()
            expired_ids = []
            chunks_by_clone = {}

            # Scroll all points using pagination.
            next_offset = None
            while True:
                points, next_offset = qdrant_client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=None,
                    with_payload=True,
                    with_vectors=False,
                    limit=200,
                    offset=next_offset,
                )

                if not points:
                    break

                for point in points:
                    payload = point.payload or {}
                    exp_date = payload.get("expiry_date")
                    clone_id = payload.get("clone_id")

                    if not exp_date or exp_date == "PERMANENT":
                        continue

                    try:
                        expiry_datetime = datetime.strptime(exp_date, "%Y-%m-%d").date()
                    except ValueError:
                        print(f"Invalid date format in chunk: {exp_date}")
                        continue

                    if expiry_datetime < now:
                        expired_ids.append(point.id)
                        chunks_by_clone.setdefault(clone_id, []).append(
                            payload.get("text", "")[:80]
                        )

                if next_offset is None:
                    break

            if expired_ids:
                qdrant_client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=expired_ids,
                )
                print(f"Deleted {len(expired_ids)} expired chunks.")

                clones = load_clones()
                for clone_id, chunks in chunks_by_clone.items():
                    phone_number = clones.get(clone_id, {}).get("phone_number")
                    if not phone_number:
                        continue
                    try:
                        msg = "The following expired items were removed from your chatbot:\n"
                        for text in chunks[:5]:
                            msg += f"- {text}...\n"
                        if len(chunks) > 5:
                            msg += f"\n...and {len(chunks) - 5} more items."
                        send_whatsapp_message(phone_number, msg)
                    except Exception as err:
                        print(f"Error sending WhatsApp notification: {err}")
            else:
                print("No expired chunks found.")

            retry_count = 0

        except socket.gaierror as err:
            retry_count += 1
            print(f"Network error during expired chunk cleanup: {err}")
            print(f"Retry {retry_count}/{max_retries} in 5 minutes...")
            if retry_count >= max_retries:
                print(f"Failed after {max_retries} retries. Will try again in 15 minutes.")
                retry_count = 0
                time.sleep(900)
            else:
                time.sleep(300)
            continue

        except Exception as err:
            print(f"Expired chunk cleanup error: {err}")
            import traceback

            traceback.print_exc()

        # Wait 15 minutes before next check
        time.sleep(900)
