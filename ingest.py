import threading, time
from dataclasses import dataclass
from typing import Optional
from convert import SessionResult
from corpus import Corpus

@dataclass
class SyncStatus:
    state: str = "idle" # idle, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[SessionResult] = None
    message: str = ""

class IngestionCoordinator:
    def __init__(self, corpus_ref):
        self.corpus_ref = corpus_ref
        self._lock = threading.Lock()
        self.status = SyncStatus()

    def sync(self):
        if not self._lock.acquire(blocking=False):
            return False, "Sync already in progress"

        try:
            self.status = SyncStatus(state="running", start_time=time.time())

            # We need to run conversion.
            # Instead of calling main() which exits, we'll implement a function in convert.py that returns SessionResult.
            # For now, let's assume we can call a function.
            import convert

            # To avoid sys.exit in convert.main, we should probably wrap it or modify convert.py
            # Let's assume we modified convert.py to have a 'run_conversion()' function.
            res = convert.run_conversion()

            # Atomic reindex
            new_corpus = Corpus().load()
            if new_corpus.N > 0:
                self.corpus_ref[0] = new_corpus
                self.status = SyncStatus(
                    state="completed",
                    start_time=self.status.start_time,
                    end_time=time.time(),
                    result=res,
                    message="Sync completed and corpus updated"
                )
            else:
                self.status = SyncStatus(
                    state="failed",
                    start_time=self.status.start_time,
                    end_time=time.time(),
                    result=res,
                    message="Sync completed but no documents found"
                )
            return True, self.status.message
        except Exception as e:
            self.status = SyncStatus(
                state="failed",
                start_time=self.status.start_time,
                end_time=time.time(),
                message=str(e)
            )
            return False, str(e)
        finally:
            self._lock.release()

    def get_status(self):
        return self.status
