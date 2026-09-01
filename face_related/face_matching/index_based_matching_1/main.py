import io
import pathlib
import json
import faiss
import numpy as np
# from google.cloud import storage

_BUCKETS = pathlib.Path(__file__).parent / "buckets"
_TEST_IMAGES = pathlib.Path(__file__).parent / "test_images"

class LocalFaceStore:
    """
    LocalFaceStore manages a FAISS index for face encodings and persists it to local storage.
    It allows adding new face encodings, searching for matches, and syncing the index with local files.
    """
    def __init__(self, bucket_name: str, dimension: int = 128):
        self.bucket_name = bucket_name
        self.dimension = dimension
        self.bucket = _BUCKETS / bucket_name

        # FAISS Index using Flat L2 distance (matches face_recognition logic)
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Mapping FAISS integer IDs to User String IDs
        self.id_to_user = {}
        self.user_to_id = {}
        
        # Initialize or download from local storage
        self._load_from_local()

    def _load_from_local(self):
        """Loads index and mappings from local storage on application start."""
        index_path = self.bucket / "faiss_index.bin"
        map_path = self.bucket / "id_map.json"

        if index_path.exists() and map_path.exists():
            # Load Index
            index_bytes = index_path.read_bytes()
            self.index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
            
            # Load ID Mapping
            mapping_data = json.loads(map_path.read_text())
            self.id_to_user = {int(k): v for k, v in mapping_data["id_to_user"].items()}
            self.user_to_id = mapping_data["user_to_id"]
            print("Loaded existing FAISS index from local storage.")
        else:
            print("No existing index found in local storage. Initialized clean index.")

    def sync_to_local(self):
        """Persists the current in-memory index back to local storage."""
        # Serialize FAISS index
        chunk = faiss.serialize_index(self.index)
        index_path = self.bucket / "faiss_index.bin"
        index_path.write_bytes(chunk.tobytes())

        # Save ID mapping
        map_path = self.bucket / "id_map.json"
        map_path.write_text(
            json.dumps({"id_to_user": self.id_to_user, "user_to_id": self.user_to_id}),
            encoding="utf-8"
        )

    def add_user_encoding(self, user_id: str, encoding: np.ndarray):
        """Stores individual raw array to GCS and updates the FAISS index."""
        # 1. Format encoding to float32
        vec = np.array([encoding], dtype=np.float32)
        
        # 2. Store raw backup in local storage (simulating GCS)
        buffer = io.BytesIO()
        np.save(buffer, encoding)

        if not (self.bucket / "encodings").exists():
            (self.bucket / "encodings").mkdir(parents=True, exist_ok=True)
        
        with open(self.bucket / f"encodings/{user_id}.npy", "wb+") as f:
            f.write(buffer.getvalue())

        # 3. Add to FAISS in-memory index
        faiss_id = self.index.ntotal
        self.index.add(vec)
        
        # 4. Save metadata relationships
        self.id_to_user[faiss_id] = user_id
        self.user_to_id[user_id] = faiss_id

        # Sync index changes back to local storage
        self.sync_to_local()

    def search_face(self, target_encoding: np.ndarray, threshold: float = 0.6):
        """
        Searches target encoding against the index.
        Returns matched user_id if distance <= threshold (lower means closer match).
        """
        if self.index.ntotal == 0:
            return None, None

        # Format vector
        vec = np.array([target_encoding], dtype=np.float32)
        
        # Search Top 1 nearest neighbor
        distances, indices = self.index.search(vec, k=1)
        
        best_distance = distances[0][0]
        best_index = indices[0][0]

        # Convert squared Euclidean distance to standard Euclidean distance
        # face_recognition library default match threshold is typically 0.6 distance
        euclidean_distance = np.sqrt(best_distance)

        if euclidean_distance <= threshold and best_index in self.id_to_user:
            matched_user_id = self.id_to_user[best_index]
            return matched_user_id, float(euclidean_distance)
        
        return None, float(euclidean_distance)

    def search_all_faces(self, target_encoding: np.ndarray, threshold: float = 0.6):
        """
        Searches target encoding against the index.
        Returns all matched user_ids with distances <= threshold.
        """
        if self.index.ntotal == 0:
            return []

        # Format vector
        vec = np.array([target_encoding], dtype=np.float32)
        
        # Search all neighbors
        distances, indices = self.index.search(vec, k=self.index.ntotal)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            euclidean_distance = np.sqrt(dist)
            if euclidean_distance <= threshold and idx in self.id_to_user:
                matched_user_id = self.id_to_user[idx]
                results.append((matched_user_id, float(euclidean_distance)))
        
        return results
    
if __name__ == "__main__":
    import face_recognition

    # Initialize store (pulls latest index from GCS)
    face_store = LocalFaceStore(bucket_name="face_store_bucket", dimension=128)

    __supported_image_formats = [".jpg", ".jpeg", ".png"]

    # 1. ADD / REGISTER USER
    for file in _TEST_IMAGES.glob("*"):
        if file.suffix.lower() not in __supported_image_formats:
            continue

        image_to_register = face_recognition.load_image_file(file)
        target_encoding = face_recognition.face_encodings(image_to_register)[0]
        # add user encoding to FAISS index and local storage
        face_store.add_user_encoding(user_id=file.stem, encoding=target_encoding)


    # 2. SEARCH / RECOGNIZE USER
    query_image = face_recognition.load_image_file("unknown_target.jpeg")
    target_encoding = face_recognition.face_encodings(query_image)[0]

    # Search in O(1) time
    matched_user_id, distance = face_store.search_face(target_encoding, threshold=0.6)

    if matched_user_id:
        print(f"Match found! User ID: {matched_user_id} (Distance: {distance:.4f})")
    else:
        print("No matching user found.")

    # 3. SEARCH ALL FACES
    all_matches = face_store.search_all_faces(target_encoding, threshold=0.6)
    if all_matches:
        print("All matching users found:")
        for user_id, dist in all_matches:
            print(f" - User ID: {user_id} (Distance: {dist:.4f})")
    else:
        print("No matching users found.")