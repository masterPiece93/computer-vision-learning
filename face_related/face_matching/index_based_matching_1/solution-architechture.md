
# Using GCS as Vector Store and Performing Vector Search

Using Google Cloud Storage (GCS) directly as a database for high-frequency search and write operations presents a major challenge: GCS is an object store, not a vector database.

If you store individual JSON/NPY files in GCS and scan all 10,000 files per search, network latency and read overhead will cause response times to degrade significantly $O(N)$ instead of remaining constant $O(1)$.

To achieve constant-time $O(1)$ response times while honoring the GCS restriction, you must separate persistence (GCS) from in-memory search runtime.

```
┌───────────────────────────────────────────────────────────┐
│                    IN-MEMORY RUNTIME                      │
│  (FastAPI / Flask Service / Cloud Run with Min Instances) │
└───────────────────────────┬───────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
    ┌───────────────────────┐ ┌───────────────────┐
    │   FAISS Index (ANNS)  │ │   ID Mapping      │
    │ [128-dim vectors]     │ │ {0: "user_101"}   │
    └───────────────────────┘ └───────────────────┘
                ▲                       ▲
                │ Sync / Backup         │ Sync / Backup
                └───────────┬───────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   GCS BUCKET (Store)                    │
│  - faiss_index.bin (Consolidated Index)                 │
│  - mapping.json    (Index ID -> User ID)                │
│  - encodings/      (User Raw Encodings Backup)          │
└─────────────────────────────────────────────────────────┘
```

## How It Works

1. Storage (GCS): Store individual user encodings as standard Numpy arrays (`.npy` files) under `gs://bucket/encodings/{user_id}.npy` for backup/raw access. Additionally, keep a consolidated FAISS Index file (`faiss_index.bin`) and a metadata mapping file (`id_map.json`) in the bucket.

2. Search Runtime (In-Memory Index): Load the `faiss_index.bin` into server RAM on startup. The face_recognition library generates 128-dimensional floating-point vectors. Searching 10,000 vectors of size 128 in-memory takes < 1 millisecond using L2 Euclidean Distance.

3. Frequent Writes (Dynamic Updates): When adding a new user, update the in-memory FAISS index instantly, and periodically flush/sync the updated index file back to GCS.

- Setup & Requirements
    ```bash
    pip install face_recognition faiss-cpu google-cloud-storage numpy
    ```

## Key Optimizations for Scaling to 10,000+ Encodings

- Constant Search Time ($O(1)$ / $O(\log N)$): Searching 10,000 128-dimensional float32 encodings via `faiss.IndexFlatL2` in memory takes less than 1 millisecond.

- Minimal GCS IO Overhead: Queries execute completely in memory without performing network API requests to GCS per search.

- Low Memory Footprint: 10,000 encodings of 128 floats require only ~5.1 MB of RAM, making it feasible to host on even the smallest Serverless Cloud instances (e.g., Google Cloud Run).

- Distributed Writes Handling: If you run multiple server instances writing concurrently, use a message queue (e.g., Google Cloud Pub/Sub) to handle index updates sequentially on a single writer service, or rebuild the index dynamically from the raw `gs://bucket/encodings/` files.