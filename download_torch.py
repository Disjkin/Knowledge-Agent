"""多线程分块下载 torch cu126 wheel（利用 HTTP Range 并行加速）。"""
import concurrent.futures
import os
import sys
import threading
import urllib.request

URL = "https://mirrors.aliyun.com/pytorch-wheels/cu126/torch-2.13.0%2Bcu126-cp311-cp311-win_amd64.whl"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_cu126.whl")
NUM_THREADS = 8
CHUNK_SIZE = 32 * 1024 * 1024  # 32MB per chunk

sys.stdout.reconfigure(encoding='utf-8')
progress_lock = threading.Lock()
done_chunks = 0
total_chunks = 0


def get_content_length():
    req = urllib.request.Request(URL, method='HEAD')
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers['Content-Length'])


def download_chunk(start, end, index, total):
    global done_chunks
    req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    with open(OUT, 'r+b') as f:
        f.seek(start)
        f.write(data)
    with progress_lock:
        done_chunks += 1
        pct = done_chunks / total * 100
        print(f"[{done_chunks}/{total}] {pct:.0f}% (chunk {index} {len(data)/1024/1024:.0f}MB)", flush=True)


def main():
    global total_chunks
    size = get_content_length()
    print(f"Total size: {size/1024/1024:.0f} MB", flush=True)

    chunks = list(range(0, size, CHUNK_SIZE))
    total_chunks = len(chunks)
    # 预分配文件
    with open(OUT, 'wb') as f:
        f.truncate(size)

    ranges = []
    for i, start in enumerate(chunks):
        end = min(start + CHUNK_SIZE - 1, size - 1)
        ranges.append((start, end, i))

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
        futures = [
            pool.submit(download_chunk, start, end, i, total_chunks)
            for start, end, i in ranges
        ]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    print(f"Download complete: {OUT}", flush=True)


if __name__ == '__main__':
    main()
