"""
Local S3 Emulator Service with Auto-Created Foundation Buckets.
Runs a 100% AWS S3-compatible service on port 4566 using Moto.
"""

import time
import subprocess
import sys
import boto3


def start_s3_service(host: str = "0.0.0.0", port: int = 4566):
    print("=" * 60)
    print(f"[*] Starting Local S3 Emulator on http://{host}:{port}...")
    print("=" * 60)

    proc = subprocess.Popen(["moto_server", "-H", host, "-p", str(port)])
    time.sleep(2)

    endpoint_url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}"
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1"
    )

    # Auto-initialize foundation buckets
    buckets = ["ai-data-raw", "ai-data-interm", "ai-data-shards"]
    for b in buckets:
        try:
            s3.create_bucket(Bucket=b)
            print(f"[+] Initialized S3 bucket: s3://{b}/")
        except Exception as e:
            print(f"[-] Bucket {b} check: {e}")

    print(f"\n[✓] Local S3 Service is ready and listening on port {port}.")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    start_s3_service()
