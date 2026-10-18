"""
High-Throughput AWS S3 Storage and Checkpoint Manager for LLM Training Data.
Supports parallel multipart uploads, shard streaming, and distributed state persistence.
"""

import os
from typing import Optional, List, Dict, Any
import boto3
from botocore.exceptions import ClientError
from botocore import UNSIGNED
from botocore.config import Config


class S3DataManager:
    """
    Manages high-throughput object storage operations for raw WARCs,
    intermediate JSONL corpora, and final binary memory-mapped shards.
    """

    def __init__(
        self,
        bucket_name: str = "ai-training-data-lake",
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None,
        unsigned_access: bool = False
    ):
        self.bucket_name = bucket_name
        self.region = region
        self.endpoint_url = endpoint_url or os.getenv("AWS_ENDPOINT_URL") or os.getenv("S3_ENDPOINT_URL")
        aws_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")

        client_kwargs: Dict[str, Any] = {"region_name": region}
        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url
            if not aws_key or not aws_secret:
                aws_key = "test"
                aws_secret = "test"
        if aws_key and aws_secret:
            client_kwargs["aws_access_key_id"] = aws_key
            client_kwargs["aws_secret_access_key"] = aws_secret

        if unsigned_access:
            # Used for accessing open datasets like Common Crawl's public S3 bucket
            client_kwargs["config"] = Config(signature_version=UNSIGNED)

        self.s3_client = boto3.client("s3", **client_kwargs)

    def ensure_bucket_exists(self) -> bool:
        """Creates the bucket if it does not already exist."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return True
        except ClientError:
            try:
                if self.region == "us-east-1":
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": self.region}
                    )
                return True
            except ClientError as e:
                print(f"[ERROR] Failed to create bucket {self.bucket_name}: {e}")
                return False

    def upload_file_multipart(
        self,
        local_file_path: str,
        s3_key: str,
        extra_args: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Uploads large binary shards or archives using boto3 multipart transfers.
        """
        config = boto3.s3.transfer.TransferConfig(
            multipart_threshold=64 * 1024 * 1024,  # 64 MB
            max_concurrency=10,
            multipart_chunksize=32 * 1024 * 1024,  # 32 MB
            use_threads=True
        )
        try:
            self.s3_client.upload_file(
                Filename=local_file_path,
                Bucket=self.bucket_name,
                Key=s3_key,
                Config=config,
                ExtraArgs=extra_args
            )
            return True
        except ClientError as e:
            print(f"[ERROR] Failed to upload {local_file_path} to s3://{self.bucket_name}/{s3_key}: {e}")
            return False

    def list_warc_dumps(self, prefix: str = "crawl-data/CC-MAIN-", max_keys: int = 100) -> List[str]:
        """
        Lists available Common Crawl segment WARCs in the bucket.
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            return [item["Key"] for item in response.get("Contents", [])]
        except ClientError as e:
            print(f"[ERROR] Failed listing objects: {e}")
            return []

    def save_checkpoint(self, checkpoint_key: str, processed_sources: List[str]) -> bool:
        """
        Persists list of completed data chunks to prevent duplicate worker processing.
        """
        payload = "\n".join(processed_sources).encode("utf-8")
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=checkpoint_key,
                Body=payload
            )
            return True
        except ClientError as e:
            print(f"[ERROR] Checkpoint save failed: {e}")
            return False

    def upload_bytes(self, data: bytes, s3_key: str) -> bool:
        """Uploads raw bytes directly to an S3 key."""
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=data
            )
            return True
        except ClientError as e:
            print(f"[ERROR] Failed to upload bytes to s3://{self.bucket_name}/{s3_key}: {e}")
            return False

    def download_file(self, s3_key: str, local_destination_path: str) -> bool:
        """Downloads an object from S3 to a local file."""
        try:
            os.makedirs(os.path.dirname(local_destination_path), exist_ok=True)
            self.s3_client.download_file(
                Bucket=self.bucket_name,
                Key=s3_key,
                Filename=local_destination_path
            )
            return True
        except ClientError as e:
            print(f"[ERROR] Failed to download s3://{self.bucket_name}/{s3_key}: {e}")
            return False

    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> List[Dict[str, Any]]:
        """Lists objects with metadata under a prefix."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            return [
                {
                    "key": item["Key"],
                    "size": item["Size"],
                    "last_modified": item["LastModified"].isoformat()
                }
                for item in response.get("Contents", [])
            ]
        except ClientError as e:
            print(f"[ERROR] Failed listing objects in s3://{self.bucket_name}/{prefix}: {e}")
            return []

    def load_checkpoint(self, checkpoint_key: str) -> List[str]:
        """
        Retrieves processed chunks from S3 checkpoint state.
        """
        try:
            resp = self.s3_client.get_object(Bucket=self.bucket_name, Key=checkpoint_key)
            lines = resp["Body"].read().decode("utf-8").splitlines()
            return [line.strip() for line in lines if line.strip()]
        except ClientError:
            return []
