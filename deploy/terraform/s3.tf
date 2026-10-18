# S3 Data Lake Infrastructure (Raw, Intermediate, and Shards Buckets)

# 1. Raw Data Lake (Streaming WARCs, Repo Archives, Books)
resource "aws_s3_bucket" "raw_data" {
  bucket        = "ai-training-data-raw-${var.environment}-${random_id.stack_suffix.hex}"
  force_destroy = false

  tags = {
    Name = "ai-training-data-raw"
    Tier = "RawCorpora"
  }
}

# 2. Intermediate Filtered Data (Decontaminated JSONL, Deduplication Indices)
resource "aws_s3_bucket" "intermediate_data" {
  bucket        = "ai-training-data-interm-${var.environment}-${random_id.stack_suffix.hex}"
  force_destroy = false

  tags = {
    Name = "ai-training-data-intermediate"
    Tier = "FilteredJSONL"
  }
}

# 3. Final Shards & Multimodal Lake (Megatron .bin/.idx, Images, RL Experience)
resource "aws_s3_bucket" "shards_data" {
  bucket        = "ai-training-data-shards-${var.environment}-${random_id.stack_suffix.hex}"
  force_destroy = false

  tags = {
    Name = "ai-training-data-shards"
    Tier = "BinaryShardsAndRL"
  }
}

# Server-Side Encryption on all buckets
resource "aws_s3_bucket_server_side_encryption_configuration" "raw_enc" {
  bucket = aws_s3_bucket.raw_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "interm_enc" {
  bucket = aws_s3_bucket.intermediate_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "shards_enc" {
  bucket = aws_s3_bucket.shards_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block Public Access on all buckets
resource "aws_s3_bucket_public_access_block" "raw_pab" {
  bucket                  = aws_s3_bucket.raw_data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "interm_pab" {
  bucket                  = aws_s3_bucket.intermediate_data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "shards_pab" {
  bucket                  = aws_s3_bucket.shards_data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle Management: Abort incomplete multipart uploads and transition raw data
resource "aws_s3_bucket_lifecycle_configuration" "raw_lifecycle" {
  bucket = aws_s3_bucket.raw_data.id

  rule {
    id     = "abort-multipart-and-glacier"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "shards_lifecycle" {
  bucket = aws_s3_bucket.shards_data.id

  rule {
    id     = "abort-multipart"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# CORS Rule for Shards Bucket (Allows web visualizer to stream token arrays / manifests)
resource "aws_s3_bucket_cors_configuration" "shards_cors" {
  bucket = aws_s3_bucket.shards_data.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "HEAD"]
    allowed_origins = ["*"]
    max_age_seconds = 3000
  }
}
