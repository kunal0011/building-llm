# IAM Roles and Policies for Ray Cluster Multi-Node EC2 Instances

# 1. EC2 Assume Role Policy
data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    effect  = "Allow"

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

# 2. Ray Cluster IAM Role
resource "aws_iam_role" "ray_cluster_role" {
  name               = "RayClusterEC2Role-${var.environment}-${random_id.stack_suffix.hex}"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json

  tags = {
    Name = "RayClusterRole"
  }
}

# 3. Fine-Grained S3 Access Policy
data "aws_iam_policy_document" "s3_access" {
  statement {
    effect = "Allow"
    actions = [
      "s3:ListBucket",
      "s3:GetBucketLocation",
      "s3:ListBucketMultipartUploads"
    ]
    resources = [
      aws_s3_bucket.raw_data.arn,
      aws_s3_bucket.intermediate_data.arn,
      aws_s3_bucket.shards_data.arn
    ]
  }

  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:AbortMultipartUpload",
      "s3:ListMultipartUploadParts"
    ]
    resources = [
      "${aws_s3_bucket.raw_data.arn}/*",
      "${aws_s3_bucket.intermediate_data.arn}/*",
      "${aws_s3_bucket.shards_data.arn}/*"
    ]
  }

  # Allow reading open public Common Crawl S3 bucket
  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:ListBucket"
    ]
    resources = [
      "arn:aws:s3:::commoncrawl",
      "arn:aws:s3:::commoncrawl/*"
    ]
  }
}

resource "aws_iam_policy" "s3_data_policy" {
  name        = "RayClusterS3DataPolicy-${var.environment}-${random_id.stack_suffix.hex}"
  description = "Provides read/write access to LLM training data buckets"
  policy      = data.aws_iam_policy_document.s3_access.json
}

resource "aws_iam_role_policy_attachment" "attach_s3" {
  role       = aws_iam_role.ray_cluster_role.name
  policy_arn = aws_iam_policy.s3_data_policy.arn
}

# 4. Attach AWS Systems Manager (SSM) for Zero-SSH Key Terminal Access
resource "aws_iam_role_policy_attachment" "attach_ssm" {
  role       = aws_iam_role.ray_cluster_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# 5. Attach CloudWatch Agent Policy
resource "aws_iam_role_policy_attachment" "attach_cloudwatch" {
  role       = aws_iam_role.ray_cluster_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

# 6. Instance Profile
resource "aws_iam_instance_profile" "ray_cluster_profile" {
  name = "RayClusterInstanceProfile-${var.environment}-${random_id.stack_suffix.hex}"
  role = aws_iam_role.ray_cluster_role.name
}
