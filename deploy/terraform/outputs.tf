# Terraform Output Values for Infrastructure Connection & Cluster Orchestration

output "vpc_id" {
  description = "VPC ID where the cluster is deployed"
  value       = aws_vpc.cluster_vpc.id
}

output "s3_bucket_raw" {
  description = "S3 Bucket for raw Common Crawl WARC dumps and code archives"
  value       = aws_s3_bucket.raw_data.bucket
}

output "s3_bucket_intermediate" {
  description = "S3 Bucket for intermediate filtered and decontaminated corpora"
  value       = aws_s3_bucket.intermediate_data.bucket
}

output "s3_bucket_shards" {
  description = "S3 Bucket for final Megatron-LM memory-mapped binary shards, images, and RL experience"
  value       = aws_s3_bucket.shards_data.bucket
}

output "ray_head_public_ip" {
  description = "Public IP of the Ray Head Node"
  value       = aws_instance.ray_head.public_ip
}

output "ray_head_private_ip" {
  description = "Private IP of the Ray Head Node (used for worker clustering)"
  value       = aws_instance.ray_head.private_ip
}

output "ray_dashboard_url" {
  description = "Web URL for Ray Cluster Telemetry Dashboard"
  value       = "http://${aws_instance.ray_head.public_ip}:8265"
}

output "visualizer_url" {
  description = "Web URL for Data Forge Visual Pipeline Dashboard"
  value       = "http://${aws_instance.ray_head.public_ip}:3000"
}

output "ray_job_submit_command" {
  description = "Command to submit distributed pipeline jobs to the AWS cluster"
  value       = "ray job submit --address http://${aws_instance.ray_head.public_ip}:8265 -- python3 pipelines/ray_distributed_pipeline.py"
}
