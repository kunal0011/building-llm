# Terraform Input Variables for LLM Data Infrastructure

variable "aws_region" {
  type        = string
  description = "AWS Region for deployment (e.g. us-east-1, us-west-2)"
  default     = "us-east-1"
}

variable "environment" {
  type        = string
  description = "Deployment environment name (e.g. dev, staging, production)"
  default     = "production"
}

variable "vpc_cidr" {
  type        = string
  description = "CIDR block for the dedicated cluster VPC"
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  type        = list(string)
  description = "CIDR blocks for public subnets across availability zones"
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnet_cidrs" {
  type        = list(string)
  description = "CIDR blocks for private subnets across availability zones"
  default     = ["10.0.10.0/24", "10.0.20.0/24"]
}

variable "head_instance_type" {
  type        = string
  description = "EC2 Instance type for Ray Head node (coordinates cluster state, GCS, and Dashboard)"
  default     = "m6i.4xlarge" # 16 vCPUs, 64 GB RAM
}

variable "worker_instance_type" {
  type        = string
  description = "EC2 Instance type for Ray Worker compute nodes (high-throughput filtering & tokenization)"
  default     = "c6i.16xlarge" # 64 vCPUs, 128 GB RAM
}

variable "min_workers" {
  type        = number
  description = "Minimum number of worker instances in Auto Scaling Group"
  default     = 2
}

variable "max_workers" {
  type        = number
  description = "Maximum number of worker instances in Auto Scaling Group"
  default     = 32
}

variable "desired_workers" {
  type        = number
  description = "Initial desired worker instance count"
  default     = 4
}

variable "use_spot_workers" {
  type        = bool
  description = "Enable EC2 Spot Instances for worker nodes (saves 70-80% on compute cost)"
  default     = true
}

variable "allowed_admin_cidrs" {
  type        = list(string)
  description = "CIDR blocks allowed to access the Ray Dashboard and Visualizer web ports"
  default     = ["0.0.0.0/0"]
}
