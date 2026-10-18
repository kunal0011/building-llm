# Terraform Provider and Versions Configuration
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "Universal-LLM-Data-Forge"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

resource "random_id" "stack_suffix" {
  byte_length = 4
}
