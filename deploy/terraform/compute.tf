# EC2 Compute Infrastructure (Ray Head Node & Spot Worker Auto Scaling Group)

# 1. Lookup Latest Ubuntu 22.04 LTS AMI
data "aws_ami" "ubuntu" {
  most_recent = true

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  owners = ["099720109477"] # Canonical
}

# 2. Ray Head Node User Data Script
locals {
  head_user_data = <<-EOF
    #!/bin/bash
    set -e

    # Kernel & File Descriptor Tuning for 100Gbps network & high open files
    sysctl -w net.core.somaxconn=65535
    sysctl -w fs.file-max=2097152
    sysctl -w vm.max_map_count=1048576
    echo "* soft nofile 1048576" >> /etc/security/limits.conf
    echo "* hard nofile 1048576" >> /etc/security/limits.conf

    # Install Python 3, pip, and Docker
    apt-get update -y
    apt-get install -y python3-pip python3-dev git curl jq build-essential

    # Install Ray and Pretraining Pipeline dependencies
    pip3 install --upgrade pip
    pip3 install "ray[default,data]>=2.35.0" boto3 botocore warcio trafilatura tokenizers numpy pyyaml fastapi uvicorn sympy

    # Configure Environment Variables
    echo "export AWS_DEFAULT_REGION=${var.aws_region}" >> /etc/profile.d/ray_env.sh
    echo "export S3_BUCKET_RAW=${aws_s3_bucket.raw_data.bucket}" >> /etc/profile.d/ray_env.sh
    echo "export S3_BUCKET_SHARDS=${aws_s3_bucket.shards_data.bucket}" >> /etc/profile.d/ray_env.sh

    # Start Ray Head Node
    ray stop --force || true
    ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

    echo "Ray Head Node initialized successfully."
  EOF
}

# 3. Ray Head Node (Public Subnet for Web Dashboard Access)
resource "aws_instance" "ray_head" {
  ami                  = data.aws_ami.ubuntu.id
  instance_type        = var.head_instance_type
  subnet_id            = aws_subnet.public[0].id
  iam_instance_profile = aws_iam_instance_profile.ray_cluster_profile.name
  vpc_security_group_ids = [aws_security_group.ray_cluster_sg.id]

  root_block_device {
    volume_size           = 300
    volume_type           = "gp3"
    iops                  = 6000
    throughput            = 500
    delete_on_termination = true
  }

  user_data = local.head_user_data

  tags = {
    Name = "ray-head-node-${var.environment}"
    Role = "RayHead"
  }
}

# 4. Ray Worker Launch Template
locals {
  worker_user_data = <<-EOF
    #!/bin/bash
    set -e

    # Kernel & File Descriptor Tuning
    sysctl -w net.core.somaxconn=65535
    sysctl -w fs.file-max=2097152
    sysctl -w vm.max_map_count=1048576
    echo "* soft nofile 1048576" >> /etc/security/limits.conf
    echo "* hard nofile 1048576" >> /etc/security/limits.conf

    # Install Python & Dependencies
    apt-get update -y
    apt-get install -y python3-pip python3-dev git curl jq build-essential
    pip3 install --upgrade pip
    pip3 install "ray[default,data]>=2.35.0" boto3 botocore warcio trafilatura tokenizers numpy pyyaml sympy

    # Wait for Head node GCS to be reachable
    HEAD_IP="${aws_instance.ray_head.private_ip}"
    until nc -z -v -w5 $HEAD_IP 6379; do
      echo "Waiting for Ray Head at $HEAD_IP:6379..."
      sleep 2
    done

    # Start Ray Worker connected to Head
    ray stop --force || true
    ray start --address=$HEAD_IP:6379 --block
  EOF
}

resource "aws_launch_template" "ray_worker_lt" {
  name_prefix   = "ray-worker-lt-${var.environment}-"
  image_id      = data.aws_ami.ubuntu.id
  instance_type = var.worker_instance_type

  iam_instance_profile {
    name = aws_iam_instance_profile.ray_cluster_profile.name
  }

  vpc_security_group_ids = [aws_security_group.ray_cluster_sg.id]

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size           = 250
      volume_type           = "gp3"
      iops                  = 6000
      throughput            = 500
      delete_on_termination = true
    }
  }

  user_data = base64encode(local.worker_user_data)

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "ray-worker-compute"
      Role = "RayWorker"
    }
  }
}

# 5. Ray Worker Auto Scaling Group (Private Subnets with Spot Instance Mix)
resource "aws_autoscaling_group" "ray_workers" {
  name_prefix         = "ray-workers-asg-${var.environment}-"
  vpc_zone_identifier = aws_subnet.private[*].id
  min_size            = var.min_workers
  max_size            = var.max_workers
  desired_capacity    = var.desired_workers

  mixed_instances_policy {
    instances_distribution {
      on_demand_base_capacity                  = 0
      on_demand_percentage_above_base_capacity = var.use_spot_workers ? 10 : 100
      spot_allocation_strategy                 = "price-capacity-optimized"
    }

    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.ray_worker_lt.id
        version            = "$Latest"
      }

      override {
        instance_type = var.worker_instance_type
      }
      override {
        instance_type = "c6i.8xlarge"
      }
      override {
        instance_type = "c6a.16xlarge"
      }
    }
  }

  tag {
    key                 = "Name"
    value               = "ray-worker-node"
    propagate_at_launch = true
  }

  tag {
    key                 = "Project"
    value               = "Universal-LLM-Data-Forge"
    propagate_at_launch = true
  }

  depends_on = [aws_instance.ray_head]
}
