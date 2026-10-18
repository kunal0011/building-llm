# Security Groups for Distributed Ray Cluster & Visualizer Web Dashboard

resource "aws_security_group" "ray_cluster_sg" {
  name        = "ray-cluster-sg-${var.environment}-${random_id.stack_suffix.hex}"
  description = "Controls inter-node communication for Ray cluster and web dashboards"
  vpc_id      = aws_vpc.cluster_vpc.id

  # 1. Intra-Cluster Communication (All traffic allowed between Head and Worker nodes)
  ingress {
    description = "Inter-node cluster communication (GCS 6379, client 10001, object store, and worker ports)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }

  # 2. Ray Web Dashboard (Port 8265)
  ingress {
    description = "Ray Cluster Telemetry Dashboard"
    from_port   = 8265
    to_port     = 8265
    protocol    = "tcp"
    cidr_blocks = var.allowed_admin_cidrs
  }

  # 3. Data Forge Visual Pipeline Dashboard (Port 3000)
  ingress {
    description = "Data Forge Visual Pipeline Web Interface"
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = var.allowed_admin_cidrs
  }

  # 4. HTTPS & HTTP (Ports 443 & 80)
  ingress {
    description = "HTTPS Web Traffic"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = var.allowed_admin_cidrs
  }

  ingress {
    description = "HTTP Web Traffic"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.allowed_admin_cidrs
  }

  # 5. Outbound Access (Full internet access for streaming Common Crawl WARCs & S3)
  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "ray-cluster-sg"
  }
}
