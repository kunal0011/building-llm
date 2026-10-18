# High-Performance VPC & Multi-AZ Network Topology

data "aws_availability_zones" "available" {
  state = "available"
}

# 1. Dedicated VPC
resource "aws_vpc" "cluster_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "llm-data-cluster-vpc-${var.environment}"
  }
}

# 2. Internet Gateway
resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.cluster_vpc.id

  tags = {
    Name = "llm-data-igw-${var.environment}"
  }
}

# 3. Public Subnets (Head Node & NAT Gateway)
resource "aws_subnet" "public" {
  count                   = length(var.public_subnet_cidrs)
  vpc_id                  = aws_vpc.cluster_vpc.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "llm-data-public-subnet-${count.index + 1}-${var.environment}"
    Type = "Public"
  }
}

# 4. Private Subnets (Worker Compute Nodes)
resource "aws_subnet" "private" {
  count             = length(var.private_subnet_cidrs)
  vpc_id            = aws_vpc.cluster_vpc.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "llm-data-private-subnet-${count.index + 1}-${var.environment}"
    Type = "Private"
  }
}

# 5. Elastic IP & NAT Gateway (for Private Subnets Outbound Internet)
resource "aws_eip" "nat" {
  domain     = "vpc"
  depends_on = [aws_internet_gateway.igw]

  tags = {
    Name = "llm-data-nat-eip-${var.environment}"
  }
}

resource "aws_nat_gateway" "nat" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id

  tags = {
    Name = "llm-data-nat-gateway-${var.environment}"
  }

  depends_on = [aws_internet_gateway.igw]
}

# 6. Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.cluster_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = {
    Name = "llm-data-public-rt-${var.environment}"
  }
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.cluster_vpc.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat.id
  }

  tags = {
    Name = "llm-data-private-rt-${var.environment}"
  }
}

# 7. Subnet Route Table Associations
resource "aws_route_table_association" "public" {
  count          = length(var.public_subnet_cidrs)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = length(var.private_subnet_cidrs)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# 8. S3 Gateway Endpoint (Zero-cost, line-rate private transfer directly to S3)
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.cluster_vpc.id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.public.id, aws_route_table.private.id]

  tags = {
    Name = "llm-data-s3-gateway-endpoint-${var.environment}"
  }
}
