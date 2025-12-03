# terraform/environments/production/terraform.tfvars
# Production environment configuration

environment = "production"
aws_region  = "us-east-1"

# Networking
vpc_cidr = "10.0.0.0/16"

# Database
db_instance_class = "db.t3.small"  # Larger for production
db_name           = "healthcare_production"

# Compute
container_image  = "ghcr.io/your-org/healthcare/api:latest"
container_cpu    = 1024  # More resources for production
container_memory = 2048
desired_count    = 3     # More replicas for HA
