# terraform/environments/staging/terraform.tfvars
# Staging environment configuration

environment = "staging"
aws_region  = "us-east-1"

# Networking
vpc_cidr = "10.1.0.0/16"

# Database
db_instance_class = "db.t3.micro"
db_name           = "healthcare_staging"

# Compute
container_image  = "ghcr.io/your-org/healthcare/api:staging"
container_cpu    = 256
container_memory = 512
desired_count    = 2
