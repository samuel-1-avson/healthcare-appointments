# Terraform Infrastructure

This directory contains Infrastructure as Code (IaC) for deploying the Healthcare No-Show Prediction System to AWS.

## Structure

```
terraform/
├── main.tf                    # Root configuration
├── variables.tf               # Variable definitions
├── modules/                   # Reusable modules
│   ├── networking/           # VPC, subnets, routing
│   ├── database/             # RDS PostgreSQL
│   ├── compute/              # ECS/Fargate
│   ├── storage/              # S3 buckets
│   └── monitoring/           # CloudWatch, alarms
└── environments/             # Environment-specific configs
    ├── staging/
    │   └── terraform.tfvars
    └── production/
        └── terraform.tfvars
```

## Prerequisites

1. **AWS CLI** configured with appropriate credentials
2. **Terraform** >= 1.5.0 installed
3. **S3 bucket** for Terraform state (create manually first)

## Setup

### 1. Create S3 Backend for State

```bash
aws s3 mb s3://healthcare-terraform-state --region us-east-1
aws s3api put-bucket-versioning \
  --bucket healthcare-terraform-state \
  --versioning-configuration Status=Enabled
```

### 2. Create DynamoDB Table for State Locking

```bash
aws dynamodb create-table \
  --table-name terraform-state-lock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
```

### 3. Initialize Terraform

```bash
cd terraform
terraform init
```

## Deployment

### Staging Environment

```bash
# Plan
terraform plan -var-file=environments/staging/terraform.tfvars

# Apply
terraform apply -var-file=environments/staging/terraform.tfvars
```

### Production Environment

```bash
# Plan
terraform plan -var-file=environments/production/terraform.tfvars

# Apply
terraform apply -var-file=environments/production/terraform.tfvars
```

## Outputs

After successful deployment, Terraform will output:
- **load_balancer_dns**: Public DNS for accessing the application
- **database_endpoint**: RDS endpoint (sensitive)

## Destroy

To tear down infrastructure:

```bash
terraform destroy -var-file=environments/staging/terraform.tfvars
```

## State Management

Terraform state is stored in S3 with DynamoDB locking for safe concurrent operations.

**⚠️ Never commit `terraform.tfstate` files to git!**
