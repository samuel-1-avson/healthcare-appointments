# terraform/variables.tf
# Variable definitions for infrastructure

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "healthcare"
}

variable "container_image" {
  description = "Docker container image for the application"
  type        = string
}

variable "container_cpu" {
  description = "CPU units for ECS task"
  type        = number
  default     = 512
}

variable "container_memory" {
  description = "Memory for ECS task (MB)"
  type        = number
  default     = 1024
}

variable "desired_count" {
  description = "Desired number of ECS tasks"
  type        = number
  default     = 2
}
