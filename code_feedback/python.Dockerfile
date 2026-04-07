# Use a minimal, secure Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the student's code into the container.
# We will do this dynamically from our Python script,
# but we can add a placeholder COPY command here for clarity.
# COPY . .

# The command to run the student's code.
# The actual file to run will be provided by our script.
# CMD ["python", "main.py"]