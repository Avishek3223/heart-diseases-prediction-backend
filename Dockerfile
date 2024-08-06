# Use a specific Python version
FROM python:3.8

# Set the working directory
WORKDIR /functions

# Copy requirements file
COPY requirements_clean.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements_clean.txt

# Copy the rest of your application code
COPY . .

# Command to run the application (adjust as needed)
CMD ["functions/predict.handler"]
