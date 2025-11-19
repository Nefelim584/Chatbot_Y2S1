import kagglehub

# Download latest version
path = kagglehub.dataset_download("arshkon/linkedin-job-postings")

print("Path to dataset files:", path)